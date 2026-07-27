"""
Single source of truth for per-session state.

The DB is authoritative for anything that must survive a restart: in-progress
lead-collection fields (LeadSession), a completed lead's info (Lead), and the
feedback ask/declined/given latch (FeedbackState). The in-memory dicts kept
in this module are only a same-process fast cache in front of those tables -
never the source of truth. A cache miss (fresh session, session predates this
process, multi-worker deployment) reads through to the DB before falling
back to a blank default, and every write goes to memory + DB together so a
restart never loses more than the single in-flight turn.

This generalizes a pattern that used to be reimplemented ad hoc in
lead_agent.py (three times over: in-progress state, completed-lead recall,
and the is_collecting flag) and was missing entirely from feedback_agent.py
(so a restart mid-feedback-flow used to silently forget it had already
asked). lead_agent.py and feedback_agent.py no longer keep their own
module-level dicts/locks/TTL-sweep - they call the functions here. See
SESSION_STATE.md for the full flow.
"""
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Generic, TypeVar

from sqlalchemy.orm import Session

from app.core.config import SESSION_TTL_SECONDS
from app.database import (
    SessionLocal,
    delete_feedback_state,
    delete_lead_session,
    delete_stale_feedback_states,
    delete_stale_lead_sessions,
    get_latest_lead_by_session,
    load_feedback_state,
    load_lead_session,
    save_feedback_state,
    save_lead_session,
)

logger = logging.getLogger("xqora.session_service")

LEAD_FIELDS = ["name", "company", "email", "phone", "message"]

_SWEEP_INTERVAL_SECONDS = 300

T = TypeVar("T")


class SessionCache(Generic[T]):
    """A generic in-memory cache-in-front-of-DB store keyed by session_id.
    Owns the three primitives that used to be duplicated across
    lead_agent.py and feedback_agent.py: a per-session_id dict guarded by a
    lock, a "last touched" timestamp per session_id (for TTL eviction), and
    read-through-to-DB on a cache miss.

    `loader(db, session_id)` reads the persisted value (or returns None);
    `persister(db, session_id, value)` writes it. Both are optional so this
    can also back a read-only derived cache (see the completed-lead cache
    below, which is filled from Lead rows rather than round-tripped through
    its own table).
    """

    def __init__(
        self,
        loader: Callable[[Session, str], T | None] | None = None,
        persister: Callable[[Session, str, T], None] | None = None,
        deleter: Callable[[Session, str], None] | None = None,
    ):
        self._loader = loader
        self._persister = persister
        self._deleter = deleter
        self._lock = threading.Lock()
        self._store: dict[str, T] = {}
        self._touch_lock = threading.Lock()
        self._last_touch: dict[str, float] = {}

    def touch(self, session_id: str) -> None:
        with self._touch_lock:
            self._last_touch[session_id] = time.monotonic()

    def peek(self, session_id: str) -> T | None:
        """In-memory only, no DB read-through."""
        with self._lock:
            return self._store.get(session_id)

    def keys(self) -> set[str]:
        with self._lock:
            return set(self._store.keys())

    def get(self, session_id: str, default_factory: Callable[[], T]) -> T:
        """Cache hit returns immediately. On a miss, reads through
        `loader` against a short-lived DB session, falls back to
        `default_factory()` if nothing's persisted either, then backfills
        the in-memory cache so repeat reads in this process don't hit the
        DB again."""
        self.touch(session_id)
        with self._lock:
            state = self._store.get(session_id)
            if state is not None:
                return state

        loaded = None
        if self._loader is not None:
            db = SessionLocal()
            try:
                loaded = self._loader(db, session_id)
            finally:
                db.close()

        state = loaded if loaded is not None else default_factory()
        with self._lock:
            return self._store.setdefault(session_id, state)

    def put(self, session_id: str, value: T) -> None:
        """Cache-only write, no DB persist - for backfilling from a value
        that's already authoritative elsewhere (e.g. a Lead row just read,
        or a lead just finalized to the leads table by the caller)."""
        self.touch(session_id)
        with self._lock:
            self._store[session_id] = value

    def set(self, session_id: str, value: T, db: Session | None = None) -> None:
        """Writes the in-memory cache and persists to DB together, so no
        caller ever mutates one without the other."""
        self.touch(session_id)
        with self._lock:
            self._store[session_id] = value
        if self._persister is None:
            return
        owns_db = db is None
        db = db or SessionLocal()
        try:
            self._persister(db, session_id, value)
        finally:
            if owns_db:
                db.close()

    def evict(self, session_id: str) -> None:
        self.evict_memory_only(session_id)
        if self._deleter is None:
            return
        db = SessionLocal()
        try:
            self._deleter(db, session_id)
        finally:
            db.close()

    def evict_memory_only(self, session_id: str) -> None:
        """Drops the in-memory entry without touching the DB row - used to
        simulate a restart (cache wiped, DB state intact) in tests, and
        internally by evict_expired/evict."""
        with self._lock:
            self._store.pop(session_id, None)
        with self._touch_lock:
            self._last_touch.pop(session_id, None)

    def evict_expired(self, now: float, ttl_seconds: int) -> None:
        with self._touch_lock:
            expired = [sid for sid, ts in self._last_touch.items() if now - ts > ttl_seconds]
            for sid in expired:
                del self._last_touch[sid]
        if not expired:
            return
        with self._lock:
            for sid in expired:
                self._store.pop(sid, None)


# ---------------------------------------------------------------------------
# Per-session lock registry - shared by lead_agent and feedback_agent so a
# lead-flow turn and a feedback-flow turn for the *same* session_id are
# serialized against each other too, not just against same-domain turns.
# ---------------------------------------------------------------------------

_session_locks_lock = threading.Lock()
_session_locks: dict[str, threading.Lock] = {}


def get_session_lock(session_id: str) -> threading.Lock:
    with _session_locks_lock:
        return _session_locks.setdefault(session_id, threading.Lock())


def _evict_session_lock(session_id: str) -> None:
    with _session_locks_lock:
        _session_locks.pop(session_id, None)


# ---------------------------------------------------------------------------
# Lead-collection state
# ---------------------------------------------------------------------------


def _blank_lead_state() -> dict:
    state = {field: None for field in LEAD_FIELDS}
    state["_awaiting"] = None
    return state


def _load_lead_state(db: Session, session_id: str) -> dict | None:
    return load_lead_session(db, session_id)


_lead_cache: SessionCache[dict] = SessionCache(
    loader=_load_lead_state, persister=save_lead_session, deleter=delete_lead_session
)

# Cache for a *completed* lead's info, read-only from this module's
# perspective (filled from the leads table, never written straight through
# to it - see get_lead_info). No loader: the DB fallback for this tier is
# get_latest_lead_by_session, which needs a different query shape than a
# simple by-PK loader, so it's called explicitly in get_lead_info instead.
_completed_lead_cache: SessionCache[dict] = SessionCache()

# Feedback ask/declined/given latch.
_feedback_cache: SessionCache[str] = SessionCache(
    loader=load_feedback_state, persister=save_feedback_state, deleter=delete_feedback_state
)

_all_caches = [_lead_cache, _completed_lead_cache, _feedback_cache]


def get_lead_state(session_id: str) -> dict:
    """In-progress lead-collection state for this session: cache hit, or a
    read-through to the lead_sessions table, or a blank state if neither has
    anything - this is what lets a partially-filled form resume correctly
    after a restart wiped the in-memory cache."""
    return _lead_cache.get(session_id, _blank_lead_state)


def update_lead_state(session_id: str, patch: dict, db: Session | None = None) -> dict:
    """Applies `patch` on top of the current lead state and writes the
    result to memory + lead_sessions together. Returns the new state."""
    state = dict(get_lead_state(session_id))
    state.update(patch)
    _lead_cache.set(session_id, state, db=db)
    return state


def reset_lead_state(session_id: str) -> None:
    _lead_cache.evict(session_id)
    _evict_session_lock(session_id)


def store_completed_lead(session_id: str, state: dict) -> None:
    # Stores all LEAD_FIELDS (not just name/company/email/phone) so
    # get_lead_info() can answer "what did I say I needed help with?" too.
    _completed_lead_cache.put(session_id, {field: state.get(field) for field in LEAD_FIELDS})


def get_completed_lead(session_id: str) -> dict | None:
    return _completed_lead_cache.peek(session_id)


def get_lead_info(session_id: str) -> dict:
    """Read-only recall of this session's own previously-given lead info
    (name/company/email/phone/message) - used by orchestrator's "what's my
    name" style self-recall. Works whether the form is still in progress or
    already completed, preferring the live in-progress state when one
    exists since it's the most current.

    Strictly scoped to `session_id`: only ever reads this session's own
    state, never another session's. Three tiers, in order: live in-progress
    cache, completed-lead cache, then the leads table (filtered by this
    session_id only) - so a completed lead survives a restart or worker
    recycle the same way an in-progress one already does via lead_sessions.
    A DB hit is cached back so repeat asks in the same process don't re-hit
    the DB. Returns {} if this session hasn't given anything yet, anywhere.
    """
    in_progress = _lead_cache.peek(session_id)
    if in_progress is not None:
        return {field: in_progress.get(field) for field in LEAD_FIELDS}

    completed = get_completed_lead(session_id)
    if completed:
        return dict(completed)

    db = SessionLocal()
    try:
        lead = get_latest_lead_by_session(db, session_id)
    finally:
        db.close()

    if lead is None:
        return {}

    info = {field: getattr(lead, field) for field in LEAD_FIELDS}
    store_completed_lead(session_id, info)
    return info


def is_collecting(session_id: str) -> bool:
    """True if this session is mid-way through lead collection. Checked
    against the in-memory cache first; on a miss, reads through to the
    lead_sessions table so a form still mid-collection when the server
    restarted is correctly recognized as such on the very next message."""
    state = _lead_cache.peek(session_id)
    if state is not None:
        return bool(state.get("_awaiting"))

    db = SessionLocal()
    try:
        loaded = load_lead_session(db, session_id)
    finally:
        db.close()
    return bool(loaded and loaded.get("_awaiting"))


# ---------------------------------------------------------------------------
# Feedback ask/declined/given latch
# ---------------------------------------------------------------------------


def get_feedback_status(session_id: str) -> str | None:
    return _feedback_cache.get(session_id, lambda: None)


def set_feedback_status(session_id: str, status: str, db: Session | None = None) -> None:
    _feedback_cache.set(session_id, status, db=db)


def reset_feedback_status(session_id: str) -> None:
    _feedback_cache.evict(session_id)
    _evict_session_lock(session_id)


def should_ask_feedback(session_id: str) -> bool:
    """True if this session hasn't been asked for feedback yet at all."""
    return get_feedback_status(session_id) is None


def is_awaiting_feedback_response(session_id: str) -> bool:
    return get_feedback_status(session_id) == "asked"


# ---------------------------------------------------------------------------
# Unified read/write across both domains - for callers that want the whole
# picture in one call rather than one domain at a time.
# ---------------------------------------------------------------------------


def get_session_state(session_id: str) -> dict:
    """Everything session_service knows about a session: lead fields +
    _awaiting (from the lead cache/lead_sessions table) plus feedback_status
    (from the feedback cache/feedback_states table). Each part
    independently follows cache -> DB -> default, per get_lead_state /
    get_feedback_status above."""
    state = dict(get_lead_state(session_id))
    state["feedback_status"] = get_feedback_status(session_id)
    return state


def update_session_state(session_id: str, patch: dict, db: Session | None = None) -> dict:
    """Applies `patch` to session state, writing memory + DB together for
    whichever domain(s) the patch touches - no more manually updating an
    in-memory dict in one place and remembering to persist it in another.
    `patch` may mix lead fields (any of LEAD_FIELDS, or "_awaiting") with
    "feedback_status" in the same call."""
    lead_keys = set(LEAD_FIELDS) | {"_awaiting"}
    lead_patch = {k: v for k, v in patch.items() if k in lead_keys}
    result = get_session_state(session_id)

    if lead_patch:
        result.update(update_lead_state(session_id, lead_patch, db=db))

    if "feedback_status" in patch:
        set_feedback_status(session_id, patch["feedback_status"], db=db)
        result["feedback_status"] = patch["feedback_status"]

    return result


# ---------------------------------------------------------------------------
# TTL sweep - one gate, one call site (piggybacked off ordinary traffic in
# orchestrator.handle_message), instead of the two near-identical sweeps
# lead_agent.py and feedback_agent.py used to each run independently.
# ---------------------------------------------------------------------------

_sweep_lock = threading.Lock()
_last_sweep = 0.0


def sweep_expired_sessions(now: float | None = None, ttl_seconds: int = SESSION_TTL_SECONDS) -> None:
    """Evicts session state (lead progress, completed-lead reuse cache,
    feedback latch, per-session locks) untouched for over `ttl_seconds`, so
    a long-running instance doesn't accumulate one entry per visitor
    forever. Runs at most once per _SWEEP_INTERVAL_SECONDS, gated globally
    across all caches rather than per-cache.

    Also clears matching rows out of lead_sessions and feedback_states, on
    the same cadence but keyed off each table's own updated_at column
    rather than in-memory touch times - that way state from BEFORE this
    process started (never touched in this process's cache at all) still
    eventually gets cleaned up, not just entries evicted from memory this
    run."""
    global _last_sweep
    now = now if now is not None else time.monotonic()
    with _sweep_lock:
        if now - _last_sweep < _SWEEP_INTERVAL_SECONDS:
            return
        _last_sweep = now

    for cache in _all_caches:
        cache.evict_expired(now, ttl_seconds)

    with _session_locks_lock:
        # Session locks have no per-lock touch time of their own; safe to
        # drop any not currently backing a live cache entry.
        live_sessions: set[str] = set()
        for cache in _all_caches:
            live_sessions |= cache.keys()
        stale_locks = [sid for sid in _session_locks if sid not in live_sessions]
        for sid in stale_locks:
            _session_locks.pop(sid, None)

    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)
        delete_stale_lead_sessions(db, cutoff)
        delete_stale_feedback_states(db, cutoff)
    except Exception:
        logger.exception("Stale session cleanup failed")
    finally:
        db.close()
