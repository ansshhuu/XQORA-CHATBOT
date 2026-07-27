"""
Lead agent: collects lead details step-by-step across turns, tracked per
session_id, then saves to the Leads table and forwards asynchronously via
lead_service (email) and sheets_service (Google Sheets row) - doesn't block
the chat response on either send.

Only collects name, company (optional), email, phone, and a short need
description. No budget or service-fit discussion here, that's handled by
the team on a call.

Every incoming message is checked against the field currently being asked
for before being accepted as that field's value. This used to be a purely
heuristic check (word count, question marks, "@" presence) which reliably
caught only narrow cases (very short input, obvious greetings) and let real
sentences like "need somewhere to host my app" slip through as a literal
company name. It's now an AI-backed plausibility check (_field_matches_ai,
same paradigm intent_agent.py already uses for intent/greeting judgment),
with the old heuristic kept only as a fallback for when the AI call itself
fails. handle_lead_message signals a rejection back via
LeadStepResult.consumed=False, and orchestrator.py is responsible for
routing that message normally and nudging the user back toward the paused
field afterward, rather than lead_agent trying to do that routing itself.
Partially-collected fields are never touched by this path, so the flow
resumes from wherever it paused rather than restarting.

Users can also bail out of collection entirely at any step ("never mind",
"cancel", "stop") - see _is_cancel_request.

Session progress is held in an in-memory dict, keyed by session_id, for fast
same-process access - but that dict is only a cache now, not the source of
truth: every mutating turn also persists the state to the lead_sessions
table (see _persist_session / database.save_lead_session), and a cache miss
(fresh session, or a session that predates this process - e.g. right after a
restart) reads it back from there (see _get_session / database.load_lead_session)
instead of starting over. That's what makes an in-progress form survive a
server restart. A multi-worker/multi-instance deployment would still need a
shared cache (e.g. Redis) in front of the DB to avoid every request round-
tripping to it, but correctness no longer depends on the in-memory dict
surviving.
"""
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from app.core.config import SESSION_TTL_SECONDS
from app.database import (
    Lead,
    SessionLocal,
    delete_lead_session,
    delete_stale_lead_sessions,
    get_latest_lead_by_session,
    load_lead_session,
    sanitize_input,
    save_lead_session,
)
from app.services.ai_service import AIServiceError, get_ai_response
from app.services.lead_service import forward_lead
from app.services.sheets_service import append_lead_row
from app.utils import is_bare_greeting

logger = logging.getLogger("xqora.lead_agent")


@dataclass
class LeadStepResult:
    reply: str
    consumed: bool  
    finalized: bool = False 


LEAD_FIELDS = ["name", "company", "email", "phone", "message"]

FIELD_PROMPTS = {
    "name": "What's your name?",
    "company": "Which company are you with? (just say 'none' if not applicable)",
    "email": "What's the best email to reach you at?",
    "phone": "And a good phone number to reach you on?",
    "message": "What do you need help with? A quick description is fine.",
}

VALIDATION_PROMPTS = {
    "email": "Hmm, that doesn't look like a valid email address, mind double-checking and sending it again?",
    "phone": "That doesn't look like a valid phone number, could you send it again? (10 digits, with country code if you're outside India)",
}

CANCEL_MESSAGE = "No worries, I've cancelled that. Let me know if you'd like to start over or need anything else!"

REUSE_DETAILS_PROMPT = (
    "Want me to use your previous details (name/email/phone) for this, or provide new details?"
)

_REUSE_FIELDS = ("name", "company", "email", "phone")

_REUSE_AFFIRMATIVE_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok(ay)?|reuse|same|previous|use\s+(the\s+)?(same|previous|it))\b",
    re.IGNORECASE,
)
_REUSE_NEGATIVE_RE = re.compile(
    r"\b(no|nope|new|different|fresh|start\s+over|provide\s+new)\b",
    re.IGNORECASE,
)


def _wants_reuse(message: str) -> bool:
    return bool(_REUSE_AFFIRMATIVE_RE.search(message.strip().lower()))


def _wants_new_details(message: str) -> bool:
    return bool(_REUSE_NEGATIVE_RE.search(message.strip().lower()))


_EMAIL_EXTRACT_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_EMAIL_VALIDATE_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

_PHONE_EXTRACT_RE = re.compile(r"\+?[\d][\d\s\-\+\(\)]{6,}\d")
_PHONE_STRIP_RE = re.compile(r"[\s\-\(\)]")
_PHONE_VALIDATE_RE = re.compile(r"^\+?\d{10,13}$")

_QUESTION_STARTER_RE = re.compile(
    r"^\s*(can|could|would|will|do|does|did|is|are|what|why|how|when|where|who)\b",
    re.IGNORECASE,
)

_CANCEL_RE = re.compile(
    r"^\s*(never\s?mind|nevermind|cancel|stop|forget\s+it|nvm|not\s+interested|quit)\s*[!.,]*$",
    re.IGNORECASE,
)

_FIELD_LABELS = {
    "name": "name",
    "company": "company name",
    "email": "email",
    "phone": "phone number",
}

_FIELD_DESCRIPTIONS = {
    "name": 'the person\'s own name (e.g. "Jane Smith", "Raj Patel")',
    "company": 'a company/organization name, or something like "none"/"n/a" if they don\'t have one',
    "email": "an email address, even a slightly malformed or typo'd one",
    "phone": "a phone number, even a slightly malformed or oddly formatted one",
}

_FIELD_CHECK_SYSTEM_PROMPT = """You are checking whether a chat message plausibly
answers a specific question in a lead-capture form, or whether the user has gone
off-flow instead (asking something else, raising a new need or complaint, making
small talk, etc). Reply with ONLY one word: YES if the message plausibly provides
the requested value, or NO if it's an unrelated question, comment, or a new topic."""


def _is_cancel_request(message: str) -> bool:
    return bool(_CANCEL_RE.match(message.strip()))


def _looks_off_flow_heuristic(field: str, message: str) -> bool:
    """Fallback used only if the AI field-match check itself fails (see
    _field_matches_ai) - i.e. only when the AI is unavailable. Deliberately
    stricter than the AI judgment call: when we can't get a real answer, the
    safer failure mode is to reject an ambiguous message (worst case, a
    slightly unusual real answer gets rerouted and the user re-sends it) than
    to accept it (worst case, an unrelated sentence gets silently stored as
    someone's "name" - the actual bug this whole check exists to prevent).
    """
    stripped = message.strip()
    word_count = len(stripped.split())
    generic_aside = "?" in stripped or bool(_QUESTION_STARTER_RE.match(stripped))

    if field == "name":
        return generic_aside or word_count > 4
    if field == "company":
        return generic_aside or word_count > 5
    if field == "email":
        return "@" not in stripped and (generic_aside or word_count > 4)
    if field == "phone":
        digit_count = sum(ch.isdigit() for ch in stripped)
        return digit_count < 7 and (generic_aside or word_count > 4)
    return False  # the free-form "message" field accepts anything


def _field_matches_ai(field: str, message: str) -> bool:
    """Does `message` plausibly provide the value for `field`? Uses the same
    AI-based judgment approach as intent_agent's classification, since a
    fixed heuristic (word count, question marks) reliably failed on real
    sentences like "need somewhere to host my app" that are clearly not a
    company name but don't trip any narrow rule either."""
    description = _FIELD_DESCRIPTIONS.get(field)
    if not description:
        return True  
    prompt = (
        f"The user is being asked to provide: {description}.\n"
        f'Their message: "{message}"\n\n'
        "Does this message plausibly provide that value? Reply with only YES or NO."
    )
    try:
        raw = get_ai_response(prompt, system_prompt=_FIELD_CHECK_SYSTEM_PROMPT)
    except AIServiceError:
        logger.warning("Lead field-match AI check failed for field=%s; falling back to heuristic", field)
        return not _looks_off_flow_heuristic(field, message)
    return raw.strip().upper().startswith("Y")


def _resume_nudge(field: str) -> str:
    return f"Got it! Picking back up on getting you connected with the team. {FIELD_PROMPTS[field]}"


_sessions_lock = threading.Lock()
_sessions: dict[str, dict] = {}

_session_locks_lock = threading.Lock()
_session_locks: dict[str, threading.Lock] = {}


def _get_session_lock(session_id: str) -> threading.Lock:
    with _session_locks_lock:
        return _session_locks.setdefault(session_id, threading.Lock())


_completed_leads_lock = threading.Lock()
_completed_leads: dict[str, dict] = {}

_SWEEP_INTERVAL_SECONDS = 300

_touch_lock = threading.Lock()
_last_touch: dict[str, float] = {}
_last_sweep = 0.0


def _touch(session_id: str) -> None:
    """Record activity for `session_id` so a TTL sweep won't evict it while
    it's still in use."""
    with _touch_lock:
        _last_touch[session_id] = time.monotonic()


def _sweep_expired_sessions(now: float) -> None:
    """Evict session state (collection progress, completed-lead reuse
    details, per-session locks) that's been untouched for over
    SESSION_TTL_SECONDS, so a long-running instance doesn't accumulate one
    entry per visitor forever. Runs at most once per _SWEEP_INTERVAL_SECONDS,
    triggered opportunistically off real traffic (see is_collecting below)
    rather than a background thread/timer.

    Also clears matching rows out of the lead_sessions table, on the same
    cadence but keyed off the DB's own updated_at column rather than the
    in-memory touch times - that way an abandoned form from BEFORE this
    process started (so it was never touched in this process's _last_touch
    at all) still eventually gets cleaned up, not just ones evicted from the
    in-memory cache this run."""
    global _last_sweep
    with _touch_lock:
        if now - _last_sweep < _SWEEP_INTERVAL_SECONDS:
            return
        _last_sweep = now
        expired = [sid for sid, ts in _last_touch.items() if now - ts > SESSION_TTL_SECONDS]
        for sid in expired:
            del _last_touch[sid]

    if expired:
        with _sessions_lock:
            for sid in expired:
                _sessions.pop(sid, None)
        with _completed_leads_lock:
            for sid in expired:
                _completed_leads.pop(sid, None)
        with _session_locks_lock:
            for sid in expired:
                _session_locks.pop(sid, None)

    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=SESSION_TTL_SECONDS)
        delete_stale_lead_sessions(db, cutoff)
    except Exception:
        logger.exception("Stale lead_sessions cleanup failed")
    finally:
        db.close()


def _get_session(session_id: str) -> dict:
    """In-memory session state for the current process. On a cache miss,
    falls back to the persisted lead_sessions row (see database.py) before
    defaulting to a blank state - this is what lets an in-progress form
    resume correctly after a restart wiped the in-memory cache. Callers that
    mutate the returned dict are responsible for persisting the change via
    _persist_session using their own request-scoped db session; this
    function only needs a short-lived connection of its own for the (
    uncommon) read-through fallback."""
    _touch(session_id)
    with _sessions_lock:
        state = _sessions.get(session_id)
        if state is not None:
            return state

    db = SessionLocal()
    try:
        loaded = load_lead_session(db, session_id)
    finally:
        db.close()

    state = {field: None for field in LEAD_FIELDS}
    if loaded:
        for field in LEAD_FIELDS:
            state[field] = loaded.get(field)
        state["_awaiting"] = loaded.get("_awaiting")

    with _sessions_lock:
        return _sessions.setdefault(session_id, state)


def _persist_session(db: Session, session_id: str, state: dict) -> None:
    """Saves the current in-progress state to the lead_sessions table using
    the caller's request-scoped db session. Called on every turn that
    mutates `state`, so a restart between turns never loses more than the
    single in-flight message."""
    save_lead_session(db, session_id, state)


def reset_session(session_id: str) -> None:
    with _sessions_lock:
        _sessions.pop(session_id, None)
    with _session_locks_lock:
        _session_locks.pop(session_id, None)
    with _touch_lock:
        _last_touch.pop(session_id, None)

    db = SessionLocal()
    try:
        delete_lead_session(db, session_id)
    finally:
        db.close()


def _store_completed_lead(session_id: str, state: dict) -> None:
    # Stores all LEAD_FIELDS (not just _REUSE_FIELDS) so get_lead_info() can
    # answer "what did I say I needed help with?" too, not just the
    # name/company/email/phone subset the reuse-previous-details flow cares
    # about. The reuse flow below only ever reads _REUSE_FIELDS back out of
    # this dict, so the extra "message" key is inert there.
    with _completed_leads_lock:
        _completed_leads[session_id] = {field: state.get(field) for field in LEAD_FIELDS}


def _get_completed_lead(session_id: str) -> dict | None:
    with _completed_leads_lock:
        return _completed_leads.get(session_id)


def get_lead_info(session_id: str) -> dict:
    """Read-only recall of this session's own previously-given lead info
    (name/company/email/phone/message) for the "what's my name" style
    self-recall check in orchestrator.py - works whether the form is still
    in progress or already completed, preferring the live in-progress
    state when one exists since it's the most current.

    Strictly scoped to `session_id`: only ever reads this session's own
    state, never another session's. Checks in-memory state first
    (_sessions / _completed_leads), and only on a cache miss falls back to
    the leads table (filtered by this session_id only - see
    get_latest_lead_by_session) so a completed lead survives a restart or
    worker recycle the same way an in-progress one already does via
    lead_sessions. A DB hit is cached back into _completed_leads so repeat
    asks in the same process don't re-hit the DB. Returns {} if this
    session hasn't given anything yet, anywhere."""
    with _sessions_lock:
        in_progress = _sessions.get(session_id)
        if in_progress is not None:
            return {field: in_progress.get(field) for field in LEAD_FIELDS}

    completed = _get_completed_lead(session_id)
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
    _store_completed_lead(session_id, info)
    return info


def is_collecting(session_id: str) -> bool:
    """True if this session is mid-way through lead collection.

    Lets the orchestrator route straight back into lead_agent on the next
    turn instead of re-running intent classification, which would otherwise
    misroute or guardrail-block plain answers like "John Doe" or a bare
    email address.

    Called on every turn by the orchestrator regardless of session, which
    also makes it a convenient place to piggyback the TTL sweep so expired
    session state gets cleaned up off ordinary traffic.

    Checked against the in-memory cache first; on a miss, falls back to the
    persisted lead_sessions row so a form still mid-collection when the
    server restarted is correctly recognized as such on the very next
    message, instead of that message being misrouted through normal intent
    classification because the in-memory cache came back empty.
    """
    _sweep_expired_sessions(time.monotonic())
    with _sessions_lock:
        state = _sessions.get(session_id)
    if state is not None:
        return bool(state.get("_awaiting"))

    db = SessionLocal()
    try:
        loaded = load_lead_session(db, session_id)
    finally:
        db.close()
    return bool(loaded and loaded.get("_awaiting"))


def _next_missing_field(state: dict) -> str | None:
    for field in LEAD_FIELDS:
        if not state.get(field):
            return field
    return None


def _extract_value(field: str, message: str) -> str:
    message = message.strip()
    if field == "email":
        match = _EMAIL_EXTRACT_RE.search(message)
        return match.group(0).rstrip(".,!?;:") if match else message
    if field == "phone":
        match = _PHONE_EXTRACT_RE.search(message)
        return match.group(0).strip() if match else message
    return message


def _validate_email(value: str) -> bool:
    return bool(_EMAIL_VALIDATE_RE.match(value.strip()))


def _normalize_phone(value: str) -> str:
    return _PHONE_STRIP_RE.sub("", value.strip())


def _validate_phone(value: str) -> bool:
    return bool(_PHONE_VALIDATE_RE.match(_normalize_phone(value)))


def _forward_lead_async(lead_id: int, name: str, company: str, email: str, phone: str, need: str) -> None:
    """Fire-and-forget background job for a just-completed lead: forwards it
    to the team by email, and appends it as a row to the shared Google
    Sheet. The two are independent - a Sheets failure (bad sheet ID,
    revoked/misconfigured credentials, network error) must never stop the
    email from going out, and an email failure must never stop the Sheets
    row, so each is wrapped in its own try/except rather than one failure
    short-circuiting the other. (Both forward_lead and append_lead_row
    already catch their own errors internally and return a bool rather than
    raising - the try/except here is defense in depth, not the primary
    safety net.)"""
    contact = email or phone or ""
    try:
        ok = forward_lead(name, contact, need)
    except Exception:
        logger.exception("Lead forward (email) failed for lead_id=%s", lead_id)
        ok = False

    if ok:
        session = SessionLocal()
        try:
            lead = session.get(Lead, lead_id)
            if lead:
                lead.forwarded = True
                session.commit()
        finally:
            session.close()

    try:
        append_lead_row(name, company, email, phone, need)
    except Exception:
        logger.exception("Lead forward (Sheets) failed for lead_id=%s", lead_id)


def _finalize_lead(db: Session, session_id: str, state: dict, background_tasks: BackgroundTasks) -> LeadStepResult:
    lead = Lead(
        session_id=session_id,
        name=state["name"],
        company=state["company"],
        email=state["email"],
        phone=state["phone"],
        message=state["message"],
    )
    db.add(lead)
    db.commit()
    db.refresh(lead)

    need = lead.message or ""
    background_tasks.add_task(
        _forward_lead_async, lead.id, lead.name or "", lead.company or "", lead.email or "", lead.phone or "", need
    )

    _store_completed_lead(session_id, state)
    reset_session(session_id)
    return LeadStepResult(reply="Thanks, got it! Our team will reach out to you shortly.", consumed=True, finalized=True)


def _advance_to_next_field(state: dict) -> LeadStepResult | None:
    """Returns the next-field prompt, or None if all fields are filled
    (caller should finalize the lead in that case)."""
    next_field = _next_missing_field(state)
    if next_field is None:
        return None
    state["_awaiting"] = next_field
    already_started = any(state.get(f) for f in LEAD_FIELDS)
    prefix = "Got it. " if already_started else "Great, let's get you connected with the team. "
    return LeadStepResult(reply=prefix + FIELD_PROMPTS[next_field], consumed=True)


def handle_lead_message(
    db: Session, session_id: str, message: str, background_tasks: BackgroundTasks
) -> LeadStepResult:
    """Serialized per session_id so concurrent requests for the same session
    (double-clicked send, a client retry) can't race on the session's dict -
    e.g. both reading a field as unfilled and both writing to it, or both
    finalizing the same lead."""
    with _get_session_lock(session_id):
        return _handle_lead_message_locked(db, session_id, message, background_tasks)


def _handle_lead_message_locked(
    db: Session, session_id: str, message: str, background_tasks: BackgroundTasks
) -> LeadStepResult:
    state = _get_session(session_id)
    pending_field = state.get("_awaiting")

    if pending_field is None and _next_missing_field(state) == "name" and _get_completed_lead(session_id):
        state["_awaiting"] = "_reuse_confirm"
        _persist_session(db, session_id, state)
        return LeadStepResult(reply=REUSE_DETAILS_PROMPT, consumed=True)

    if pending_field:
        stripped = message.strip()

        if _is_cancel_request(stripped):
            reset_session(session_id)
            return LeadStepResult(reply=CANCEL_MESSAGE, consumed=True)

        if pending_field == "_reuse_confirm":
            if _wants_reuse(stripped):
                prev = _get_completed_lead(session_id) or {}
                for field in _REUSE_FIELDS:
                    state[field] = prev.get(field)
            elif not _wants_new_details(stripped):
              
                return LeadStepResult(reply=REUSE_DETAILS_PROMPT, consumed=True)
            state["_awaiting"] = None
        elif is_bare_greeting(stripped):
            
            return LeadStepResult(reply=_resume_nudge(pending_field), consumed=False)

        elif pending_field in ("email", "phone"):
            extracted = _extract_value(pending_field, stripped)
            is_valid = _validate_email(extracted) if pending_field == "email" else _validate_phone(extracted)

            if is_valid:
                if pending_field == "phone":
                    extracted = _normalize_phone(extracted)
                state[pending_field] = sanitize_input(extracted, 500)
                state["_awaiting"] = None
            elif not _looks_off_flow_heuristic(pending_field, stripped):
                
                return LeadStepResult(reply=VALIDATION_PROMPTS[pending_field], consumed=True)
            else:
                return LeadStepResult(reply=_resume_nudge(pending_field), consumed=False)
        elif pending_field == "message":
            state["message"] = sanitize_input(stripped, 500)
            state["_awaiting"] = None
        else:  
            if not _field_matches_ai(pending_field, stripped):
                return LeadStepResult(reply=_resume_nudge(pending_field), consumed=False)
            state[pending_field] = sanitize_input(stripped, 500)
            state["_awaiting"] = None

    next_step = _advance_to_next_field(state)
    if next_step is not None:
        _persist_session(db, session_id, state)
        return next_step
    return _finalize_lead(db, session_id, state, background_tasks)
