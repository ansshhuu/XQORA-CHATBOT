# Session state: one flow, one place to look

Every piece of per-`session_id` state in this app goes through
[`app/services/session_service.py`](app/services/session_service.py). If you're
about to add a new `dict[session_id, ...]` somewhere, or a new column that
tracks per-session progress, put it there instead - that's the whole point
of this doc.

## The rule

**The DB is the source of truth for anything that must survive a restart.
The in-memory cache in `session_service` is only a same-process speed-up in
front of it, never the source of truth.**

Before this refactor, that rule was applied once, ad hoc, to fix a specific
bug (lead self-recall forgetting a completed lead after a redeploy). It's
now the standing pattern for every kind of session state in the app,
including one gap the ad hoc fix didn't cover: feedback's ask/declined/given
latch used to live only in memory and would silently reset on every
restart, letting the same session get asked for feedback twice. It's
persisted now too (see the `feedback_states` table below).

## What lives where

| Data | In-memory cache | DB table | Owner |
|---|---|---|---|
| In-progress lead fields (`name`/`company`/`email`/`phone`/`message`) + which field is currently being asked (`_awaiting`) | `session_service._lead_cache` | `lead_sessions` (`LeadSession` model) | `session_service.get_lead_state` / `update_lead_state` / `reset_lead_state` |
| A session's most recently **completed** lead, for "what's my email" style recall | `session_service._completed_lead_cache` | `leads` (`Lead` model, via `get_latest_lead_by_session`) | `session_service.get_lead_info` / `store_completed_lead` / `get_completed_lead` |
| Feedback ask/declined/given latch | `session_service._feedback_cache` | `feedback_states` (`FeedbackState` model) | `session_service.get_feedback_status` / `set_feedback_status` / `reset_feedback_status` |
| Actual feedback answers (rating/comment) | *(none - write-once)* | `feedback` (`Feedback` model) | `database.save_feedback` |
| Conversation history / last classified intent | *(none - read fresh every turn)* | `chat_history` (`ChatHistory` model) | `database.save_chat_history` / `get_recent_chat_history` |
| Per-session lock (serializes concurrent turns for one session) | `session_service._session_locks` | — | `session_service.get_session_lock` |

Chat history has no cache layer on purpose: it's only ever read as "the last
3 turns" to build `recent_context`, so caching it wouldn't save much and
would add another thing that could drift from the DB.

## The cache-in-front-of-DB pattern (`SessionCache`)

Every cached row above is backed by the same generic primitive,
`session_service.SessionCache`:

- **`get(session_id, default_factory)`** - in-memory hit returns immediately.
  On a miss (fresh process, session predates this process, multi-worker
  deployment) it reads through to the DB via the cache's `loader`, falls
  back to `default_factory()` if nothing's persisted either, then backfills
  the in-memory entry so the next read in this process doesn't hit the DB.
- **`set(session_id, value, db=None)`** - writes memory *and* the DB
  together, via the cache's `persister`. This is what `update_lead_state`
  and `set_feedback_status` call - there is no path in this codebase that
  updates the in-memory copy without also persisting it, or vice versa.
- **`evict(session_id)`** - clears both memory and the DB row, via the
  cache's `deleter`.

This is the exact shape that used to be hand-rolled three separate times in
`lead_agent.py` (once each for in-progress state, completed-lead recall, and
the `is_collecting` flag) and was missing entirely for feedback state. Now
it's written once and reused for every kind of session data.

## Reading/writing session state

Two ways to touch state, pick based on what you're doing:

- **Domain-specific functions** (`get_lead_state` / `update_lead_state`,
  `get_feedback_status` / `set_feedback_status`, etc.) - what `lead_agent.py`
  and `feedback_agent.py` actually use turn-by-turn, since the lead
  collection state machine mutates one field at a time and needs the real
  dict shape.
- **`get_session_state(session_id)` / `update_session_state(session_id, patch, db=None)`**
  - a unified view across both domains in one call (`patch` can mix lead
  fields with `"feedback_status"` in the same dict). Use this for anything
  that wants the whole picture at once (e.g. a future debug/admin endpoint)
  rather than reaching into one domain at a time.

Neither `lead_agent.py`, `feedback_agent.py`, nor `orchestrator.py` keep
their own session dicts anymore - they all call into `session_service`.
`intent_agent.py` never had any session state (it's a pure function of
`message` + `recent_context` + `previous_intent`, all passed in as
arguments), so there was nothing to migrate there.

## TTL sweep

One gate, one call site: `orchestrator.handle_message` calls
`session_service.sweep_expired_sessions()` at the top of every turn. It's
rate-limited internally (runs at most once per 5 minutes) so this is cheap
to call unconditionally. A single sweep evicts stale entries from *all*
three in-memory caches, drops per-session locks no longer backing a live
entry, and deletes stale `lead_sessions` / `feedback_states` rows (keyed off
each table's own `updated_at`, so state from before this process started
still gets cleaned up eventually, not just entries evicted from memory this
run).

Before this refactor, `lead_agent.py` and `feedback_agent.py` each ran this
same sweep independently, piggybacked off `is_collecting()` and
`is_awaiting_response()` respectively - two copies of the same logic,
gated separately, that could drift out of sync.

## Adding a new kind of session state

1. Add a DB model + `load_x` / `save_x` / `delete_x` functions in
   `app/database.py`, following `LeadSession`/`FeedbackState` as the
   template (a `session_id` primary key, an `updated_at` column for the TTL
   sweep to key off).
2. In `session_service.py`, create a `SessionCache` instance with those
   three functions as `loader`/`persister`/`deleter`, add it to
   `_all_caches` so the TTL sweep picks it up, and write thin
   `get_x_state` / `update_x_state` / `reset_x_state` wrappers (see the
   lead/feedback sections for the shape).
3. Call those wrappers from the agent/route that owns the data. Don't add a
   new module-level `dict[session_id, ...]` anywhere else in the codebase -
   if you're tempted to, that state belongs in `session_service` instead.
