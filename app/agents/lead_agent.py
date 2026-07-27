"""
Lead agent: collects lead details step-by-step across turns, tracked per
session_id, then saves to the Leads table and forwards inline via
lead_service (email) and sheets_service (Google Sheets row) before the chat
response is returned - see the forwarding note further down for why.

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

Session progress (in-progress fields, the completed-lead recall cache, and
the "who's currently mid-collection" flag) is owned by
app.services.session_service now - see SESSION_STATE.md for the full flow.
This module only holds the lead-collection state *machine* (which field is
asked next, validation, AI plausibility checks); session_service owns where
that state actually lives, including the in-memory-cache-in-front-of-DB
pattern that used to be reimplemented here directly.

On finalization, the email (Resend) and Google Sheets forwarding happen
inline, synchronously, before handle_lead_message returns - not via
FastAPI's BackgroundTasks. BackgroundTasks only runs its callbacks after the
HTTP response is sent, which on a serverless platform (Vercel) is exactly
when the function may be frozen or torn down, so a deferred send could
silently never happen. Both forward_lead and append_lead_row already block
with their own bounded timeout (see lead_service.py/sheets_service.py), so
calling them inline just moves a few hundred ms of already-bounded latency
into the request instead of after it - a deliberate tradeoff for guaranteed
delivery over shaving response time.
"""
import logging
import re
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.database import Lead, SessionLocal, sanitize_input
from app.services import session_service
from app.services.ai_service import AIServiceError, get_ai_response
from app.services.lead_service import forward_lead
from app.services.session_service import LEAD_FIELDS
from app.services.sheets_service import append_lead_row
from app.utils import is_bare_greeting

logger = logging.getLogger("xqora.lead_agent")


@dataclass
class LeadStepResult:
    reply: str
    consumed: bool
    finalized: bool = False

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


def get_lead_info(session_id: str) -> dict:
    """Read-only recall of this session's own previously-given lead info -
    thin pass-through to session_service, kept here so orchestrator.py's
    existing `from app.agents.lead_agent import get_lead_info` doesn't need
    to change. See session_service.get_lead_info for the actual cache/DB
    cascade."""
    return session_service.get_lead_info(session_id)


def is_collecting(session_id: str) -> bool:
    """True if this session is mid-way through lead collection - lets the
    orchestrator route straight back into lead_agent on the next turn
    instead of re-running intent classification, which would otherwise
    misroute or guardrail-block plain answers like "John Doe" or a bare
    email address. Thin pass-through to session_service; see
    session_service.is_collecting for the cache/DB cascade."""
    return session_service.is_collecting(session_id)


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


def _forward_lead_now(lead_id: int, name: str, company: str, email: str, phone: str, need: str) -> None:
    """Called synchronously (inline, before handle_lead_message returns) for
    a just-completed lead: forwards it to the team by email, and appends it
    as a row to the shared Google Sheet. The two are independent - a Sheets
    failure (bad sheet ID, revoked/misconfigured credentials, network error)
    must never stop the email from going out, and an email failure must
    never stop the Sheets row, so each is wrapped in its own try/except
    rather than one failure short-circuiting the other. (Both forward_lead
    and append_lead_row already catch their own errors internally and return
    a bool rather than raising - the try/except here is defense in depth,
    not the primary safety net.)"""
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


def _finalize_lead(db: Session, session_id: str, state: dict) -> LeadStepResult:
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
    _forward_lead_now(lead.id, lead.name or "", lead.company or "", lead.email or "", lead.phone or "", need)

    session_service.store_completed_lead(session_id, state)
    session_service.reset_lead_state(session_id)
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


def handle_lead_message(db: Session, session_id: str, message: str) -> LeadStepResult:
    """Serialized per session_id so concurrent requests for the same session
    (double-clicked send, a client retry) can't race on the session's dict -
    e.g. both reading a field as unfilled and both writing to it, or both
    finalizing the same lead."""
    with session_service.get_session_lock(session_id):
        return _handle_lead_message_locked(db, session_id, message)


def _handle_lead_message_locked(db: Session, session_id: str, message: str) -> LeadStepResult:
    state = dict(session_service.get_lead_state(session_id))
    pending_field = state.get("_awaiting")

    if pending_field is None and _next_missing_field(state) == "name" and session_service.get_completed_lead(session_id):
        state["_awaiting"] = "_reuse_confirm"
        session_service.update_lead_state(session_id, state, db=db)
        return LeadStepResult(reply=REUSE_DETAILS_PROMPT, consumed=True)

    if pending_field:
        stripped = message.strip()

        if _is_cancel_request(stripped):
            session_service.reset_lead_state(session_id)
            return LeadStepResult(reply=CANCEL_MESSAGE, consumed=True)

        if pending_field == "_reuse_confirm":
            if _wants_reuse(stripped):
                prev = session_service.get_completed_lead(session_id) or {}
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
        session_service.update_lead_state(session_id, state, db=db)
        return next_step
    return _finalize_lead(db, session_id, state)
