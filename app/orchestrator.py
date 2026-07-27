"""
Single decision point for /chat: classify intent (guardrail-checked, context-
aware), route to the matching specialist agent, and log the turn to Chat
History.
"""
import logging
import re

from sqlalchemy.orm import Session

from app.agents import feedback_agent
from app.agents.escalation_agent import get_escalation_response
from app.agents.faq_agent import get_faq_response
from app.agents.intent_agent import classify_intent
from app.agents.lead_agent import get_lead_info, handle_lead_message, is_collecting
from app.agents.recommend_agent import get_recommendation_response
from app.database import get_recent_chat_history, save_chat_history
from app.prompt import CLOSING_RESPONSE, FEEDBACK_ASK
from app.services import session_service
from app.services.session_service import LEAD_FIELDS

logger = logging.getLogger("xqora.orchestrator")

_RECENT_CONTEXT_TURNS = 3

# Rule-based (no AI) "what's my X" self-recall check, run before intent
# classification so a question about the user's own already-given lead info
# never gets misrouted to the off-topic guardrail or an AI call - see
# _own_info_reply below. Matches phrasing like "what's my email", "do you
# know my name", "what did I tell you my phone number was", "what info do
# you have on me", and broader identity-recall phrasing like "do you know
# me", "do you remember me", "who am I".
_OWN_INFO_RE = re.compile(
    r"\b(?:what'?s?\s+my|what\s+is\s+my|do\s+you\s+(?:know|remember)\s+my|"
    r"do\s+you\s+(?:know|remember)\s+(?:about\s+)?me\b|who\s+am\s+i\b|"
    r"what\s+did\s+i\s+(?:tell|give)\s+you\s+(?:as\s+)?my|"
    r"what\s+did\s+i\s+(?:tell|give)\s+you\b|"
    r"what\s+(?:info|information|details)\s+(?:do\s+you\s+have|did\s+i\s+(?:give|share))\s+(?:about|on)\s+me|"
    r"show\s+(?:me\s+)?my\s+(?:details|info|information)|"
    r"remind\s+me\s+(?:of\s+)?my)\b",
    re.IGNORECASE,
)

# "who is <name>" is only a self-recall question if <name> matches this
# session's own stored lead name (checked in _own_info_reply, never another
# session's) - anchored to the whole message so it doesn't false-positive on
# unrelated questions like "who is the CEO of XQORA".
_WHO_IS_RE = re.compile(r"^\s*who\s+is\s+([a-z][a-z'\-]*(?:\s+[a-z][a-z'\-]*)?)\s*\??\s*$", re.IGNORECASE)

# A bare single word alone ("anshu") is only a self-recall trigger if it
# matches this session's own stored lead name - same scoping as _WHO_IS_RE.
# If it doesn't match, _own_info_reply returns None so the message falls
# through to intent_agent's own bare-word clarify guard instead of being fed
# to the AI classifier with recent_context, which would otherwise tend to
# just carry forward whatever topic the conversation was already on.
_BARE_WORD_RE = re.compile(r"^\s*([a-z][a-z'\-]{1,29})\s*[!.,]?\s*$", re.IGNORECASE)

_OWN_INFO_FIELD_PATTERNS = [
    ("name", re.compile(r"\bname\b", re.IGNORECASE)),
    ("email", re.compile(r"e-?mail", re.IGNORECASE)),
    ("phone", re.compile(r"\b(phone|mobile|contact\s+number)\b", re.IGNORECASE)),
    ("company", re.compile(r"\b(company|organi[sz]ation|business)\b", re.IGNORECASE)),
    ("message", re.compile(r"\b(need|require|looking\s+for|help\s+with)\b", re.IGNORECASE)),
]

_OWN_INFO_FIELD_LABELS = {
    "name": "name",
    "company": "company",
    "email": "email",
    "phone": "phone number",
    "message": "what you needed help with",
}


def _detect_own_info_field(message: str) -> str | None:
    """Returns the LEAD_FIELDS field name the message is asking about, "any"
    for a generic "what do you know about me" style ask, or None if this
    isn't a self-recall question at all."""
    if not _OWN_INFO_RE.search(message):
        return None
    for field, pattern in _OWN_INFO_FIELD_PATTERNS:
        if pattern.search(message):
            return field
    return "any"


def _format_known_info(info: dict, prefix: str = "") -> str:
    known = {k: v for k, v in info.items() if v and k in LEAD_FIELDS}
    if not known:
        return "You haven't shared any details with me yet in this chat."
    parts = [f"{_OWN_INFO_FIELD_LABELS[k]}: {known[k]}" for k in LEAD_FIELDS if known.get(k)]
    return prefix + "Here's what you've shared with me so far - " + "; ".join(parts) + "."


def _casual_identity_reply(info: dict) -> str:
    """Pattern A reply (bare name / "who is X") - a short ack referencing
    just the last known need/topic, not the full field dump. Pattern B
    (explicit "what info do you have"/"show my details" style requests)
    keeps using _format_known_info instead - see _own_info_reply."""
    name = info.get("name")
    if not name:
        return "You haven't shared any details with me yet in this chat."
    need = info.get("message")
    if need:
        return f"Hey {name}! Last I noted, you were looking into {need} - want to continue that, or something else?"
    return f"Hey {name}! Good to have you back - what can I help you with?"


def _name_matches_stored(candidate: str, info: dict) -> bool:
    stored_name = (info.get("name") or "").strip().lower()
    return bool(stored_name) and (candidate == stored_name or candidate in stored_name.split())


def _own_info_reply(session_id: str, message: str) -> str | None:
    """If `message` is asking about the user's own previously-given lead
    info, answers directly from this session's own stored state - checked
    in-memory first, falling back to this session's own leads-table row if
    the in-memory state was wiped by a restart/redeploy (see
    lead_agent.get_lead_info; always scoped to this session_id, never a
    DB-wide leads query) - and returns the reply text. Returns None if this
    isn't a self-recall question, so the caller falls through to normal
    classify_intent/AI handling."""
    # Pattern A: bare name / "who is X" - casual ack, needs-only, not the
    # full field dump (see _casual_identity_reply).
    who_is_match = _WHO_IS_RE.match(message)
    if who_is_match:
        candidate = who_is_match.group(1).strip().lower()
        info = get_lead_info(session_id)
        if _name_matches_stored(candidate, info):
            return _casual_identity_reply(info)
        return None

    bare_word_match = _BARE_WORD_RE.match(message)
    if bare_word_match:
        candidate = bare_word_match.group(1).strip().lower()
        info = get_lead_info(session_id)
        if _name_matches_stored(candidate, info):
            return _casual_identity_reply(info)
        return None

    field = _detect_own_info_field(message)
    if field is None:
        return None

    info = get_lead_info(session_id)

    if field == "any":
        return _format_known_info(info)

    value = info.get(field)
    if not value:
        return "You haven't shared that with me yet."
    return f"Your {_OWN_INFO_FIELD_LABELS[field]} is {value}."


_ROUTES = {
    "faq": lambda db, session_id, message, recent_context: get_faq_response(message, recent_context),
    "recommend": lambda db, session_id, message, recent_context: get_recommendation_response(
        message, recent_context
    ),
    "lead": lambda db, session_id, message, recent_context: handle_lead_message(db, session_id, message).reply,
    "escalate": lambda db, session_id, message, recent_context: get_escalation_response(),
}


def _build_recent_context(rows: list) -> str | None:
    if not rows:
        return None
    lines = []
    for row in rows:
        lines.append(f"User: {row.message}")
        lines.append(f"Bot: {row.response}")
    return "\n".join(lines)


def _route(
    db: Session,
    session_id: str,
    message: str,
    skip_handler_intents: frozenset[str] = frozenset(),
) -> tuple[str | None, str]:
    """Classify + dispatch a message that's NOT a lead-collection answer.
    Returns (reply, intent_label) for the caller to log. If the classified
    intent is in skip_handler_intents, the handler is not invoked (reply is
    None) - used to avoid re-running handle_lead_message when the caller
    already ran it once for this message."""
    own_info_reply = _own_info_reply(session_id, message)
    if own_info_reply is not None:
        return own_info_reply, "own_info"

    rows = get_recent_chat_history(db, session_id, limit=_RECENT_CONTEXT_TURNS)
    recent_context = _build_recent_context(rows)
    previous_intent = rows[-1].intent if rows else None
    result = classify_intent(message, recent_context=recent_context, previous_intent=previous_intent)

    if result.blocked:
        return result.refusal_message, result.intent

    if result.intent in skip_handler_intents:
        return None, result.intent

    handler = _ROUTES.get(result.intent, lambda *_: get_escalation_response())
    reply = handler(db, session_id, message, recent_context)
    return reply, result.intent


def _append_feedback_ask(db: Session, session_id: str, reply: str) -> str:
    feedback_agent.mark_asked(session_id, db=db)
    return f"{reply}\n\n{FEEDBACK_ASK}"


def _handle_message_impl(db: Session, session_id: str, message: str) -> tuple[str, str]:
    # Single piggyback point for the TTL sweep (lead state, feedback state,
    # completed-lead cache, per-session locks) - see session_service for why
    # this used to be two near-identical sweeps triggered independently.
    session_service.sweep_expired_sessions()

    if feedback_agent.is_awaiting_response(session_id):
        feedback_result = feedback_agent.handle_feedback_response(db, session_id, message)
        if feedback_result.handled:
            save_chat_history(
                db, session_id=session_id, message=message, response=feedback_result.reply, intent="feedback"
            )
            return feedback_result.reply, "feedback"

    if feedback_agent.should_ask(session_id) and feedback_agent.is_closing_message(message):
        reply = _append_feedback_ask(db, session_id, CLOSING_RESPONSE)
        save_chat_history(db, session_id=session_id, message=message, response=reply, intent="closing")
        return reply, "closing"

    if is_collecting(session_id):
        lead_result = handle_lead_message(db, session_id, message)

        if lead_result.consumed:
            reply = lead_result.reply
            if lead_result.finalized and feedback_agent.should_ask(session_id):
                reply = _append_feedback_ask(db, session_id, reply)
            save_chat_history(db, session_id=session_id, message=message, response=reply, intent="lead")
            return reply, "lead"
        reply, intent_label = _route(db, session_id, message, skip_handler_intents={"lead"})
        if intent_label == "lead":
            combined_reply = lead_result.reply
        elif intent_label == "off_topic":
            combined_reply = reply
        else:
            combined_reply = f"{reply}\n\n{lead_result.reply}"

        save_chat_history(db, session_id=session_id, message=message, response=combined_reply, intent=intent_label)
        return combined_reply, intent_label

    reply, intent_label = _route(db, session_id, message)

    if intent_label in ("off_topic", "greeting"):
        return reply, intent_label

    if intent_label == "escalate" and feedback_agent.should_ask(session_id):
        reply = _append_feedback_ask(db, session_id, reply)

    save_chat_history(db, session_id=session_id, message=message, response=reply, intent=intent_label)
    return reply, intent_label


def handle_message(db: Session, session_id: str, message: str) -> str:
    reply, _ = _handle_message_impl(db, session_id, message)
    return reply


def handle_message_with_intent(db: Session, session_id: str, message: str) -> tuple[str, str]:
    """Same as handle_message but also returns the classified intent label,
    so callers (the /chat route) can tell the frontend when a greeting fired
    - used to re-show the quick-reply buttons under every greeting, not just
    the very first one, without changing handle_message's existing str
    return type that other callers/tests depend on."""
    return _handle_message_impl(db, session_id, message)
