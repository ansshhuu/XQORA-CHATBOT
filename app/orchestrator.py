"""
Single decision point for /chat: classify intent (guardrail-checked, context-
aware), route to the matching specialist agent, and log the turn to Chat
History.
"""
import logging

from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from app.agents import feedback_agent
from app.agents.escalation_agent import get_escalation_response
from app.agents.faq_agent import get_faq_response
from app.agents.intent_agent import classify_intent
from app.agents.lead_agent import handle_lead_message, is_collecting
from app.agents.recommend_agent import get_recommendation_response
from app.database import get_recent_chat_history, save_chat_history
from app.prompt import CLOSING_RESPONSE, FEEDBACK_ASK

logger = logging.getLogger("xqora.orchestrator")

_RECENT_CONTEXT_TURNS = 3

_ROUTES = {
    "faq": lambda db, session_id, message, background_tasks, recent_context: get_faq_response(
        message, recent_context
    ),
    "recommend": lambda db, session_id, message, background_tasks, recent_context: get_recommendation_response(
        message, recent_context
    ),
    "lead": lambda db, session_id, message, background_tasks, recent_context: handle_lead_message(
        db, session_id, message, background_tasks
    ).reply,
    "escalate": lambda db, session_id, message, background_tasks, recent_context: get_escalation_response(),
}


def _build_recent_context(rows: list) -> str | None:
    if not rows:
        return None
    lines = []
    for row in rows:
        lines.append(f"User: {row.message}")
        lines.append(f"Bot: {row.response}")
    return "\n".join(lines)


def _route(db: Session, session_id: str, message: str, background_tasks: BackgroundTasks) -> tuple[str, str]:
    """Classify + dispatch a message that's NOT a lead-collection answer.
    Returns (reply, intent_label) for the caller to log."""
    rows = get_recent_chat_history(db, session_id, limit=_RECENT_CONTEXT_TURNS)
    recent_context = _build_recent_context(rows)
    previous_intent = rows[-1].intent if rows else None
    result = classify_intent(message, recent_context=recent_context, previous_intent=previous_intent)

    if result.blocked:
        return result.refusal_message, result.intent

    handler = _ROUTES.get(result.intent, lambda *_: get_escalation_response())
    reply = handler(db, session_id, message, background_tasks, recent_context)
    return reply, result.intent


def _append_feedback_ask(session_id: str, reply: str) -> str:
    feedback_agent.mark_asked(session_id)
    return f"{reply}\n\n{FEEDBACK_ASK}"


def handle_message(db: Session, session_id: str, message: str, background_tasks: BackgroundTasks) -> str:
    if feedback_agent.is_awaiting_response(session_id):
        feedback_result = feedback_agent.handle_feedback_response(db, session_id, message)
        if feedback_result.handled:
            save_chat_history(
                db, session_id=session_id, message=message, response=feedback_result.reply, intent="feedback"
            )
            return feedback_result.reply

    if feedback_agent.should_ask(session_id) and feedback_agent.is_closing_message(message):
        reply = _append_feedback_ask(session_id, CLOSING_RESPONSE)
        save_chat_history(db, session_id=session_id, message=message, response=reply, intent="closing")
        return reply

    if is_collecting(session_id):
        lead_result = handle_lead_message(db, session_id, message, background_tasks)

        if lead_result.consumed:
            reply = lead_result.reply
            if lead_result.finalized and feedback_agent.should_ask(session_id):
                reply = _append_feedback_ask(session_id, reply)
            save_chat_history(db, session_id=session_id, message=message, response=reply, intent="lead")
            return reply
        reply, intent_label = _route(db, session_id, message, background_tasks)
        if intent_label == "lead":
            combined_reply = lead_result.reply
        elif intent_label == "off_topic":
            combined_reply = reply
        else:
            combined_reply = f"{reply}\n\n{lead_result.reply}"

        save_chat_history(db, session_id=session_id, message=message, response=combined_reply, intent=intent_label)
        return combined_reply

    reply, intent_label = _route(db, session_id, message, background_tasks)

    if intent_label in ("off_topic", "greeting"):
        return reply

    if intent_label == "escalate" and feedback_agent.should_ask(session_id):
        reply = _append_feedback_ask(session_id, reply)

    save_chat_history(db, session_id=session_id, message=message, response=reply, intent=intent_label)
    return reply
