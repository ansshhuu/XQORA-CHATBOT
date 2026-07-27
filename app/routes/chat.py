import time
import uuid
from collections import defaultdict, deque

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.agents import feedback_agent
from app.core.config import MAX_MESSAGE_LENGTH, RATE_LIMIT_PER_MINUTE
from app.database import get_db
from app.orchestrator import handle_message_with_intent

router = APIRouter()

_RATE_WINDOW_SECONDS = 60

_request_log: dict[str, deque] = defaultdict(deque)

_SWEEP_INTERVAL_SECONDS = 300
_last_sweep = 0.0


def _sweep_stale_entries(now: float) -> None:
    """Drop rate-limit entries for IPs that haven't made a request in over a
    window, so the table doesn't grow unbounded over the life of the process.
    Runs at most once per _SWEEP_INTERVAL_SECONDS to keep the cost negligible."""
    global _last_sweep
    if now - _last_sweep < _SWEEP_INTERVAL_SECONDS:
        return
    _last_sweep = now
    stale_keys = [key for key, log in _request_log.items() if not log or now - log[-1] > _RATE_WINDOW_SECONDS]
    for key in stale_keys:
        del _request_log[key]


_MAX_SESSION_ID_LENGTH = 128


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    session_id: str | None = Field(default=None, max_length=_MAX_SESSION_ID_LENGTH)
    user_id: str | None = None

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message cannot be empty or whitespace-only")
        return value


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    awaiting_feedback: bool = False
    is_greeting: bool = False


def _enforce_rate_limit(key: str) -> None:
    now = time.monotonic()
    _sweep_stale_entries(now)
    log = _request_log[key]
    while log and now - log[0] > _RATE_WINDOW_SECONDS:
        log.popleft()
    if len(log) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down and try again shortly.")
    log.append(now)


@router.post("", response_model=ChatResponse)
def handle_chat(
    request: ChatRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    session_id = request.session_id or str(uuid.uuid4())
    client_ip = http_request.client.host if http_request.client else "unknown"
    _enforce_rate_limit(client_ip)

    reply, intent_label = handle_message_with_intent(db, session_id, request.message, background_tasks)
    awaiting_feedback = feedback_agent.is_awaiting_response(session_id)
    return ChatResponse(
        reply=reply,
        session_id=session_id,
        awaiting_feedback=awaiting_feedback,
        is_greeting=intent_label == "greeting",
    )
