import time
import uuid
from collections import defaultdict, deque

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.core.config import MAX_MESSAGE_LENGTH, RATE_LIMIT_PER_MINUTE
from app.database import get_db
from app.orchestrator import handle_message

router = APIRouter()

_RATE_WINDOW_SECONDS = 60
# In-memory per-process limiter (key -> recent request timestamps). Fine for a
# single-instance deployment; a multi-worker/multi-instance deployment would
# need a shared store (e.g. Redis) instead.
_request_log: dict[str, deque] = defaultdict(deque)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    session_id: str | None = None
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


def _enforce_rate_limit(key: str) -> None:
    now = time.monotonic()
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
    # Keyed on IP alone, not IP+session: session_id is client-supplied and a
    # fresh UUID is minted whenever it's omitted, so keying on the pair would
    # let spam bypass the limiter simply by never sending a session_id.
    client_ip = http_request.client.host if http_request.client else "unknown"
    _enforce_rate_limit(client_ip)

    reply = handle_message(db, session_id, request.message, background_tasks)
    return ChatResponse(reply=reply, session_id=session_id)
