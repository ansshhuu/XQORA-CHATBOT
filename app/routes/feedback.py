import time
from collections import defaultdict, deque
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agents import feedback_agent
from app.core.config import RATE_LIMIT_PER_MINUTE
from app.database import get_db

router = APIRouter()

_RATE_WINDOW_SECONDS = 60

_request_log: dict[str, deque] = defaultdict(deque)

_SWEEP_INTERVAL_SECONDS = 300
_last_sweep = 0.0


def _sweep_stale_entries(now: float) -> None:
    """Same stale-IP eviction as chat.py's rate limiter, kept as a separate
    counter/table here since this route has its own request volume."""
    global _last_sweep
    if now - _last_sweep < _SWEEP_INTERVAL_SECONDS:
        return
    _last_sweep = now
    stale_keys = [key for key, log in _request_log.items() if not log or now - log[-1] > _RATE_WINDOW_SECONDS]
    for key in stale_keys:
        del _request_log[key]


def _enforce_rate_limit(key: str) -> None:
    now = time.monotonic()
    _sweep_stale_entries(now)
    log = _request_log[key]
    while log and now - log[0] > _RATE_WINDOW_SECONDS:
        log.popleft()
    if len(log) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down and try again shortly.")
    log.append(now)


_MAX_SESSION_ID_LENGTH = 128


class FeedbackSubmitRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=_MAX_SESSION_ID_LENGTH)
    rating: Literal["happy", "ok", "sad"]
    reason: Literal["bad_app_experience", "didnt_understand_query", "slow_response", "other"] | None = None
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackCancelRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=_MAX_SESSION_ID_LENGTH)


class FeedbackResponse(BaseModel):
    reply: str
    handled: bool


@router.post("/submit", response_model=FeedbackResponse)
def submit_feedback(
    request: FeedbackSubmitRequest,
    http_request: Request,
    db: Session = Depends(get_db),
):
    client_ip = http_request.client.host if http_request.client else "unknown"
    _enforce_rate_limit(client_ip)

    result = feedback_agent.record_structured_rating(
        db,
        session_id=request.session_id,
        rating=request.rating,
        reason=request.reason,
        comment=request.comment,
    )
    return FeedbackResponse(reply=result.reply, handled=result.handled)


@router.post("/cancel", response_model=FeedbackResponse)
def cancel_feedback(
    request: FeedbackCancelRequest,
    http_request: Request,
):
    client_ip = http_request.client.host if http_request.client else "unknown"
    _enforce_rate_limit(client_ip)

    feedback_agent.mark_declined(request.session_id)
    return FeedbackResponse(reply="", handled=True)
