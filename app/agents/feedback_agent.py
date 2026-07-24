"""
Feedback agent: after a conversation reaches a natural close point (a lead
gets submitted, an escalation happens, or the user signs off with something
like "thanks"/"bye"), casually ask for a quick rating/comment and save
whatever comes back to the Feedback table.

Deliberately AI-free and deterministic, same reasoning as intent_agent's
pure-code pre-checks: this only needs to recognize a small, well-known shape
of response (a 1-5 number, or a handful of rating adjectives), so a regex/
heuristic layer is both cheaper and more predictable than a round-trip
through the AI classifier - and since is_awaiting_response() is checked
before classify_intent ever runs (see orchestrator.py), a feedback reply can
never be misrouted to another agent or refused as off-topic in the first
place.

Only an explicit signal (a number, or a recognized rating word) counts as
feedback - a short, non-question remark with neither is NOT treated as a
free-form comment, even though that misses some genuine free-text feedback
("loved chatting with you!"). A real check-in like "hey are you still
there?" or "one more thing" is just as short and non-question-shaped as
actual feedback, so accepting anything short-and-not-a-question caused
exactly that kind of message to get silently swallowed as a "comment"
instead of being declined and handled normally - a worse failure mode than
occasionally missing a comment that doesn't use a common rating word.

Ask/decline state is tracked per session_id, in memory, same
single-process caveat as lead_agent's _sessions. Once a session has been
asked, it's asked at most once: given feedback, declined, or ignored all
latch the state so the prompt never repeats and never blocks normal
conversation afterward.
"""
import re
import threading
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.database import save_feedback
from app.prompt import FEEDBACK_THANKS_COMMENT, FEEDBACK_THANKS_RATING

_CLOSING_TOKEN = (
    r"(?:thanks?|thank\s+you|thx|ty|cheers|that'?s\s+all|that'?ll\s+be\s+all|"
    r"no(?:pe)?\s*that'?s\s+it|all\s+good|bye+|goodbye|see\s+ya|see\s+you|cya|"
    r"gotta\s+go|talk\s+later)"
)
# A message made up of ONLY closing tokens (one or more, e.g. "thanks, bye!")
# separated by whitespace/punctuation - anchored end to end so a real
# sentence that merely contains "thanks" ("thanks for the info, what's the
# pricing...") is never mistaken for a sign-off.
_CLOSING_RE = re.compile(
    rf"^\s*{_CLOSING_TOKEN}(?:\s*[,!.]*\s*{_CLOSING_TOKEN})*\s*[!.,]*\s*$",
    re.IGNORECASE,
)


def is_closing_message(message: str) -> bool:
    return bool(_CLOSING_RE.match(message.strip()))


_NUMERIC_RATING_RE = re.compile(r"^\s*([1-5])\s*(?:/\s*5)?\s*[.,!-]?\s*(.*)$")

_WORD_RATING_MAP = {
    "excellent": 5, "amazing": 5, "awesome": 5, "perfect": 5, "fantastic": 5,
    "great": 4, "good": 4, "nice": 4, "helpful": 4,
    "okay": 3, "ok": 3, "fine": 3, "decent": 3, "average": 3, "meh": 3, "alright": 3,
    "bad": 2, "poor": 2, "disappointing": 2,
    "terrible": 1, "awful": 1, "horrible": 1, "useless": 1,
}
_WORD_RATING_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _WORD_RATING_MAP) + r")\b", re.IGNORECASE
)


def _extract_rating(stripped: str) -> tuple[int | None, str]:
    """Returns (rating, remainder-after-the-number) for a leading "4", "4/5",
    "4 - loved it" style reply, or (None, stripped) if it doesn't start with
    a number."""
    m = _NUMERIC_RATING_RE.match(stripped)
    if m:
        return int(m.group(1)), m.group(2).strip()
    return None, stripped


def _extract_word_rating(stripped: str) -> int | None:
    m = _WORD_RATING_RE.search(stripped.lower())
    return _WORD_RATING_MAP[m.group(1)] if m else None


@dataclass
class FeedbackStepResult:
    reply: str
    handled: bool  # True if this message was saved as feedback (rating and/or comment)


_lock = threading.Lock()
# session_id -> "asked" | "declined" | "given"
_feedback_state: dict[str, str] = {}


def should_ask(session_id: str) -> bool:
    """True if this session hasn't been asked for feedback yet at all."""
    with _lock:
        return session_id not in _feedback_state


def mark_asked(session_id: str) -> None:
    with _lock:
        _feedback_state[session_id] = "asked"


def is_awaiting_response(session_id: str) -> bool:
    with _lock:
        return _feedback_state.get(session_id) == "asked"


def reset(session_id: str) -> None:
    with _lock:
        _feedback_state.pop(session_id, None)


def handle_feedback_response(db: Session, session_id: str, message: str) -> FeedbackStepResult:
    """Only call this when is_awaiting_response(session_id) is True."""
    stripped = message.strip()

    rating, remainder = _extract_rating(stripped)

    if rating is not None:
        # A leading 1-5 was actually found - anything after it is the comment.
        comments = remainder or None
    else:
        word_rating = _extract_word_rating(stripped)
        if word_rating is not None:
            rating = word_rating
            comments = stripped
        else:
            comments = None

    if rating is None and comments is None:
        # Ignored, or moved on to something else entirely - decline for the
        # rest of this session (never re-ask) and let the caller route this
        # message normally instead of swallowing it.
        with _lock:
            _feedback_state[session_id] = "declined"
        return FeedbackStepResult(reply="", handled=False)

    save_feedback(db, session_id=session_id, rating=rating, comments=comments)
    with _lock:
        _feedback_state[session_id] = "given"
    reply = FEEDBACK_THANKS_RATING if rating is not None else FEEDBACK_THANKS_COMMENT
    return FeedbackStepResult(reply=reply, handled=True)
