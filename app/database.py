"""
SQLAlchemy setup + models: Leads, ChatHistory, FAQ, Feedback.
"""
import re
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import DATABASE_URL

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Lead(Base):
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(128), index=True, nullable=False)
    name = Column(String(200))
    company = Column(String(200))
    email = Column(String(200))
    phone = Column(String(50))
    service = Column(String(200))
    budget = Column(String(100))
    message = Column(Text)
    forwarded = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(128), index=True, nullable=False)
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    intent = Column(String(50))
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class FAQ(Base):
    __tablename__ = "faqs"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)


class LeadSession(Base):
    """Durable store for in-progress (not yet completed) lead-collection
    state, keyed by session_id. lead_agent.py loads/saves this on every
    lead-flow turn so a server restart mid-form doesn't lose partially
    collected fields - the in-memory session cache it also keeps is just a
    same-process performance cache in front of this table now, not the
    source of truth."""

    __tablename__ = "lead_sessions"

    session_id = Column(String(128), primary_key=True)
    name = Column(String(200))
    company = Column(String(200))
    email = Column(String(200))
    phone = Column(String(50))
    message = Column(Text)
    awaiting_field = Column(String(50))
    updated_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc)
    )


class Feedback(Base):
    """session_id links back to the same session used in ChatHistory, so a
    feedback row can be traced to the conversation it's about."""

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(128), index=True, nullable=False)
    rating = Column(Integer)
    comments = Column(Text)
    date = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def sanitize_input(value: str | None, max_length: int = 2000) -> str:
    """Strip control chars / null bytes and cap length before DB storage.

    ORM uses parameterized queries (no raw SQL string-building), so this is a
    defense-in-depth layer against stored-XSS payloads and control-char abuse,
    not the primary SQL-injection defense.
    """
    if not value:
        return ""
    cleaned = _CONTROL_CHARS.sub("", value).strip()
    return cleaned[:max_length]


def save_chat_history(db: Session, session_id: str, message: str, response: str, intent: str = "") -> ChatHistory:
    entry = ChatHistory(
        session_id=sanitize_input(session_id, 128),
        message=sanitize_input(message),
        response=sanitize_input(response, 4000),
        intent=sanitize_input(intent, 50),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def save_feedback(
    db: Session, session_id: str, rating: int | None = None, comments: str | None = None
) -> Feedback:
    entry = Feedback(
        session_id=sanitize_input(session_id, 128),
        rating=rating,
        comments=sanitize_input(comments, 2000) if comments else None,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def load_lead_session(db: Session, session_id: str) -> dict | None:
    """Returns the persisted in-progress lead state for `session_id`, shaped
    to match lead_agent's in-memory state dict (LEAD_FIELDS plus
    "_awaiting"), or None if nothing's been saved for it (fresh session, or
    already completed/cancelled)."""
    row = db.get(LeadSession, session_id)
    if row is None:
        return None
    return {
        "name": row.name,
        "company": row.company,
        "email": row.email,
        "phone": row.phone,
        "message": row.message,
        "_awaiting": row.awaiting_field,
    }


def save_lead_session(db: Session, session_id: str, state: dict) -> None:
    """Upserts the given lead_agent state dict into the lead_sessions table."""
    row = db.get(LeadSession, session_id)
    if row is None:
        row = LeadSession(session_id=session_id)
        db.add(row)
    row.name = state.get("name")
    row.company = state.get("company")
    row.email = state.get("email")
    row.phone = state.get("phone")
    row.message = state.get("message")
    row.awaiting_field = state.get("_awaiting")
    db.commit()


def delete_lead_session(db: Session, session_id: str) -> None:
    row = db.get(LeadSession, session_id)
    if row is not None:
        db.delete(row)
        db.commit()


def delete_stale_lead_sessions(db: Session, older_than: datetime) -> int:
    """Clears abandoned in-progress lead forms last touched before
    `older_than`, mirroring lead_agent's in-memory TTL sweep so the table
    doesn't grow forever with forms nobody ever came back to finish."""
    deleted = db.query(LeadSession).filter(LeadSession.updated_at < older_than).delete()
    db.commit()
    return deleted


def get_recent_chat_history(db: Session, session_id: str, limit: int = 3) -> list[ChatHistory]:
    """Most recent `limit` turns for this session, oldest first, for building
    conversation context. Excludes the current in-flight turn (not saved yet
    when the caller is still deciding how to handle it)."""
    rows = (
        db.query(ChatHistory)
        .filter(ChatHistory.session_id == session_id)
        .order_by(ChatHistory.id.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(rows))
