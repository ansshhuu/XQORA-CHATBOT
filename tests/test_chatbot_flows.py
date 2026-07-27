"""
Automated coverage for the manual chatbot test-message checklist: greeting,
each widget quick-reply button, the FAQ/recommend/escalate workers, the full
lead-collection form (through to the DB row + forwarding calls), lead
cancellation, self-recall, off-topic refusal, the feedback card (all three
faces + a reason chip + cancel), the feedback free-text fallback, cross-
session isolation, state surviving a simulated restart, and a defensive
check that no AI reasoning-trace leak (a literal "<think>" tag) ever reaches
a user-facing reply.

Runs entirely mocked (no real Groq/Resend/Sheets calls) so it's fast and
deterministic enough to run before every deploy - same house style as
tests/test_e2e.py's `--mocked` mode (plain script, asserts, PASS/FAIL
summary, exits 1 on any failure so it's CI-friendly as one command:

    python tests/test_chatbot_flows.py
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_chatbot_flows_test.db")
if os.path.exists(TEST_DB_PATH):
    os.remove(TEST_DB_PATH)
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "1000")

from fastapi.testclient import TestClient  # noqa: E402

from app.agents import feedback_agent, lead_agent  # noqa: E402
from app.database import (  # noqa: E402
    ChatHistory,
    Feedback,
    Lead,
    SessionLocal,
    engine,
    init_db,
    load_feedback_state,
    load_lead_session,
)
from app.main import app  # noqa: E402
from app.prompt import (  # noqa: E402
    BARE_CATEGORY_RESPONSE,
    CLARIFY_BARE_WORD_RESPONSE,
    FEEDBACK_ASK,
    FEEDBACK_THANKS_RATING,
    GUARDRAIL_REFUSAL_MESSAGE,
)
from app.services import ai_service, session_service  # noqa: E402

results = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((label, status, detail))
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and status == "FAIL" else ""))


def last_chat_history_row(session_id):
    db = SessionLocal()
    try:
        return (
            db.query(ChatHistory)
            .filter(ChatHistory.session_id == session_id)
            .order_by(ChatHistory.id.desc())
            .first()
        )
    finally:
        db.close()


def latest_lead(session_id):
    db = SessionLocal()
    try:
        return db.query(Lead).filter(Lead.session_id == session_id).order_by(Lead.id.desc()).first()
    finally:
        db.close()


def latest_feedback(session_id):
    db = SessionLocal()
    try:
        return db.query(Feedback).filter(Feedback.session_id == session_id).order_by(Feedback.id.desc()).first()
    finally:
        db.close()


def reset_session(session_id):
    session_service.reset_lead_state(session_id)
    session_service._completed_lead_cache.evict_memory_only(session_id)
    feedback_agent.reset(session_id)


# ---------------------------------------------------------------------------
# 1. Greeting
# ---------------------------------------------------------------------------


def _test_greeting(client):
    r = client.post("/chat", json={"message": "hi"})
    check("Greeting: HTTP 200", r.status_code == 200, r.text)
    body = r.json()
    check(
        "Greeting: friendly response, not the off-topic refusal",
        "outside what i can help" not in body["reply"].lower() and len(body["reply"]) > 0,
        body["reply"],
    )
    row = last_chat_history_row(body["session_id"])
    check("Greeting: not logged as a real intent (blocked pre-agent)", row is None, str(row))


# ---------------------------------------------------------------------------
# 2. Each widget quick-reply button (see static/widget.js QUICK_REPLIES)
# ---------------------------------------------------------------------------


def _test_quick_reply_hire_you(client):
    session_id = "flows-quickreply-hire"
    reset_session(session_id)

    with patch("app.agents.intent_agent.get_ai_response", return_value="lead"):
        r = client.post("/chat", json={"message": "I want to hire you", "session_id": session_id})
    check("Quick reply 'I want to hire you': HTTP 200", r.status_code == 200, r.text)
    body = r.json()
    check(
        "Quick reply 'I want to hire you': starts the lead flow (asks for name)",
        "name" in body["reply"].lower(),
        body["reply"],
    )
    row = last_chat_history_row(session_id)
    check("Quick reply 'I want to hire you': logged with intent=lead", row is not None and row.intent == "lead", str(row))
    reset_session(session_id)


def _test_quick_reply_services(client):
    session_id = "flows-quickreply-services"
    reset_session(session_id)

    with patch("app.agents.intent_agent.get_ai_response", return_value="recommend"), patch(
        "app.agents.recommend_agent.get_ai_response",
        return_value="We build websites, AI chatbots, cloud solutions, and mobile apps. What are you working on?",
    ):
        r = client.post("/chat", json={"message": "Tell me your services", "session_id": session_id})
    check("Quick reply 'Tell me your services': HTTP 200", r.status_code == 200, r.text)
    body = r.json()
    row = last_chat_history_row(session_id)
    check(
        "Quick reply 'Tell me your services': routed to recommend + logged",
        row is not None and row.intent == "recommend",
        str(row),
    )
    check("Quick reply 'Tell me your services': reply matches the worker output", "chatbots" in body["reply"].lower(), body["reply"])
    reset_session(session_id)


def _test_quick_reply_what_is_xqora(client):
    session_id = "flows-quickreply-whatisxqora"
    reset_session(session_id)

    with patch("app.agents.intent_agent.get_ai_response", return_value="faq"), patch(
        "app.agents.faq_agent.get_ai_response",
        return_value="XQORA Technologies builds websites, AI automation/chatbots, cloud solutions, and mobile apps.",
    ):
        r = client.post("/chat", json={"message": "What is XQORA?", "session_id": session_id})
    check("Quick reply 'What is XQORA?': HTTP 200", r.status_code == 200, r.text)
    body = r.json()
    row = last_chat_history_row(session_id)
    check("Quick reply 'What is XQORA?': routed to faq + logged", row is not None and row.intent == "faq", str(row))
    check("Quick reply 'What is XQORA?': reply matches the worker output", "xqora" in body["reply"].lower(), body["reply"])
    reset_session(session_id)


# ---------------------------------------------------------------------------
# 3. FAQ worker / 4. Recommend worker (grounded output, correct intent logged)
# ---------------------------------------------------------------------------


def _test_faq_worker(client):
    session_id = "flows-faq-worker"
    reset_session(session_id)
    with patch("app.agents.intent_agent.get_ai_response", return_value="faq"), patch(
        "app.agents.faq_agent.get_ai_response", return_value="Our AI automation service handles repetitive support tickets."
    ):
        r = client.post("/chat", json={"message": "What services does XQORA provide?", "session_id": session_id})
    check("FAQ worker: HTTP 200", r.status_code == 200, r.text)
    body = r.json()
    check("FAQ worker: reply is the grounded FAQ answer", "automation" in body["reply"].lower(), body["reply"])
    row = last_chat_history_row(session_id)
    check("FAQ worker: logged with intent=faq", row is not None and row.intent == "faq", str(row))
    reset_session(session_id)


def _test_recommend_worker(client):
    session_id = "flows-recommend-worker"
    reset_session(session_id)
    with patch("app.agents.intent_agent.get_ai_response", return_value="recommend"), patch(
        "app.agents.recommend_agent.get_ai_response",
        return_value="Sounds like our AI chatbot service could help. What's your current support setup like?",
    ):
        r = client.post(
            "/chat", json={"message": "I need a chatbot built for my business, what do you suggest?", "session_id": session_id}
        )
    check("Recommend worker: HTTP 200", r.status_code == 200, r.text)
    body = r.json()
    check("Recommend worker: reply is the grounded recommend answer", "chatbot" in body["reply"].lower(), body["reply"])
    row = last_chat_history_row(session_id)
    check("Recommend worker: logged with intent=recommend", row is not None and row.intent == "recommend", str(row))
    reset_session(session_id)


# ---------------------------------------------------------------------------
# 5. Full lead-form flow: fill every field, check the DB row + forwarding
# ---------------------------------------------------------------------------


def _test_full_lead_form_flow(client):
    session_id = "flows-full-lead-form"
    reset_session(session_id)

    with patch("app.agents.intent_agent.get_ai_response", return_value="lead"), patch(
        "app.agents.lead_agent.get_ai_response", return_value="YES"
    ), patch("app.agents.lead_agent.forward_lead", return_value=True) as mock_email, patch(
        "app.agents.lead_agent.append_lead_row", return_value=True
    ) as mock_sheets:
        r1 = client.post("/chat", json={"message": "I want to start a project and get a quote", "session_id": session_id})
        check("Lead form: turn 1 (trigger) asks for name", "name" in r1.json()["reply"].lower(), r1.text)

        r2 = client.post("/chat", json={"message": "Taylor Morgan", "session_id": session_id})
        check("Lead form: turn 2 (name) asks for company", "company" in r2.json()["reply"].lower(), r2.text)

        r3 = client.post("/chat", json={"message": "Morgan Studios", "session_id": session_id})
        check("Lead form: turn 3 (company) asks for email", "email" in r3.json()["reply"].lower(), r3.text)

        r4 = client.post("/chat", json={"message": "taylor@morganstudios.com", "session_id": session_id})
        check("Lead form: turn 4 (email) asks for phone", "phone" in r4.json()["reply"].lower(), r4.text)

        r5 = client.post("/chat", json={"message": "+1 555 987 6543", "session_id": session_id})
        check(
            "Lead form: turn 5 (phone) asks what they need help with",
            "help" in r5.json()["reply"].lower(),
            r5.text,
        )

        r6 = client.post("/chat", json={"message": "Need a full e-commerce rebuild", "session_id": session_id})
        body6 = r6.json()
        check("Lead form: turn 6 (message) finalizes the lead", "thanks" in body6["reply"].lower(), r6.text)

    check("Lead form: email forward attempted (background task ran)", mock_email.called, None)
    check("Lead form: sheet row write attempted (background task ran)", mock_sheets.called, None)
    if mock_sheets.called:
        sheet_args = mock_sheets.call_args.args
        check(
            "Lead form: sheet row write got the right lead data",
            sheet_args[0] == "Taylor Morgan" and sheet_args[1] == "Morgan Studios",
            sheet_args,
        )

    lead = latest_lead(session_id)
    check("Lead form: DB row created", lead is not None, None)
    if lead is not None:
        check(
            "Lead form: DB row has every field captured correctly",
            lead.name == "Taylor Morgan"
            and lead.company == "Morgan Studios"
            and lead.email == "taylor@morganstudios.com"
            and lead.phone == "+15559876543"
            and "e-commerce" in (lead.message or "").lower(),
            lead,
        )
        check("Lead form: DB row marked forwarded (email send succeeded)", lead.forwarded is True, lead)

    check(
        "Lead form: in-progress lead_sessions row cleared after finalize",
        not lead_agent.is_collecting(session_id),
        None,
    )
    reset_session(session_id)


# ---------------------------------------------------------------------------
# 6. Lead cancel mid-form
# ---------------------------------------------------------------------------


def _test_lead_cancel_mid_form(client):
    session_id = "flows-lead-cancel"
    reset_session(session_id)

    with patch("app.agents.intent_agent.get_ai_response", return_value="lead"):
        r1 = client.post("/chat", json={"message": "I want to start a project and get a quote", "session_id": session_id})
        check("Lead cancel: flow starts, asks for name", "name" in r1.json()["reply"].lower(), r1.text)

        r2 = client.post("/chat", json={"message": "never mind", "session_id": session_id})
    body2 = r2.json()
    check("Lead cancel: HTTP 200", r2.status_code == 200, r2.text)
    check("Lead cancel: acknowledges the cancellation", "cancel" in body2["reply"].lower(), body2["reply"])
    check("Lead cancel: no longer mid-collection", not lead_agent.is_collecting(session_id), None)
    check("Lead cancel: no Lead row was created", latest_lead(session_id) is None, None)

    db = SessionLocal()
    try:
        persisted = load_lead_session(db, session_id)
    finally:
        db.close()
    check("Lead cancel: persisted lead_sessions row also cleared", persisted is None, persisted)
    reset_session(session_id)


# ---------------------------------------------------------------------------
# 7. Self-recall: bare name, and an explicit "show my info" ask
# ---------------------------------------------------------------------------


def _test_self_recall(client):
    session_id = "flows-self-recall"
    reset_session(session_id)

    with patch("app.agents.intent_agent.get_ai_response", return_value="lead"), patch(
        "app.agents.lead_agent.get_ai_response", return_value="YES"
    ), patch("app.agents.lead_agent.forward_lead", return_value=True), patch(
        "app.agents.lead_agent.append_lead_row", return_value=True
    ):
        client.post("/chat", json={"message": "I want to start a project and get a quote", "session_id": session_id})
        client.post("/chat", json={"message": "Jordan Lee", "session_id": session_id})
        client.post("/chat", json={"message": "Lee Consulting", "session_id": session_id})
        client.post("/chat", json={"message": "jordan@leeconsulting.com", "session_id": session_id})
        client.post("/chat", json={"message": "+1 555 111 2222", "session_id": session_id})
        client.post("/chat", json={"message": "Need help with a new landing page", "session_id": session_id})

    # Bare name recall - no AI mock needed, _own_info_reply is checked before
    # classify_intent ever runs.
    r_bare = client.post("/chat", json={"message": "Jordan", "session_id": session_id})
    check("Self-recall (bare name): HTTP 200", r_bare.status_code == 200, r_bare.text)
    check(
        "Self-recall (bare name): casual ack referencing the stored name",
        "jordan" in r_bare.json()["reply"].lower(),
        r_bare.json()["reply"],
    )
    row_bare = last_chat_history_row(session_id)
    check(
        "Self-recall (bare name): logged as own_info, not misrouted through an AI call",
        row_bare is not None and row_bare.intent == "own_info",
        str(row_bare),
    )

    # Explicit "show my info" style ask - dumps every known field.
    r_explicit = client.post("/chat", json={"message": "show me my details", "session_id": session_id})
    check("Self-recall (explicit): HTTP 200", r_explicit.status_code == 200, r_explicit.text)
    explicit_reply_lower = r_explicit.json()["reply"].lower()
    check(
        "Self-recall (explicit): dumps name/company/email/phone/message all shared so far",
        all(
            v in explicit_reply_lower
            for v in ["jordan lee", "lee consulting", "jordan@leeconsulting.com", "landing page"]
        ),
        r_explicit.json()["reply"],
    )
    reset_session(session_id)


# ---------------------------------------------------------------------------
# 8. Escalate worker
# ---------------------------------------------------------------------------


def _test_escalate_worker(client):
    session_id = "flows-escalate"
    reset_session(session_id)
    # A tender + XQORA-service mention trips hard_escalate_signal, a
    # pure-code deterministic pre-check - no AI mock needed.
    r = client.post(
        "/chat",
        json={"message": "can you handle a large government tender project", "session_id": session_id},
    )
    check("Escalate worker: HTTP 200", r.status_code == 200, r.text)
    body = r.json()
    row = last_chat_history_row(session_id)
    check("Escalate worker: routed to escalate + logged", row is not None and row.intent == "escalate", str(row))
    check("Escalate worker: reply includes a contact email", "@" in body["reply"], body["reply"])
    check("Escalate worker: feedback ask appended after escalation", FEEDBACK_ASK in body["reply"], body["reply"])
    reset_session(session_id)


# ---------------------------------------------------------------------------
# 9. Off-topic refusal
# ---------------------------------------------------------------------------


def _test_off_topic_refusal(client):
    session_id = "flows-off-topic"
    reset_session(session_id)
    r = client.post("/chat", json={"message": "Write me a python script to sort a list", "session_id": session_id})
    check("Off-topic refusal: HTTP 200", r.status_code == 200, r.text)
    body = r.json()
    check("Off-topic refusal: matches the guardrail refusal text", body["reply"] == GUARDRAIL_REFUSAL_MESSAGE, body["reply"])
    row = last_chat_history_row(session_id)
    check("Off-topic refusal: NOT logged to Chat History (blocked pre-agent)", row is None, str(row))
    reset_session(session_id)


# ---------------------------------------------------------------------------
# 10. Feedback card: all 3 faces + a reason chip + submit, and cancel
# ---------------------------------------------------------------------------


def _test_feedback_card_faces(client):
    cases = [
        ("flows-feedback-happy", "happy", None, None, 5),
        ("flows-feedback-ok", "ok", "slow_response", "It took a while to answer.", 3),
        ("flows-feedback-sad", "sad", "didnt_understand_query", "It missed my question entirely.", 1),
    ]
    for session_id, face, reason, comment, expected_rating in cases:
        reset_session(session_id)
        payload = {"session_id": session_id, "rating": face}
        if reason:
            payload["reason"] = reason
        if comment:
            payload["comment"] = comment

        r = client.post("/feedback/submit", json=payload)
        check(f"Feedback card ({face}): HTTP 200", r.status_code == 200, r.text)
        body = r.json()
        check(f"Feedback card ({face}): reply is the rating thank-you, marked handled", body["handled"] is True, body)

        fb = latest_feedback(session_id)
        check(f"Feedback card ({face}): Feedback row saved", fb is not None, None)
        if fb is not None:
            check(
                f"Feedback card ({face}): rating mapped to the right 1-5 scale value",
                fb.rating == expected_rating,
                fb.rating,
            )
            if reason:
                check(f"Feedback card ({face}): reason chip saved", fb.reason == reason, fb.reason)
            if comment:
                check(f"Feedback card ({face}): comment saved", fb.comments == comment, fb.comments)

        check(
            f"Feedback card ({face}): session latched as 'given', no longer awaiting",
            not feedback_agent.is_awaiting_response(session_id) and not feedback_agent.should_ask(session_id),
            None,
        )
        reset_session(session_id)


def _test_feedback_card_cancel(client):
    session_id = "flows-feedback-cancel"
    reset_session(session_id)
    feedback_agent.mark_asked(session_id)
    check("Feedback card (cancel): session awaiting before cancel", feedback_agent.is_awaiting_response(session_id), None)

    r = client.post("/feedback/cancel", json={"session_id": session_id})
    check("Feedback card (cancel): HTTP 200", r.status_code == 200, r.text)
    check("Feedback card (cancel): no longer awaiting a response", not feedback_agent.is_awaiting_response(session_id), None)
    check("Feedback card (cancel): should not ask again in this session", not feedback_agent.should_ask(session_id), None)
    check("Feedback card (cancel): no Feedback row was saved", latest_feedback(session_id) is None, None)
    reset_session(session_id)


# ---------------------------------------------------------------------------
# 11. Feedback free-text fallback (typed instead of tapping a face)
# ---------------------------------------------------------------------------


def _test_feedback_text_fallback(client):
    session_id = "flows-feedback-text"
    reset_session(session_id)

    r1 = client.post("/chat", json={"message": "thanks, bye", "session_id": session_id})
    check("Feedback text fallback: closing message triggers the ask", FEEDBACK_ASK in r1.json()["reply"], r1.text)
    check("Feedback text fallback: session now awaiting a response", feedback_agent.is_awaiting_response(session_id), None)

    r2 = client.post("/chat", json={"message": "5 - super fast and helpful!", "session_id": session_id})
    body2 = r2.json()
    check("Feedback text fallback: plain thank-you reply, not misrouted elsewhere", body2["reply"] == FEEDBACK_THANKS_RATING, body2["reply"])
    check("Feedback text fallback: no longer awaiting once given", body2["awaiting_feedback"] is False, body2)

    fb = latest_feedback(session_id)
    check("Feedback text fallback: rating+comment saved to the Feedback table", fb is not None and fb.rating == 5, fb)
    if fb is not None:
        check("Feedback text fallback: free-text comment captured", "fast" in (fb.comments or "").lower(), fb.comments)
    reset_session(session_id)


def _test_feedback_text_word_rating(client):
    """A reply with a recognized rating word ("great") but no leading digit
    is accepted the same as a numeric rating - the word is mapped onto the
    1-5 scale and the whole message is kept as the comment. A message with
    NEITHER a number NOR a rating word is intentionally NOT treated as
    free-form feedback (see feedback_agent.py's module docstring) - it's
    declined instead, so there's no bare "just say thanks" case to test
    here without a rating signal in it."""
    session_id = "flows-feedback-text-word-rating"
    reset_session(session_id)
    feedback_agent.mark_asked(session_id)

    r = client.post("/chat", json={"message": "this was great, thanks so much", "session_id": session_id})
    body = r.json()
    check(
        "Feedback text fallback (word rating, no digit): rating thank-you reply",
        body["reply"] == FEEDBACK_THANKS_RATING,
        body["reply"],
    )
    fb = latest_feedback(session_id)
    check(
        "Feedback text fallback (word rating, no digit): 'great' mapped to rating 4",
        fb is not None and fb.rating == 4,
        fb,
    )
    if fb is not None:
        check(
            "Feedback text fallback (word rating, no digit): full message kept as the comment",
            fb.comments == "this was great, thanks so much",
            fb.comments,
        )
    reset_session(session_id)


# ---------------------------------------------------------------------------
# 12. Cross-session isolation
# ---------------------------------------------------------------------------


def _test_cross_session_isolation(client):
    session_a = "flows-isolation-a"
    session_b = "flows-isolation-b"
    reset_session(session_a)
    reset_session(session_b)

    with patch("app.agents.intent_agent.get_ai_response", return_value="lead"), patch(
        "app.agents.lead_agent.get_ai_response", return_value="YES"
    ):
        client.post("/chat", json={"message": "I want to start a project and get a quote", "session_id": session_a})
        client.post("/chat", json={"message": "I want to start a project and get a quote", "session_id": session_b})

        client.post("/chat", json={"message": "Amara Okafor", "session_id": session_a})
        client.post("/chat", json={"message": "Ben Cheng", "session_id": session_b})

        client.post("/chat", json={"message": "Okafor Digital", "session_id": session_a})
        client.post("/chat", json={"message": "Cheng Robotics", "session_id": session_b})

    state_a = session_service.get_lead_state(session_a)
    state_b = session_service.get_lead_state(session_b)
    check(
        "Cross-session isolation: session A kept its own data despite interleaving",
        state_a.get("name") == "Amara Okafor" and state_a.get("company") == "Okafor Digital",
        state_a,
    )
    check(
        "Cross-session isolation: session B kept its own data despite interleaving",
        state_b.get("name") == "Ben Cheng" and state_b.get("company") == "Cheng Robotics",
        state_b,
    )

    db = SessionLocal()
    try:
        persisted_a = load_lead_session(db, session_a)
        persisted_b = load_lead_session(db, session_b)
    finally:
        db.close()
    check(
        "Cross-session isolation: persisted lead_sessions rows are also kept separate",
        persisted_a is not None
        and persisted_a["name"] == "Amara Okafor"
        and persisted_b is not None
        and persisted_b["name"] == "Ben Cheng",
        (persisted_a, persisted_b),
    )

    # Chat history for each session must only ever contain that session's
    # own turns.
    db = SessionLocal()
    try:
        cross_bleed = (
            db.query(ChatHistory)
            .filter(ChatHistory.session_id == session_a, ChatHistory.message.like("%Ben Cheng%"))
            .first()
        )
    finally:
        db.close()
    check("Cross-session isolation: session B's message never landed in session A's history", cross_bleed is None, cross_bleed)

    reset_session(session_a)
    reset_session(session_b)


# ---------------------------------------------------------------------------
# 13. State survives a simulated restart (in-memory cache wiped, DB intact)
# ---------------------------------------------------------------------------


def _test_reload_persistence(client):
    session_id = "flows-reload-persistence"
    reset_session(session_id)

    with patch("app.agents.intent_agent.get_ai_response", return_value="lead"), patch(
        "app.agents.lead_agent.get_ai_response", return_value="YES"
    ):
        client.post("/chat", json={"message": "I want to start a project and get a quote", "session_id": session_id})
        client.post("/chat", json={"message": "Priya Nair", "session_id": session_id})

    db = SessionLocal()
    try:
        persisted = load_lead_session(db, session_id)
    finally:
        db.close()
    check(
        "Reload persistence: partial lead state landed in lead_sessions before any restart",
        persisted is not None and persisted["name"] == "Priya Nair" and persisted["_awaiting"] == "company",
        persisted,
    )

    # Simulate a process restart: wipe the in-memory cache + its lock. This
    # is exactly what a fresh worker process starts with.
    session_service._lead_cache.evict_memory_only(session_id)
    session_service._evict_session_lock(session_id)

    with patch("app.agents.intent_agent.get_ai_response", return_value="lead"), patch(
        "app.agents.lead_agent.get_ai_response", return_value="YES"
    ):
        r = client.post("/chat", json={"message": "Nair Analytics", "session_id": session_id})
    check(
        "Reload persistence: after simulated restart, flow resumes from DB (asks for email, not name again)",
        "email" in r.json()["reply"].lower() and "your name" not in r.json()["reply"].lower(),
        r.text,
    )

    # Feedback ask-state must also survive a restart - this is the gap the
    # session_service consolidation closed (previously in-memory only).
    feedback_session = "flows-reload-persistence-feedback"
    reset_session(feedback_session)
    feedback_agent.mark_asked(feedback_session)
    session_service._feedback_cache.evict_memory_only(feedback_session)

    db = SessionLocal()
    try:
        persisted_status = load_feedback_state(db, feedback_session)
    finally:
        db.close()
    check(
        "Reload persistence: feedback ask-state was actually written to feedback_states, not just memory",
        persisted_status == "asked",
        persisted_status,
    )
    check(
        "Reload persistence: feedback ask-state survives the simulated restart (cache-miss reads through to DB)",
        feedback_agent.is_awaiting_response(feedback_session),
        None,
    )

    reset_session(session_id)
    reset_session(feedback_session)


# ---------------------------------------------------------------------------
# 14. Vague/AI-guess-fallback input never leaks a raw <think> reasoning tag
# ---------------------------------------------------------------------------


def _test_vague_input_no_think_leak(client):
    session_id = "flows-vague-input"
    reset_session(session_id)

    # These two are deterministic pure-code paths (bare category word /
    # bare unclassified word) - no AI call happens at all, so this also
    # guards against a future refactor accidentally routing them through
    # the AI unstripped.
    r_pricing = client.post("/chat", json={"message": "pricing?", "session_id": session_id})
    check(
        "Vague input 'pricing?': deterministic clarifying reply, no AI call",
        r_pricing.json()["reply"] == BARE_CATEGORY_RESPONSE,
        r_pricing.text,
    )
    check("Vague input 'pricing?': no leaked <think> tag", "<think>" not in r_pricing.json()["reply"].lower(), r_pricing.text)

    r_help = client.post("/chat", json={"message": "something", "session_id": session_id})
    check(
        "Vague input 'something': deterministic clarifying reply, no AI call",
        r_help.json()["reply"] == CLARIFY_BARE_WORD_RESPONSE.format(word="something"),
        r_help.text,
    )
    check("Vague input 'something': no leaked <think> tag", "<think>" not in r_help.json()["reply"].lower(), r_help.text)

    # A genuinely ambiguous message that DOES reach the AI classifier +
    # worker - mocked at the ai_service boundary (not the agent-level
    # get_ai_response) so the actual <think>-stripping logic in
    # ai_service.get_groq_response runs for real, proving a raw leaking
    # model response never reaches the user-facing reply end to end.
    fake_completion = MagicMock()
    fake_completion.choices[0].message.content = "<think>weighing intent options</think>recommend"
    fake_worker_completion = MagicMock()
    fake_worker_completion.choices[0].message.content = (
        "<think>deciding which service fits best</think>Our AI automation service could help here. "
        "What's your current setup like?"
    )

    with patch("app.services.ai_service.GROQ_API_KEY", "dummy-key-for-test"), patch("groq.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [fake_completion, fake_worker_completion]
        mock_groq_cls.return_value = mock_client

        r_ambiguous = client.post(
            "/chat", json={"message": "not really sure what I need for my business", "session_id": session_id}
        )
    body_ambiguous = r_ambiguous.json()
    check("Vague input (AI-routed): HTTP 200", r_ambiguous.status_code == 200, r_ambiguous.text)
    check(
        "Vague input (AI-routed): reasoning trace stripped, no <think> tag reaches the reply",
        "<think>" not in body_ambiguous["reply"].lower() and "</think>" not in body_ambiguous["reply"].lower(),
        body_ambiguous["reply"],
    )
    check(
        "Vague input (AI-routed): the real answer text still made it through the strip",
        "automation" in body_ambiguous["reply"].lower(),
        body_ambiguous["reply"],
    )
    row = last_chat_history_row(session_id)
    check("Vague input (AI-routed): routed to recommend + logged", row is not None and row.intent == "recommend", str(row))

    reset_session(session_id)


def _test_think_leak_stripped_at_source():
    """Unit-level proof that ai_service itself strips a leaked reasoning
    block, independent of any agent - this is the actual fix, not just an
    end-to-end symptom check."""
    fake_completion = MagicMock()
    fake_completion.choices[0].message.content = "<think>internal deliberation, ignore this</think>The final answer."

    with patch("app.services.ai_service.GROQ_API_KEY", "dummy-key-for-test"), patch("groq.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_completion
        mock_groq_cls.return_value = mock_client
        reply = ai_service.get_groq_response("does this leak?")

    check("ai_service: <think> block stripped from the raw model response", "<think>" not in reply.lower(), reply)
    check("ai_service: the real answer text is preserved after stripping", reply.strip() == "The final answer.", reply)


# ---------------------------------------------------------------------------


def main():
    init_db()
    with TestClient(app) as client:
        _test_greeting(client)

        _test_quick_reply_hire_you(client)
        _test_quick_reply_services(client)
        _test_quick_reply_what_is_xqora(client)

        _test_faq_worker(client)
        _test_recommend_worker(client)

        _test_full_lead_form_flow(client)
        _test_lead_cancel_mid_form(client)

        _test_self_recall(client)

        _test_escalate_worker(client)
        _test_off_topic_refusal(client)

        _test_feedback_card_faces(client)
        _test_feedback_card_cancel(client)
        _test_feedback_text_fallback(client)
        _test_feedback_text_word_rating(client)

        _test_cross_session_isolation(client)
        _test_reload_persistence(client)

        _test_vague_input_no_think_leak(client)
        _test_think_leak_stripped_at_source()

    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"\n{passed} passed, {failed} failed out of {len(results)}")

    engine.dispose()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
