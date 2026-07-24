"""
End-to-end smoke test for /chat: one message per intent, plus an off-topic
request, run through the real orchestrator/agents (AI calls hit Groq for
real — no mocking). Confirms routing, Chat History logging, and that the
off-topic guardrail blocks before any specialist agent runs.

Run directly: python tests/test_e2e.py
(No pytest dependency in this project yet, so this is a plain script with
asserts and a PASS/FAIL summary, not a pytest suite.)
"""
import os
import sys
import time
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_e2e_test.db")
if os.path.exists(TEST_DB_PATH):
    os.remove(TEST_DB_PATH)
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"
# This script fires ~15 requests in a few seconds from one IP, which would
# otherwise trip the real 10/min rate limiter well before the scenarios finish.
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "1000")

from fastapi import BackgroundTasks  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.agents import faq_agent, feedback_agent, lead_agent, recommend_agent  # noqa: E402
from app.agents.intent_agent import classify_intent, hard_escalate_signal, _is_unrelated_professional_service  # noqa: E402
from app.database import ChatHistory, Feedback, Lead, SessionLocal, engine  # noqa: E402
from app.orchestrator import handle_message  # noqa: E402
from app.prompt import (  # noqa: E402
    BARE_CATEGORY_RESPONSE,
    CHECK_IN_RESPONSE,
    FEEDBACK_ASK,
    FEEDBACK_THANKS_RATING,
    SELF_IDENTITY_RESPONSE,
)

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


_all_response_bodies = []

# Paced deliberately: this suite now makes ~50+ real Groq calls (one or more
# per /chat turn), and firing them in a tight burst reliably trips Groq's
# free-tier per-minute rate limit mid-run even though Groq itself is fine
# outside the burst (verified independently) - that's an external throughput
# limit, not a code defect, but it was making this suite flaky, so a small
# delay between requests keeps the whole run under the cap.
_REQUEST_PACING_SECONDS = 1.0


def _post(client, payload):
    time.sleep(_REQUEST_PACING_SECONDS)
    r = client.post("/chat", json=payload)
    _all_response_bodies.append(r.text)
    return r


def main():
    with TestClient(app) as client:
        # 1. FAQ intent
        r = _post(client, {"message": "What services does XQORA provide?"})
        check("FAQ: HTTP 200", r.status_code == 200, r.text)
        body = r.json()
        row = last_chat_history_row(body["session_id"])
        check("FAQ: routed to faq intent + logged", row is not None and row.intent == "faq", str(row))
        check("FAQ: reply mentions a real service", "development" in body["reply"].lower() or "service" in body["reply"].lower())

        # 2. Recommend intent
        r = _post(client, {"message": "I need a chatbot built for my business, what do you suggest?"})
        check("Recommend: HTTP 200", r.status_code == 200, r.text)
        body = r.json()
        row = last_chat_history_row(body["session_id"])
        check("Recommend: routed to recommend intent + logged", row is not None and row.intent == "recommend", str(row))
        recommend_reply_lower = body["reply"].lower()
        check(
            "Recommend: no pricing/tier talk",
            not any(w in recommend_reply_lower for w in ["tier", "pricing", "custom quote", "$", "starter", "enterprise"]),
            body["reply"],
        )
        check("Recommend: ends with a follow-up question", body["reply"].strip().endswith("?"), body["reply"])

        # 3. Escalation intent
        r = _post(client, {"message": "This is unacceptable, I need to speak to a real human right now"})
        check("Escalate: HTTP 200", r.status_code == 200, r.text)
        body = r.json()
        row = last_chat_history_row(body["session_id"])
        check("Escalate: routed to escalate intent + logged", row is not None and row.intent == "escalate", str(row))
        check(
            "Escalate: reply includes contact email only (phone removed per content change)",
            "@" in body["reply"] and "+91" not in body["reply"] and "Phone:" not in body["reply"],
            body["reply"],
        )

        # 4. Off-topic: must be refused before reaching any specialist agent
        r = _post(client, {"message": "Write me a python script to sort a list"})
        check("Off-topic: HTTP 200", r.status_code == 200, r.text)
        body = r.json()
        check(
            "Off-topic: soft refusal returned (mentions team contact, not the old flat line)",
            "xqoratechnologies@gmail.com" in body["reply"] and "only help with questions about" not in body["reply"].lower(),
            body["reply"],
        )
        row = last_chat_history_row(body["session_id"])
        check("Off-topic: NOT logged to Chat History (blocked pre-agent)", row is None, str(row))

        # 4b. Plain greeting: must NOT hit the off-topic refusal
        r = _post(client, {"message": "hi"})
        check("Greeting: HTTP 200", r.status_code == 200, r.text)
        body = r.json()
        check(
            "Greeting: friendly response, not the off-topic refusal",
            "outside what i can help" not in body["reply"].lower() and len(body["reply"]) > 0,
            body["reply"],
        )

        # 4b-2. Typo'd/fuzzy greeting variants: must resolve CONSISTENTLY as
        # greeting every time (deterministic code layer, not AI judgment call
        # variance). Tested directly against classify_intent (not /chat) so
        # this is fast and isolates the actual fuzzy-match layer from the AI
        # classification path entirely - these should never even reach the AI.
        from app.agents.intent_agent import classify_intent as _classify_intent

        fuzzy_greetings = ["hie", "hwello", "helo", "heyyy", "hii", "heyo", "sup", "yo"]
        for variant in fuzzy_greetings:
            outcomes = [_classify_intent(variant) for _ in range(5)]
            all_greeting = all(o.intent == "greeting" and o.blocked for o in outcomes)
            check(
                f"Fuzzy greeting '{variant}': consistent across 5 calls",
                all_greeting,
                [o.intent for o in outcomes],
            )

        # Must NOT be swallowed as greetings: short real words and a longer
        # message with actual content (even one that happens to start "hey").
        non_greetings = ["faq", "cost"]
        for phrase in non_greetings:
            outcome = _classify_intent(phrase)
            check(
                f"Not a greeting: '{phrase}' classified as a real intent",
                not (outcome.intent == "greeting" and outcome.blocked),
                outcome.intent,
            )

        r = _post(client, {"message": "hey can you build me an app"})
        check("Not a greeting: longer real request gets HTTP 200", r.status_code == 200, r.text)
        longer_request_body = r.json()
        longer_request_row = last_chat_history_row(longer_request_body["session_id"])
        # Greeting/off_topic short-circuits skip Chat History logging entirely
        # (see orchestrator.py) - a logged row here proves this got a real
        # intent (faq/recommend/lead/escalate), not just the canned greeting.
        check(
            "Not a greeting: longer real request routed to a real intent, not swallowed as a greeting",
            longer_request_row is not None and longer_request_row.intent in ("faq", "recommend", "lead", "escalate"),
            f"reply={longer_request_body['reply']!r} row={longer_request_row}",
        )

        # 4c. Near-empty input ("?" passes pydantic's min_length but is
        # semantically empty): must NOT hit the off-topic refusal either
        r = _post(client, {"message": "?"})
        check("Near-empty: HTTP 200", r.status_code == 200, r.text)
        body = r.json()
        check(
            "Near-empty: warm invite to share more, not the off-topic refusal",
            "outside what i can help" not in body["reply"].lower() and len(body["reply"]) > 0,
            body["reply"],
        )

        # 4d. Formal-sounding but genuinely unrelated professional service ask
        # (a legal/compliance audit isn't anything XQORA's service catalog
        # covers): must be refused like any other off_topic request, NOT
        # escalated - escalating would wrongly imply the team can help.
        r = _post(client, {"message": "Can you help with a legal compliance audit for my fintech app?"})
        check("Unrelated professional service query: HTTP 200", r.status_code == 200, r.text)
        body = r.json()
        check(
            "Unrelated professional service query: refused like off_topic, not escalated",
            "xqoratechnologies@gmail.com" in body["reply"] and "this one's best handled" not in body["reply"].lower(),
            body["reply"],
        )
        row = last_chat_history_row(body["session_id"])
        check(
            "Unrelated professional service query: NOT logged as escalate (blocked pre-agent like off_topic)",
            row is None,
            str(row),
        )

        # 4e. Same "tender" keyword, but tied to one of XQORA's actual
        # services this time: must still escalate deterministically
        # (hard_escalate_signal), not rely on AI judgment call variance that
        # previously sent this to faq with an unrelated answer.
        r = _post(client, {"message": "can you handle a large government tender project"})
        check("Government tender query: HTTP 200", r.status_code == 200, r.text)
        tender_row = last_chat_history_row(r.json()["session_id"])
        check(
            "Government tender query: routed to escalate deterministically",
            tender_row is not None and tender_row.intent == "escalate",
            str(tender_row),
        )
        check(
            "hard_escalate_signal: fires for a service-relevant tender, NOT for the unrelated legal audit or "
            "the legit GDPR FAQ question",
            not hard_escalate_signal("can you help with a legal compliance audit for my fintech app")
            and hard_escalate_signal("can you handle a large government tender project")
            and not hard_escalate_signal(
                "How do XQORA's custom AI solutions comply with privacy regulations like GDPR or Indian DPDP?"
            ),
        )
        check(
            "_is_unrelated_professional_service: catches the legal audit ask, not the tender or GDPR FAQ question",
            _is_unrelated_professional_service("can you help with a legal compliance audit for my fintech app")
            and not _is_unrelated_professional_service("can you handle a large government tender project")
            and not _is_unrelated_professional_service(
                "How do XQORA's custom AI solutions comply with privacy regulations like GDPR or Indian DPDP?"
            ),
        )

        # 5. Full lead capture flow across turns: name, company, email, phone,
        # message only (no budget/service questions), with an invalid email
        # and an invalid phone thrown in to confirm re-prompting works.
        session_id = None
        lead_turns = [
            "I want to start a project and get a quote",
            "Jane Smith",
            "Acme Corp",
            "not-an-email",  # invalid, should be rejected and re-asked
            "jane@acme.com",
            "12345",  # invalid, should be rejected and re-asked
            "+1 555 123 4567",
            "Need it for iOS and Android by Q4",
        ]
        replies = []
        for turn in lead_turns:
            payload = {"message": turn}
            if session_id:
                payload["session_id"] = session_id
            r = _post(client, payload)
            check(f"Lead turn HTTP 200: '{turn[:30]}'", r.status_code == 200, r.text)
            if session_id is None:
                session_id = r.json()["session_id"]
            replies.append(r.json()["reply"])

        check("Lead: no budget question anywhere in the flow", not any("budget" in r.lower() for r in replies), replies)
        check("Lead: no service-fit question anywhere in the flow", not any("interested in" in r.lower() for r in replies), replies)
        check("Lead: invalid email rejected with re-prompt", "valid email" in replies[3].lower(), replies[3])
        check("Lead: invalid phone rejected with re-prompt", "valid phone" in replies[5].lower(), replies[5])

        db = SessionLocal()
        try:
            lead = db.query(Lead).filter(Lead.session_id == session_id).first()
        finally:
            db.close()
        check("Lead: saved to Leads table", lead is not None)
        if lead:
            check("Lead: name captured", lead.name == "Jane Smith", lead.name)
            check("Lead: company captured", lead.company == "Acme Corp", lead.company)
            check("Lead: email captured (invalid attempt discarded)", lead.email == "jane@acme.com", lead.email)
            check("Lead: phone captured (invalid attempt discarded)", lead.phone == "+15551234567", lead.phone)
            check("Lead: message captured", "ios" in (lead.message or "").lower(), lead.message)
            check("Lead: budget left unset (no longer asked)", not lead.budget, lead.budget)
            check("Lead: service left unset (no longer asked)", not lead.service, lead.service)

        # 8. Contextual follow-up reply: the bot's own recommend-agent
        # question is answered with a short casual reply that has no keyword
        # overlap with any service on its own - must be understood as
        # continuing the conversation (via recent chat history passed into
        # classify_intent), not refused as off-topic.
        r = _post(client, {"message": "automate customer support"})
        check("Context follow-up setup: HTTP 200", r.status_code == 200, r.text)
        setup_body = r.json()
        followup_session = setup_body["session_id"]
        check("Context follow-up setup: ends with a question", setup_body["reply"].strip().endswith("?"), setup_body["reply"])

        r = _post(client, {"message": "yeah we get flooded with tickets every day", "session_id": followup_session})
        check("Context follow-up: HTTP 200", r.status_code == 200, r.text)
        followup_body = r.json()
        row = last_chat_history_row(followup_session)
        check(
            "Context follow-up: NOT refused as off-topic, logged with a real intent",
            row is not None and row.intent in ("faq", "recommend", "lead", "escalate"),
            f"reply={followup_body['reply']!r} row={row}",
        )
        check(
            "Context follow-up: reply isn't the off-topic/guardrail refusal",
            "outside what i can help" not in followup_body["reply"].lower(),
            followup_body["reply"],
        )

        # 9. Lead flow off-flow aside: mid-collection (at the "name" step), an
        # unrelated complex question must NOT be swallowed as the name - it
        # should be handled on its own, with the lead flow paused and then
        # resumable with the real name afterward.
        r = _post(client, {"message": "I want to start a project and get a quote"})
        check("Off-flow setup: HTTP 200", r.status_code == 200, r.text)
        offflow_session = r.json()["session_id"]

        r = _post(client, {
            "message": "Can you help with a legal compliance audit for my fintech app",
            "session_id": offflow_session,
        })
        check("Off-flow aside: HTTP 200", r.status_code == 200, r.text)
        aside_reply = r.json()["reply"]
        check("Off-flow aside: nudges back toward the paused name field", "name" in aside_reply.lower(), aside_reply)

        db = SessionLocal()
        try:
            no_lead_yet = db.query(Lead).filter(Lead.session_id == offflow_session).first()
        finally:
            db.close()
        check("Off-flow aside: no Lead row created yet (flow still paused, not corrupted)", no_lead_yet is None, no_lead_yet)

        r = _post(client, {"message": "Priya Shah", "session_id": offflow_session})
        check("Off-flow resume: name accepted after the aside", r.status_code == 200, r.text)
        check("Off-flow resume: moved on to company", "company" in r.json()["reply"].lower(), r.json()["reply"])

        for turn in ["Acme Fintech", "priya@acme.com", "+91 9876543210", "Need help building a payments app"]:
            r = _post(client, {"message": turn, "session_id": offflow_session})
            check(f"Off-flow resume turn HTTP 200: '{turn[:30]}'", r.status_code == 200, r.text)

        db = SessionLocal()
        try:
            offflow_lead = db.query(Lead).filter(Lead.session_id == offflow_session).first()
        finally:
            db.close()
        check("Off-flow resume: lead saved", offflow_lead is not None)
        if offflow_lead:
            check(
                "Off-flow resume: name is the real name, NOT the off-topic sentence",
                offflow_lead.name == "Priya Shah",
                offflow_lead.name,
            )
            check("Off-flow resume: company captured", offflow_lead.company == "Acme Fintech", offflow_lead.company)

        # 9b. The EXACT phrases from this bug report, seeded directly into
        # lead_agent's session state (bypassing classify_intent) so this
        # test is deterministic and doesn't depend on the AI having first
        # routed the opening message to "lead" - isolates the actual fix
        # (field-plausibility detection) from that separate routing step.
        bug_session_a = "regression-need-somewhere-to-host"
        state_a = lead_agent._get_session(bug_session_a)
        state_a["name"] = "Alex Kumar"
        state_a["_awaiting"] = "company"
        r = _post(client, {"message": "need somewhere to host my app", "session_id": bug_session_a})
        check("Bug report phrase A: HTTP 200", r.status_code == 200, r.text)
        check(
            "Bug report phrase A: NOT silently accepted as the company name",
            lead_agent._sessions[bug_session_a]["company"] is None
            and lead_agent._sessions[bug_session_a]["_awaiting"] == "company",
            lead_agent._sessions[bug_session_a],
        )

        bug_session_b = "regression-want-an-app"
        state_b = lead_agent._get_session(bug_session_b)
        state_b["name"] = "Alex Kumar"
        state_b["company"] = "Acme Corp"
        state_b["_awaiting"] = "email"
        r = _post(client, {"message": "want an app for my restaurant", "session_id": bug_session_b})
        check("Bug report phrase B: HTTP 200", r.status_code == 200, r.text)
        check(
            "Bug report phrase B: NOT silently accepted as the email",
            lead_agent._sessions[bug_session_b]["email"] is None
            and lead_agent._sessions[bug_session_b]["_awaiting"] == "email",
            lead_agent._sessions[bug_session_b],
        )
        lead_agent.reset_session(bug_session_a)
        lead_agent.reset_session(bug_session_b)

        # 9c. Explicit bail-out: "never mind" mid-flow should cancel and
        # reset, not be swallowed as a field value.
        cancel_session = "regression-cancel"
        state_c = lead_agent._get_session(cancel_session)
        state_c["name"] = "Someone"
        state_c["_awaiting"] = "company"
        r = _post(client, {"message": "never mind", "session_id": cancel_session})
        check("Cancel mid-flow: HTTP 200", r.status_code == 200, r.text)
        check("Cancel mid-flow: acknowledges the cancellation", "cancel" in r.json()["reply"].lower(), r.json()["reply"])
        check("Cancel mid-flow: session fully reset", not lead_agent.is_collecting(cancel_session))

        # 9d. faq_agent matching, unit-level: the exact wrong-match reported
        # ("automate customer support" -> post-project-support FAQ) must not
        # recur, and the correct AI-automation FAQ should rank first.
        faq_matches = faq_agent.find_best_faqs("automate customer support")
        check("faq_agent: 'automate customer support' gets at least one match", len(faq_matches) > 0, faq_matches)
        if faq_matches:
            check(
                "faq_agent: top match is the AI automation FAQ, not the unrelated post-project-support one",
                "automation" in faq_matches[0]["question"].lower(),
                faq_matches[0]["question"],
            )
        check(
            "faq_agent: 'government tender project' does not confidently match an unrelated FAQ",
            faq_agent.find_best_faqs("government tender project") == [],
        )

        # 10. recommend_agent/faq_agent context-threading (unit-level, not
        # HTTP): a fragment with zero keyword overlap on its own must still
        # produce a real contextual answer, not the generic fallback, once
        # recent_context supplies the topic. Tested directly against the
        # agent functions (not through /chat) since which of the 4 intents
        # an ambiguous fragment lands on is an AI judgment call - this
        # isolates the actual fix (context-aware generation) from that
        # separate, expected variability.
        no_context_matches, _ = recommend_agent.find_best_services("yeah a lot of requests")
        check("recommend_agent: fragment alone has no keyword match (test validity check)", no_context_matches == [])

        recommend_reply = recommend_agent.get_recommendation_response(
            "yeah a lot of requests",
            recent_context=(
                "User: automate customer support\n"
                "Bot: You should look into our AI Automation & Chatbots service, it can "
                "help automate replies and free up your team. What's your current "
                "support volume like?"
            ),
        )
        check(
            "recommend_agent: context-threaded fragment avoids the generic fallback",
            recommend_reply != recommend_agent._FALLBACK_MESSAGE,
            recommend_reply,
        )
        check(
            "recommend_agent: context-threaded reply still has no pricing/tier talk",
            not any(w in recommend_reply.lower() for w in ["tier", "pricing", "custom quote", "$", "starter", "enterprise"]),
            recommend_reply,
        )

        no_context_faq_matches = faq_agent.find_best_faqs("yeah for sure")
        check("faq_agent: fragment alone has no keyword match (test validity check)", no_context_faq_matches == [])

        faq_reply = faq_agent.get_faq_response(
            "yeah for sure",
            recent_context=(
                "User: Is the XQORA internship paid?\n"
                "Bot: It depends on the role and how you perform during the process."
            ),
        )
        check(
            "faq_agent: context-threaded fragment avoids the generic fallback",
            faq_reply != faq_agent._FALLBACK_MESSAGE,
            faq_reply,
        )

        # 11. "hi" typed at the lead-collection "name" step: must be rejected
        # as off-flow (too obviously a greeting to be a real name), not
        # silently accepted as the literal name value.
        r = _post(client, {"message": "I want to start a project and get a quote"})
        check("Greeting-at-name setup: HTTP 200", r.status_code == 200, r.text)
        greeting_name_session = r.json()["session_id"]

        r = _post(client, {"message": "hi", "session_id": greeting_name_session})
        check("Greeting-at-name: HTTP 200", r.status_code == 200, r.text)
        greeting_name_reply = r.json()["reply"]
        check(
            "Greeting-at-name: rejected as off-flow, nudges back to the name field",
            "name" in greeting_name_reply.lower(),
            greeting_name_reply,
        )

        r = _post(client, {"message": "Alex Kumar", "session_id": greeting_name_session})
        check("Greeting-at-name: real name accepted afterward", "company" in r.json()["reply"].lower(), r.json()["reply"])

        db = SessionLocal()
        try:
            no_lead_from_greeting = db.query(Lead).filter(Lead.session_id == greeting_name_session).first()
        finally:
            db.close()
        check(
            "Greeting-at-name: no Lead row yet (still mid-flow, not corrupted by 'hi')",
            no_lead_from_greeting is None,
            no_lead_from_greeting,
        )

        # 6. Input validation
        r = _post(client, {"message": "   "})
        check("Validation: blank message rejected (422)", r.status_code == 422, r.text)
        r = _post(client, {"message": "x" * 5000})
        check("Validation: oversized message rejected (422)", r.status_code == 422, r.text)

        # 7. Error-safety: response bodies never contain API keys/tracebacks
        leak_markers = ["Traceback", "api_key", "gsk_", "AIza", "re_test", ".env"]
        combined = "\n".join(_all_response_bodies)
        found = [m for m in leak_markers if m in combined]
        check("Security: no leak markers in any response body", not found, str(found))

    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"\n{passed} passed, {failed} failed out of {len(results)}")

    engine.dispose()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

    sys.exit(1 if failed else 0)


# ---------------------------------------------------------------------------
# Mocked regression tests for a later round of bug reports (see commit/PR
# description). Unlike main() above, these mock app.*.get_ai_response instead
# of hitting Groq for real - they isolate the actual code-level fixes
# (deterministic overrides, gating, session-state handling) from AI judgment
# call variance, and don't burn real API quota/rate limit budget on every
# run. Invoke with `python tests/test_e2e.py --mocked` (main()'s live suite
# stays the default for plain `python tests/test_e2e.py`, matching the
# README).
# ---------------------------------------------------------------------------


def _bug_escalate_vs_unrelated_professional_service():
    audit_ask = "Can you help with a legal compliance audit for my fintech app?"
    tender_ask = "can you handle a large government tender project"
    tax_ask = "can you help me file my taxes"
    gdpr_faq = "How do XQORA's custom AI solutions comply with privacy regulations like GDPR or Indian DPDP?"

    check(
        "Escalate fix (mocked): legal compliance audit is NOT a hard escalate signal",
        not hard_escalate_signal(audit_ask),
        None,
    )
    check(
        "Escalate fix (mocked): legal compliance audit IS an unrelated-professional-service signal",
        _is_unrelated_professional_service(audit_ask),
        None,
    )
    check(
        "Escalate fix (mocked): service-relevant government tender still hard-escalates",
        hard_escalate_signal(tender_ask),
        None,
    )
    check(
        "Escalate fix (mocked): bare tender with no XQORA service tie does NOT hard-escalate",
        not hard_escalate_signal("we submitted a tender for the office catering contract"),
        None,
    )
    check(
        "Escalate fix (mocked): tax filing is an unrelated-professional-service signal",
        _is_unrelated_professional_service(tax_ask),
        None,
    )
    check(
        "Escalate fix (mocked): GDPR/DPDP compliance FAQ trips neither signal",
        not hard_escalate_signal(gdpr_faq) and not _is_unrelated_professional_service(gdpr_faq),
        None,
    )

    r = classify_intent(audit_ask)
    check(
        "Escalate fix (mocked): full classify_intent refuses the legal audit ask as off_topic, not escalate",
        r.intent == "off_topic" and r.blocked,
        r,
    )
    r2 = classify_intent(tender_ask)
    check(
        "Escalate fix (mocked): full classify_intent still escalates the service-relevant tender",
        r2.intent == "escalate" and not r2.blocked,
        r2,
    )
    r3 = classify_intent(tax_ask)
    check(
        "Escalate fix (mocked): full classify_intent refuses tax filing as off_topic",
        r3.intent == "off_topic" and r3.blocked,
        r3,
    )

    # A genuinely complex-but-relevant ask has no hard-coded pattern and
    # relies on the AI's own judgment - confirm the new off-topic gate
    # doesn't interfere with a legitimate escalate verdict here.
    with patch("app.agents.intent_agent.get_ai_response", return_value="escalate"):
        r4 = classify_intent("can you build a large-scale enterprise CRM system")
    check(
        "Escalate fix (mocked): AI-judged escalate for a complex-but-relevant ask passes through untouched",
        r4.intent == "escalate" and not r4.blocked,
        r4,
    )


def _bug_check_in_and_trivia_misclassification():
    check_in_phrases = ["hey are you still there", "you still there?", "anyone there", "anybody there?", "are you there"]
    for msg in check_in_phrases:
        r = classify_intent(msg)
        check(
            f"Check-in fix (mocked): '{msg}' gets a casual acknowledgment, not the FAQ fallback/handoff",
            r.blocked and r.refusal_message == CHECK_IN_RESPONSE,
            r,
        )

    # "hello?" alone stays a plain greeting (unaffected, not double-handled
    # by the new check-in path).
    r_hello = classify_intent("hello?")
    check(
        "Check-in fix (mocked): 'hello?' alone is still handled as a plain greeting, not the check-in path",
        r_hello.blocked and r_hello.refusal_message != CHECK_IN_RESPONSE,
        r_hello,
    )

    # A real question that happens to start similarly must NOT be swallowed
    # as a check-in.
    with patch("app.agents.intent_agent.get_ai_response", return_value="faq"):
        r_real = classify_intent("are you there to help with pricing questions?")
    check(
        "Check-in fix (mocked): a real question isn't misclassified as a bare check-in",
        r_real.refusal_message != CHECK_IN_RESPONSE,
        r_real,
    )

    trivia_phrases = [
        "what time is it",
        "what's the time",
        "what's the weather today",
        "who is the prime minister",
    ]
    for msg in trivia_phrases:
        r = classify_intent(msg)
        check(
            f"Trivia fix (mocked): '{msg}' gets the plain off_topic refusal, not an escalate/team-handoff framing",
            r.intent == "off_topic" and r.blocked and "best handled by our team" not in r.refusal_message.lower(),
            r,
        )

    # These trivia phrases must never even reach faq_agent's matching (which
    # is where they'd otherwise pick up a false-positive FAQ match - see the
    # flagged _tokens_match finding) since hard_topic_filter blocks them
    # before any specialist agent runs.
    check(
        "Trivia fix (mocked): 'who is the prime minister' never reaches faq_agent (blocked pre-agent)",
        classify_intent("who is the prime minister").blocked,
        None,
    )


def _bug1_context_off_topic_override():
    recent_context = (
        "User: automate customer support\n"
        "Bot: You should look into our AI Automation and Chatbots service, it can help "
        "automate replies and free up your team. What's your current support volume like?"
    )
    # First call: off_topic (the observed non-determinism). Retry call: the
    # AI actually reads it as a continuation of the conversation.
    with patch("app.agents.intent_agent.get_ai_response", side_effect=["off_topic", "recommend"]):
        r = classify_intent("yeah a lot of requests", recent_context=recent_context, previous_intent="recommend")
        check(
            "Bug1 (mocked): flipped-on-retry off_topic verdict is trusted, not refused",
            r.intent == "recommend" and not r.blocked,
            r,
        )

    with patch("app.agents.intent_agent.get_ai_response", return_value="off_topic"):
        r2 = classify_intent("yeah a lot of requests", recent_context=None, previous_intent=None)
    check(
        "Bug1 (mocked): off_topic verdict stands with no context/previous_intent to fall back on",
        r2.intent == "off_topic" and r2.blocked,
        r2,
    )

    # Regression for the gibberish-after-escalate bug: if BOTH the initial
    # call and the retry agree on off_topic, it must stay off_topic - even
    # though previous_intent is a real intent (escalate) - not get waved
    # through into whatever the previous turn happened to be. Previously this
    # looped pure gibberish ("hwellll", "fhdshttt") straight back into
    # escalate whenever it followed an escalate turn, wasting the team's time
    # on non-genuine noise.
    with patch("app.agents.intent_agent.get_ai_response", return_value="off_topic"):
        for gibberish in ("hwellll", "fhdshttt"):
            r3 = classify_intent(gibberish, recent_context=recent_context, previous_intent="escalate")
            check(
                f"Bug1 (mocked): gibberish '{gibberish}' after an escalate turn stays off_topic, not escalate",
                r3.intent == "off_topic" and r3.blocked,
                r3,
            )


def _bug2_lead_requires_explicit_signal():
    with patch("app.agents.intent_agent.get_ai_response", return_value="lead"):
        r = classify_intent(
            "1000 users",
            recent_context="User: what does your app look like?\nBot: what's your app size and expected traffic?",
            previous_intent="recommend",
        )
        check(
            "Bug2 (mocked): factual reply not blindly accepted as lead, falls back to previous_intent",
            r.intent == "recommend" and not r.blocked,
            r,
        )

        r2 = classify_intent("1000 users", recent_context=None, previous_intent=None)
        check(
            "Bug2 (mocked): no context/previous_intent falls back to recommend default, not lead",
            r2.intent == "recommend" and not r2.blocked,
            r2,
        )

        r3 = classify_intent("I want to hire you for this project", recent_context=None, previous_intent=None)
        check(
            "Bug2 (mocked): an explicit lead phrase is still accepted as lead",
            r3.intent == "lead" and not r3.blocked,
            r3,
        )


def _bug3_bare_category_words():
    for word in ["faq", "faqs", "pricing", "price", "cost", "quote"]:
        r = classify_intent(word)
        check(
            f"Bug3 (mocked): bare '{word}' gets a clarifying question, not an FAQ lookup",
            r.blocked and r.refusal_message == BARE_CATEGORY_RESPONSE,
            r,
        )

    with patch("app.agents.intent_agent.get_ai_response", return_value="faq"):
        r = classify_intent("What services does XQORA provide?")
        check(
            "Bug3 (mocked): a real sentence containing a category word is NOT short-circuited",
            r.intent == "faq" and not r.blocked,
            r,
        )


def _bug4_no_nudge_after_off_topic_refusal():
    session_id = "regression-bug4-nudge-offtopic"
    lead_agent.reset_session(session_id)
    state = lead_agent._get_session(session_id)
    state["_awaiting"] = "name"

    db = SessionLocal()
    try:
        with patch("app.agents.lead_agent.get_ai_response", return_value="NO"), patch(
            "app.agents.intent_agent.get_ai_response", return_value="off_topic"
        ):
            reply = handle_message(db, session_id, "asdkjqwer zxcvasdf nonsense blah", BackgroundTasks())
    finally:
        db.close()

    check(
        "Bug4 (mocked): flat off-topic refusal has no resume-nudge appended",
        "picking back up" not in reply.lower() and "your name" not in reply.lower(),
        reply,
    )
    check(
        "Bug4 (mocked): lead flow still paused after the refusal (not corrupted)",
        lead_agent.is_collecting(session_id),
        lead_agent._sessions.get(session_id),
    )
    lead_agent.reset_session(session_id)

    # Contrast case: a genuine alternate-topic answer (escalate, via the
    # deterministic hard_escalate_signal so no AI mock is needed for
    # classification itself) SHOULD still get the resume nudge appended.
    session_id2 = "regression-bug4-nudge-escalate"
    lead_agent.reset_session(session_id2)
    state2 = lead_agent._get_session(session_id2)
    state2["_awaiting"] = "name"

    db = SessionLocal()
    try:
        with patch("app.agents.lead_agent.get_ai_response", return_value="NO"):
            reply2 = handle_message(db, session_id2, "can you handle a large government tender project", BackgroundTasks())
    finally:
        db.close()

    check(
        "Bug4 (mocked): genuine alternate-topic handling (escalate) still gets the resume nudge",
        "name" in reply2.lower(),
        reply2,
    )
    lead_agent.reset_session(session_id2)


def _bug5_self_identity():
    identity_questions = [
        "who are you",
        "Who are you?",
        "are you a bot",
        "What is XQORA Assistant?",
        "what do you do",
        "tell me about yourself",
    ]
    for msg in identity_questions:
        r = classify_intent(msg)
        check(
            f"Bug5 (mocked): self-identity '{msg}' answered directly, no FAQ lookup/fallback",
            r.blocked and r.refusal_message == SELF_IDENTITY_RESPONSE,
            r,
        )

    with patch("app.agents.intent_agent.get_ai_response", return_value="faq"):
        r = classify_intent("What services does XQORA provide?")
        check(
            "Bug5 (mocked): a real FAQ question about services is not swallowed as self-identity",
            r.refusal_message != SELF_IDENTITY_RESPONSE,
            r,
        )


def _feature_reuse_lead_details():
    session_id = "regression-reuse-lead"
    lead_agent.reset_session(session_id)
    lead_agent._completed_leads.pop(session_id, None)
    lead_agent._store_completed_lead(
        session_id,
        {"name": "Jane Smith", "company": "Acme Corp", "email": "jane@acme.com", "phone": "+15551234567"},
    )

    db = SessionLocal()
    try:
        bg = BackgroundTasks()
        r1 = lead_agent.handle_lead_message(db, session_id, "I want to get a quote", bg)
        check(
            "Feature (mocked): reuse-or-new prompt offered on a second lead trigger",
            r1.reply == lead_agent.REUSE_DETAILS_PROMPT,
            r1,
        )

        r2 = lead_agent.handle_lead_message(db, session_id, "yes, use the same details", bg)
        state = lead_agent._sessions[session_id]
        check(
            "Feature (mocked): reuse fills in the prior contact fields",
            state["name"] == "Jane Smith" and state["email"] == "jane@acme.com" and state["phone"] == "+15551234567",
            state,
        )
        check(
            "Feature (mocked): reuse jumps straight to the message field, not name/company/email/phone again",
            "help" in r2.reply.lower(),
            r2.reply,
        )

        lead_agent.handle_lead_message(db, session_id, "Need a new landing page built", bg)
        lead = db.query(Lead).filter(Lead.session_id == session_id).order_by(Lead.id.desc()).first()
        check(
            "Feature (mocked): lead finalized with reused contact info + the new inquiry message",
            lead is not None and lead.email == "jane@acme.com" and "landing" in (lead.message or "").lower(),
            lead,
        )
    finally:
        db.close()
        lead_agent.reset_session(session_id)
        lead_agent._completed_leads.pop(session_id, None)

    # Contrast case: declining reuse falls back to the normal from-scratch flow.
    session_id2 = "regression-reuse-lead-decline"
    lead_agent.reset_session(session_id2)
    lead_agent._completed_leads.pop(session_id2, None)
    lead_agent._store_completed_lead(
        session_id2,
        {"name": "Old Name", "company": "Old Co", "email": "old@x.com", "phone": "+15550000000"},
    )

    db = SessionLocal()
    try:
        bg = BackgroundTasks()
        r1 = lead_agent.handle_lead_message(db, session_id2, "I want to get a quote", bg)
        check(
            "Feature (mocked): reuse-or-new prompt offered (decline case)",
            r1.reply == lead_agent.REUSE_DETAILS_PROMPT,
            r1,
        )
        r2 = lead_agent.handle_lead_message(db, session_id2, "no, I'll provide new details", bg)
        check(
            "Feature (mocked): declining reuse asks for name from scratch",
            "name" in r2.reply.lower(),
            r2.reply,
        )
        state2 = lead_agent._sessions[session_id2]
        check(
            "Feature (mocked): declined reuse leaves all fields unset (no stale data carried over)",
            state2["name"] is None and state2["email"] is None,
            state2,
        )
    finally:
        db.close()
        lead_agent.reset_session(session_id2)
        lead_agent._completed_leads.pop(session_id2, None)


def _feedback_response_parsing_unit():
    check("Feedback (mocked): is_closing_message matches 'thanks'", feedback_agent.is_closing_message("thanks"), None)
    check(
        "Feedback (mocked): is_closing_message matches 'that's all, bye!'",
        feedback_agent.is_closing_message("that's all, bye!"),
        None,
    )
    check(
        "Feedback (mocked): is_closing_message does NOT match a real sentence that merely contains 'thanks'",
        not feedback_agent.is_closing_message("thanks for the info, what's the pricing for cloud hosting"),
        None,
    )

    rating, remainder = feedback_agent._extract_rating("4 - really helpful, thanks")
    check(
        "Feedback (mocked): numeric rating extracted with trailing comment kept",
        rating == 4 and "helpful" in remainder,
        (rating, remainder),
    )
    no_rating, remainder2 = feedback_agent._extract_rating("it was great")
    check("Feedback (mocked): non-numeric text has no numeric rating", no_rating is None, (no_rating, remainder2))

    check(
        "Feedback (mocked): word-rating extraction picks up 'great'",
        feedback_agent._extract_word_rating("this was great, thanks") == 4,
        None,
    )
    check(
        "Feedback (mocked): word-rating extraction returns None for neutral text",
        feedback_agent._extract_word_rating("here is the info you needed") is None,
        None,
    )


def _feedback_after_lead_submission():
    session_id = "regression-feedback-after-lead"
    lead_agent.reset_session(session_id)
    lead_agent._completed_leads.pop(session_id, None)
    feedback_agent.reset(session_id)

    # Seed the session one field away from finalization - the "message" field
    # is free-form (no AI field-matching call), so finishing the lead here is
    # fully deterministic and needs no mocking.
    state = lead_agent._get_session(session_id)
    state["name"] = "Sam Lee"
    state["company"] = "Sam Co"
    state["email"] = "sam@samco.com"
    state["phone"] = "+15551234567"
    state["_awaiting"] = "message"

    db = SessionLocal()
    try:
        reply = handle_message(db, session_id, "Need a website redesign", BackgroundTasks())
    finally:
        db.close()

    check(
        "Feedback (mocked): lead finalization reply includes the feedback ask",
        "Thanks, got it" in reply and FEEDBACK_ASK in reply,
        reply,
    )
    check(
        "Feedback (mocked): session now awaiting a feedback response after lead submission",
        feedback_agent.is_awaiting_response(session_id),
        None,
    )
    lead_agent.reset_session(session_id)
    lead_agent._completed_leads.pop(session_id, None)
    return session_id


def _feedback_rating_and_comment_saved(session_id):
    db = SessionLocal()
    try:
        reply = handle_message(db, session_id, "5 - loved how fast this was!", BackgroundTasks())
        fb = db.query(Feedback).filter(Feedback.session_id == session_id).order_by(Feedback.id.desc()).first()
    finally:
        db.close()

    check(
        "Feedback (mocked): rating+comment reply is a plain thank-you, not misrouted elsewhere",
        reply == FEEDBACK_THANKS_RATING,
        reply,
    )
    check(
        "Feedback (mocked): rating+comment saved to the Feedback table",
        fb is not None and fb.rating == 5 and "fast" in (fb.comments or "").lower(),
        fb,
    )
    check(
        "Feedback (mocked): session no longer awaiting feedback once given",
        not feedback_agent.is_awaiting_response(session_id),
        None,
    )
    feedback_agent.reset(session_id)


def _feedback_after_escalation():
    session_id = "regression-feedback-after-escalate"
    feedback_agent.reset(session_id)
    lead_agent.reset_session(session_id)

    # hard_escalate_signal is a pure-code deterministic gate (no AI call
    # needed for classification itself), and get_escalation_response() is
    # canned text - this whole scenario needs no mocking.
    db = SessionLocal()
    try:
        reply = handle_message(db, session_id, "can you handle a large government tender project", BackgroundTasks())
    finally:
        db.close()

    check("Feedback (mocked): escalation reply includes the feedback ask", FEEDBACK_ASK in reply, reply)
    check(
        "Feedback (mocked): session awaiting feedback after an escalation",
        feedback_agent.is_awaiting_response(session_id),
        None,
    )
    feedback_agent.reset(session_id)


def _feedback_ignored_does_not_loop_or_misfire():
    session_id = "regression-feedback-ignored"
    feedback_agent.reset(session_id)
    lead_agent.reset_session(session_id)

    db = SessionLocal()
    try:
        reply1 = handle_message(db, session_id, "thanks, bye", BackgroundTasks())
    finally:
        db.close()
    check("Feedback (mocked): closing phrase triggers the feedback ask", FEEDBACK_ASK in reply1, reply1)
    check(
        "Feedback (mocked): session awaiting feedback after a closing phrase",
        feedback_agent.is_awaiting_response(session_id),
        None,
    )

    # User ignores the ask and asks a real, unrelated question instead.
    with patch("app.agents.intent_agent.get_ai_response", return_value="faq"):
        db = SessionLocal()
        try:
            reply2 = handle_message(db, session_id, "what services does XQORA provide?", BackgroundTasks())
        finally:
            db.close()

    check(
        "Feedback (mocked): unrelated follow-up is NOT swallowed as a feedback comment",
        FEEDBACK_ASK not in reply2,
        reply2,
    )
    check(
        "Feedback (mocked): feedback now marked declined, no longer awaited",
        not feedback_agent.is_awaiting_response(session_id) and not feedback_agent.should_ask(session_id),
        None,
    )

    db = SessionLocal()
    try:
        fb_count = db.query(Feedback).filter(Feedback.session_id == session_id).count()
    finally:
        db.close()
    check("Feedback (mocked): nothing saved to the Feedback table for the declined session", fb_count == 0, fb_count)

    # A later closing-style message in the same session must not re-ask.
    with patch("app.agents.intent_agent.get_ai_response", return_value="faq"):
        db = SessionLocal()
        try:
            reply3 = handle_message(db, session_id, "thanks, bye", BackgroundTasks())
        finally:
            db.close()
    check(
        "Feedback (mocked): feedback ask does not repeat later in the same session",
        FEEDBACK_ASK not in reply3,
        reply3,
    )

    feedback_agent.reset(session_id)
    lead_agent.reset_session(session_id)


def run_mocked_tests():
    TEST_DB_PATH_MOCKED = TEST_DB_PATH
    if os.path.exists(TEST_DB_PATH_MOCKED):
        os.remove(TEST_DB_PATH_MOCKED)

    from app.database import init_db

    init_db()

    _bug1_context_off_topic_override()
    _bug2_lead_requires_explicit_signal()
    _bug3_bare_category_words()
    _bug4_no_nudge_after_off_topic_refusal()
    _bug5_self_identity()
    _bug_escalate_vs_unrelated_professional_service()
    _bug_check_in_and_trivia_misclassification()
    _feature_reuse_lead_details()

    _feedback_response_parsing_unit()
    feedback_session = _feedback_after_lead_submission()
    _feedback_rating_and_comment_saved(feedback_session)
    _feedback_after_escalation()
    _feedback_ignored_does_not_loop_or_misfire()

    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"\n{passed} passed, {failed} failed out of {len(results)}")

    engine.dispose()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    if "--mocked" in sys.argv:
        run_mocked_tests()
    else:
        main()
