"""
Escalation agent: hands off to a human with XQORA's contact info. Used when
intent_agent flags escalate, or when a specialist agent can't resolve a query
confidently.
"""
from app.core.config import TEAM_EMAIL
from app.prompt import ESCALATION_RESPONSE, ESCALATION_RESPONSE_WITH_REASON


def get_escalation_response(reason: str = "") -> str:
    contact_email = TEAM_EMAIL or "xqoratechnologies@gmail.com"
    if reason.strip():
        return ESCALATION_RESPONSE_WITH_REASON.format(reason=reason.strip(), email=contact_email)
    return ESCALATION_RESPONSE.format(email=contact_email)
