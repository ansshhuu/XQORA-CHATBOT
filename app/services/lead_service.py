"""
Handles the contact/service-request branch: collects lead details
and forwards them to the team via the Resend API.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

import resend

from app.core.config import RESEND_API_KEY, TEAM_EMAIL

logger = logging.getLogger("xqora.lead_service")

resend.api_key = RESEND_API_KEY

# The resend SDK has no timeout parameter anywhere (Emails.send() calls
# requests.request() internally with none, and there's no global config hook
# either), so a stalled connection to Resend could otherwise hang forever.
# Enforced here at the application level instead, since the SDK gives us no
# other option. This is called from lead_agent's background task, so a
# timeout here only delays marking the lead "forwarded" - it never touches
# the user-facing chat response either way.
_SEND_TIMEOUT_SECONDS = 10
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="resend-send")


def forward_lead(name: str, contact: str, need: str) -> bool:
    def _send() -> None:
        resend.Emails.send(
            {
                "from": "onboarding@resend.dev",
                "to": TEAM_EMAIL,
                "subject": "New Chatbot Lead",
                "text": f"Name: {name}\nContact: {contact}\nNeed: {need}",
            }
        )

    try:
        _executor.submit(_send).result(timeout=_SEND_TIMEOUT_SECONDS)
        return True
    except FutureTimeoutError:
        logger.error("Resend request timed out after %ss", _SEND_TIMEOUT_SECONDS)
        return False
    except Exception:
        logger.exception("Failed to forward lead via Resend")
        return False
