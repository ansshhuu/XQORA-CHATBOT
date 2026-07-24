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
