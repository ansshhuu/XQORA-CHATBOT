"""
Google Sheets integration: appends each completed lead as a new row (name,
company, email, phone, requirement, timestamp) to a shared spreadsheet, via
a service account - server-to-server auth, no per-user OAuth consent flow
needed. Wired into lead_agent.py's _forward_lead_async, the same
fire-and-forget background job that sends the email notification: both run
there, each independently error-handled, so a Sheets failure never affects
the email or the other way around.

GOOGLE_SHEETS_CREDENTIALS_PATH accepts either a filesystem path to the
service account's JSON key, or the JSON key's contents pasted inline
(detected by a leading "{") - useful for deployment targets where writing a
credentials file to disk isn't convenient and the key is injected as a
single env var instead.

The gspread client authenticates once per process and is reused across
calls - service account credentials don't expire the way user OAuth tokens
do, so there's no benefit to re-authenticating on every single lead.
"""
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timezone

from app.core.config import GOOGLE_SHEET_ID, GOOGLE_SHEETS_CREDENTIALS_PATH

logger = logging.getLogger("xqora.sheets_service")

_SEND_TIMEOUT_SECONDS = 10
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sheets-append")

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

_client_lock = threading.Lock()
_client = None


def _load_credentials():
    from google.oauth2.service_account import Credentials

    value = GOOGLE_SHEETS_CREDENTIALS_PATH.strip()
    if value.startswith("{"):
        return Credentials.from_service_account_info(json.loads(value), scopes=_SCOPES)
    return Credentials.from_service_account_file(value, scopes=_SCOPES)


def _get_client():
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            import gspread

            _client = gspread.authorize(_load_credentials())
    return _client


def append_lead_row(name: str, company: str, email: str, phone: str, requirement: str) -> bool:
    if not GOOGLE_SHEETS_CREDENTIALS_PATH or not GOOGLE_SHEET_ID:
        logger.warning("Google Sheets not configured (missing credentials or sheet ID); skipping row append")
        return False

    def _append() -> None:
        client = _get_client()
        worksheet = client.open_by_key(GOOGLE_SHEET_ID).sheet1
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        worksheet.append_row([name, company, email, phone, requirement, timestamp])

    try:
        _executor.submit(_append).result(timeout=_SEND_TIMEOUT_SECONDS)
        return True
    except FutureTimeoutError:
        logger.error("Google Sheets append timed out after %ss", _SEND_TIMEOUT_SECONDS)
        return False
    except Exception:
        logger.exception("Failed to append lead row to Google Sheets")
        return False
