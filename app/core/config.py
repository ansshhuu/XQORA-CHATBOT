import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("xqora.config")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
TEAM_EMAIL = os.getenv("TEAM_EMAIL", "")

GOOGLE_SHEETS_CREDENTIALS_PATH = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH", "")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./xqora_chatbot.db")

CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "https://xqora.com,https://www.xqora.com").split(",") if o.strip()
]

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))

MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "1000"))

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))


def check_required_keys() -> None:
    """Log (never print values) which required keys are missing. Call once at startup."""
    missing = [
        name
        for name, val in [
            ("GROQ_API_KEY", GROQ_API_KEY),
        ]
        if not val
    ]
    if missing:
        logger.warning("Missing required env keys: %s", ", ".join(missing))
