import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("xqora.config")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
TEAM_EMAIL = os.getenv("TEAM_EMAIL", "")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./xqora_chatbot.db")

# Comma-separated allowed origins for the widget to call this API from.
# PLACEHOLDER default below — replace with XQORA's real production domain(s)
# via the CORS_ORIGINS env var before deploying. Never use "*" in production:
# it would let any website embed the widget and hit this API.
CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "https://xqora.com,https://www.xqora.com").split(",") if o.strip()
]

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))

MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "1000"))


def check_required_keys() -> None:
    """Log (never print values) which required keys are missing. Call once at startup."""
    missing = [
        name
        for name, val in [
            ("GEMINI_API_KEY", GEMINI_API_KEY),
            ("GROQ_API_KEY", GROQ_API_KEY),
        ]
        if not val
    ]
    if missing:
        logger.warning("Missing required env keys: %s", ", ".join(missing))
