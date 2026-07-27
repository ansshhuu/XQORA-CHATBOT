import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("xqora.config")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
TEAM_EMAIL = os.getenv("TEAM_EMAIL", "")

GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON", "")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "")

try:
    DATABASE_URL = os.environ["DATABASE_URL"]
except KeyError as exc:
    raise RuntimeError(
        "DATABASE_URL environment variable is required (a Postgres connection string "
        "in production/Vercel; sqlite:///./some.db is fine for local dev). There is no "
        "default - set it in .env for local dev, in Vercel's project env vars for "
        "deployment, or export it explicitly before running tests."
    ) from exc

CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "https://xqora.com,https://www.xqora.com").split(",") if o.strip()
]

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))

MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "1000"))

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))

# Only ever set (to "true") in local/CI test runs - never in production or
# Vercel env vars. Gates ai_service.py and lead_service.py so test runs never
# hit real Groq/OpenRouter/Resend quota. See TESTING.md.
TEST_MODE = os.getenv("TEST_MODE", "false").strip().lower() in ("1", "true", "yes")


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


def check_test_mode_production_safety() -> None:
    """TEST_MODE must never be active in production - it stubs out real
    Groq/OpenRouter/Resend calls, so a deploy with it accidentally left on
    would silently stop sending real AI replies and real lead emails. Call
    once at startup; this only logs (loudly) rather than raising, since a
    misconfigured env var shouldn't crash the app outright."""
    if not TEST_MODE:
        return

    looks_like_prod_db = DATABASE_URL.startswith(("postgres://", "postgresql://")) and not any(
        marker in DATABASE_URL for marker in ("localhost", "127.0.0.1")
    )
    looks_like_prod_env = os.getenv("VERCEL_ENV", "").strip().lower() == "production"

    if looks_like_prod_db or looks_like_prod_env:
        logger.error(
            "!!! TEST_MODE=true is set in what looks like a PRODUCTION environment "
            "(VERCEL_ENV=%r, DATABASE_URL looks like a remote Postgres=%s) - real AI "
            "responses and lead emails are being STUBBED OUT. Unset TEST_MODE immediately.",
            os.getenv("VERCEL_ENV", ""),
            looks_like_prod_db,
        )
