"""
AI model service layer. Groq is the sole provider - kept abstracted so
switching/adding providers later doesn't touch agent/route code.
"""
import logging
import os

from app.core.config import GROQ_API_KEY

logger = logging.getLogger("xqora.ai_service")

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")


class AIServiceError(Exception):
    """Raised whenGroq fail to produce a response."""





def get_groq_response(message: str, system_prompt: str | None = None) -> str:
    if not GROQ_API_KEY:
        raise AIServiceError("Groq API key not configured")

    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            extra_body={"reasoning_format": "hidden"},
        )
    except Exception as exc:
        logger.warning("Groq call failed: %s", type(exc).__name__)
        raise AIServiceError("Groq request failed") from exc

    choice = completion.choices[0].message.content if completion.choices else None
    if not choice:
        raise AIServiceError("Groq returned empty response")
    return choice.strip()


def get_ai_response(message: str, system_prompt: str | None = None) -> str:
    """Groq only for now."""
    return get_groq_response(message, system_prompt)
