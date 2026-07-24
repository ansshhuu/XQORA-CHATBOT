"""
AI model service layer.

Gemini is the primary model; Groq is the fallback on failure or rate-limit.
Kept abstracted so switching models later doesn't touch agent/route code.
"""
import logging
import os

from app.core.config import GEMINI_API_KEY, GROQ_API_KEY

logger = logging.getLogger("xqora.ai_service")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


class AIServiceError(Exception):
    """Raised when both Gemini and Groq fail to produce a response."""


def get_gemini_response(message: str, system_prompt: str | None = None) -> str:
    if not GEMINI_API_KEY:
        raise AIServiceError("Gemini API key not configured")

    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt,
    )
    try:
        result = model.generate_content(message)
    except Exception as exc:
        logger.warning("Gemini call failed: %s", type(exc).__name__)
        raise AIServiceError("Gemini request failed") from exc

    text = getattr(result, "text", None)
    if not text:
        raise AIServiceError("Gemini returned empty response")
    return text.strip()


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
        )
    except Exception as exc:
        logger.warning("Groq call failed: %s", type(exc).__name__)
        raise AIServiceError("Groq request failed") from exc

    choice = completion.choices[0].message.content if completion.choices else None
    if not choice:
        raise AIServiceError("Groq returned empty response")
    return choice.strip()


def get_ai_response(message: str, system_prompt: str | None = None) -> str:
    """Groq only for now; Gemini disabled due to quota issues on the API key."""
    # try:
    #     return get_gemini_response(message, system_prompt)
    # except AIServiceError as gemini_err:
    #     logger.info("Falling back to Groq after Gemini failure")
    return get_groq_response(message, system_prompt)
