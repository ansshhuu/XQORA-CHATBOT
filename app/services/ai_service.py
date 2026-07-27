"""
AI model service layer. Groq is the primary provider; OpenRouter is a
one-shot fallback used only when a Groq call itself fails (timeout, rate
limit, connection error, non-2xx, empty response) - kept abstracted so
neither the fallback nor any future provider change touches agent/route
code, which only ever calls get_ai_response().
"""
import logging
import os
import re

import requests

from app.core.config import GROQ_API_KEY, OPENROUTER_API_KEY, TEST_MODE

logger = logging.getLogger("xqora.ai_service")

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")

# meta-llama/llama-3.3-70b-instruct:free was the original pick for this but
# is no longer listed on openrouter.ai/models (confirmed via the models API
# before choosing this replacement) - Meta has pulled its free Llama
# endpoints from OpenRouter entirely as of this writing. gpt-oss-20b is a
# general-purpose (not code- or vision-specialized) instruct model on the
# current free tier with a large context window, a reasonable substitute
# for the same fallback role.
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_TIMEOUT_SECONDS = 15


class AIServiceError(Exception):
    """Raised whenGroq fail to produce a response."""


# The model we use is a reasoning model; `extra_body={"reasoning_format":
# "hidden"}` below asks Groq to keep the reasoning trace out of the response
# entirely, but that's a provider-side hint, not a guarantee - a provider
# hiccup or model change could still hand back a raw <think>...</think>
# block. Every caller (intent classification, FAQ/recommend answers, lead
# field-plausibility checks) treats this return value as user-facing or
# decision text as-is, so this is the one place to strip a leaked block
# rather than relying on every call site to remember to.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.IGNORECASE | re.DOTALL)


def _strip_reasoning_leak(text: str) -> str:
    stripped = _THINK_BLOCK_RE.sub("", text)
    stripped = _UNCLOSED_THINK_RE.sub("", stripped)
    return stripped.strip()


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
    cleaned = _strip_reasoning_leak(choice.strip())
    if not cleaned:
        raise AIServiceError("Groq response was only a reasoning trace, no answer")
    return cleaned


def _call_openrouter(messages: list[dict], **kwargs) -> str:
    """Hits OpenRouter's OpenAI-compatible chat completions endpoint and
    normalizes the response to the same plain-string shape get_groq_response
    returns, so get_ai_response can treat both providers identically.
    `**kwargs` is passed straight through into the request payload (e.g.
    temperature, max_tokens) for future callers that need it - unused today."""
    if not OPENROUTER_API_KEY:
        raise AIServiceError("OpenRouter API key not configured")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": OPENROUTER_MODEL, "messages": messages, **kwargs}

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=_OPENROUTER_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("OpenRouter call failed: %s", type(exc).__name__)
        raise AIServiceError("OpenRouter request failed") from exc

    if data.get("error"):
        raise AIServiceError(f"OpenRouter returned an error: {data['error']}")

    choices = data.get("choices") or []
    choice = choices[0]["message"]["content"] if choices else None
    if not choice:
        raise AIServiceError("OpenRouter returned empty response")
    cleaned = _strip_reasoning_leak(choice.strip())
    if not cleaned:
        raise AIServiceError("OpenRouter response was only a reasoning trace, no answer")
    return cleaned


def get_ai_response(message: str, system_prompt: str | None = None) -> str:
    """Groq first; on any failure, one fallback attempt against OpenRouter
    before giving up. Not a retry loop - each provider gets exactly one try,
    so a genuinely down AI layer fails fast instead of hammering either
    provider's rate limit (OpenRouter's free tier in particular caps at
    20 req/min, but this path only fires when Groq has already failed, so
    it stays rare). If OpenRouter also fails, this raises AIServiceError
    same as before - every call site (intent_agent, faq_agent,
    recommend_agent, lead_agent) already catches that and falls back to its
    own keyword-matched/heuristic response, so that existing last-resort
    behavior is unchanged and needs no edits here."""
    if TEST_MODE:
        # Never hit real Groq/OpenRouter in a test run. Raising the same
        # AIServiceError a genuine provider outage would produce - rather
        # than inventing a new canned string - means every caller already has
        # tested, deterministic fallback behavior for this (see
        # tests/test_e2e.py's _bug7_total_ai_outage_no_fake_content and each
        # agent's own AIServiceError handling), so no new mock fixtures are
        # needed here. Tests that need a specific AI answer instead patch
        # get_ai_response directly at the call site (see test_e2e.py
        # --mocked), which takes precedence over this guard entirely.
        logger.info("[TEST MODE] skipping real Groq/OpenRouter call; raising AIServiceError for caller fallback")
        raise AIServiceError("TEST_MODE active - real AI providers are not called")

    try:
        result = get_groq_response(message, system_prompt)
        logger.info("AI response served_by=groq")
        return result
    except AIServiceError as exc:
        logger.warning("Groq failed (%s); falling back to OpenRouter", exc)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})

    try:
        result = _call_openrouter(messages)
        logger.info("AI response served_by=openrouter")
        return result
    except AIServiceError as exc:
        logger.error("OpenRouter fallback also failed (%s); no AI provider available", exc)
        raise
