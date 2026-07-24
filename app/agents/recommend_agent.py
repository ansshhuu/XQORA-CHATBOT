"""
Recommendation agent: matches the user's stated need to XQORA services via
keyword overlap, then uses ai_service to phrase a short, natural recommendation
grounded in services.json facts (never invents services). No pricing or tier
talk here, that's handled by the team once a lead is captured.

Accepts optional recent_context (recent chat history, same shape orchestrator
already builds for intent_agent) so a short follow-up reply with no service
keywords of its own can still be matched and answered in context, instead of
falling back to a generic "tell me more" reply.
"""
import json
import logging
import re
from pathlib import Path

from app.prompt import RECOMMEND_SYSTEM_PROMPT
from app.services.ai_service import AIServiceError, get_ai_response

logger = logging.getLogger("xqora.recommend_agent")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SERVICES_PATH = DATA_DIR / "services.json"

_STOPWORDS = {
    "the", "a", "an", "is", "are", "do", "does", "how", "what", "can", "i",
    "you", "we", "to", "for", "of", "in", "on", "and", "or", "my", "your",
    "xqora", "with", "about", "need", "want", "looking", "help", "build",
}

_FALLBACK_MESSAGE = (
    "Could you share a bit more detail about what you're looking to build? "
    "That'll help me point you to the right XQORA service, or you can reach "
    "our team directly at xqoratechnologies@gmail.com."
)


def _load_services() -> list[dict]:
    with open(SERVICES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["services"]


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _tokens_match(query_token: str, keyword_token: str) -> bool:
    """Exact match, or a lightweight stem/prefix match for word variants
    (automate / automation / automating, host / hosting, etc.) without a
    full stemming library. Short tokens (<4 chars) require an exact match to
    avoid coincidental prefix collisions like "ai" vs "air"."""
    if query_token == keyword_token:
        return True
    if len(query_token) < 4 or len(keyword_token) < 4:
        return False

    shorter_len = min(len(query_token), len(keyword_token))
    common_prefix = 0
    for a, b in zip(query_token, keyword_token):
        if a != b:
            break
        common_prefix += 1
    return common_prefix >= max(4, round(shorter_len * 0.75))


def _phrase_matches(phrase_tokens: set[str], query_tokens: set[str]) -> bool:
    """A multi-word keyword phrase only counts if EVERY one of its words is
    present in the query (in any order), otherwise a single generic word
    shared between phrases (e.g. "app" inside both "web app" and "mobile
    app") would wrongly credit every service that happens to use it."""
    if not phrase_tokens:
        return False
    return all(any(_tokens_match(qt, pt) for qt in query_tokens) for pt in phrase_tokens)


def find_best_services(message: str) -> tuple[list[dict], bool]:
    """Score each service by how many of its keyword phrases (or its name)
    fully match the query, then return the winner(s).

    Returns (matches, ambiguous). "ambiguous" is True when two or more
    services tie for the top score, the caller should ask a clarifying
    question in that case rather than silently picking one.
    """
    query_tokens = _tokenize(message)
    if not query_tokens:
        return [], False

    scored = []
    for service in _load_services():
        phrases = service["keywords"] + [service["name"]]
        score = sum(1 for phrase in phrases if _phrase_matches(_tokenize(phrase), query_tokens))
        if score:
            scored.append((score, service))

    if not scored:
        return [], False

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top_score = scored[0][0]
    tied = [service for score, service in scored if score == top_score]
    return tied, len(tied) > 1


def get_recommendation_response(message: str, recent_context: str | None = None) -> str:
    matches, ambiguous = find_best_services(message)

    if not matches and recent_context:
        # A short follow-up like "yeah a lot of requests" has no service
        # keywords of its own, but the conversation it's replying to
        # (recent_context includes the bot's own last message) usually does -
        # retry against the combined text instead of falling back blind.
        matches, ambiguous = find_best_services(f"{recent_context}\n{message}")

    if not matches:
        return _FALLBACK_MESSAGE

    if ambiguous:
        names = " or ".join(s["name"] for s in matches[:3])
        return (
            f"A couple of things could fit here: {names}. "
            "Could you say a bit more about what you need so I point you to the right one?"
        )

    service_context = "\n\n".join(
        f"Service: {service['name']}\nDescription: {service['description']}" for service in matches
    )

    prompt = (
        (f"Recent conversation so far:\n{recent_context}\n\n" if recent_context else "")
        + f"Service context:\n{service_context}\n\n"
        + f"Latest user message: {message}\n\n"
        + "Mention the best-fitting service(s) above and briefly why, like a natural "
        "suggestion in conversation, not a templated report. If the latest message is a "
        "short reply continuing the conversation above, answer in that context instead "
        "of treating it as a fresh, standalone request. No pricing, cost, or tier talk. "
        "End with one short, relevant follow-up question (about their current setup, "
        "volume, or pain point) to keep the conversation going."
    )
    try:
        return get_ai_response(prompt, system_prompt=RECOMMEND_SYSTEM_PROMPT)
    except AIServiceError:
        logger.warning("AI phrasing failed for recommendation; returning raw match")
        names = " or ".join(s["name"] for s in matches)
        return f"Sounds like {names} could be a good fit for that. What does your current setup look like?"
