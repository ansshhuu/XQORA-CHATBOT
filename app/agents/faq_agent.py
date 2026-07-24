"""
FAQ agent: lightweight RAG over data/faq.json. Rather than pre-selecting one
"best" entry with a keyword/fuzzy matcher and hoping it guessed right, the
full FAQ list (all ~46 Q&A pairs - small enough to fit comfortably in one
prompt, no vector DB needed) is passed as context on every call, and the AI
itself finds whatever entry (or entries) actually answer the question and
phrases a natural reply grounded strictly in that content.

This replaced an earlier keyword/fuzzy-token-overlap matcher used to
pre-select context. That approach kept failing on real phrasings that didn't
share enough vocabulary with the FAQ text it was semantically closest to -
e.g. "about your product" scored zero against every entry and fell through
to the generic "I don't have that information" reply, while "pricing
details" scored just high enough against an unrelated entry to "win" and
answer the wrong question. Sanding down the scoring function keeps just
relocating the same class of failure to the next untested phrasing, since a
fixed keyword/stem heuristic fundamentally can't generalize to phrasing it
wasn't tuned against - the AI reading the full list and reasoning about it
does generalize, so that reasoning is now the primary path.

The matcher is kept, unchanged, as a fallback: if the AI call itself fails
(get_ai_response raises), something imperfect grounded in a real match beats
nothing when there's no AI available to reason over the full list.
"""
import json
import logging
import re
from pathlib import Path

from app.prompt import FAQ_SYSTEM_PROMPT
from app.services.ai_service import AIServiceError, get_ai_response

logger = logging.getLogger("xqora.faq_agent")

FAQ_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "faq.json"

_STOPWORDS = {
    "the", "a", "an", "is", "are", "do", "does", "how", "what", "can", "i",
    "you", "we", "to", "for", "of", "in", "on", "and", "or", "my", "your",
    "xqora", "with", "about",
}


_MIN_CONFIDENT_SCORE = 3

_FALLBACK_MESSAGE = (
    "I don't have that information on hand. Please reach out to our team "
    "directly at xqoratechnologies@gmail.com for more details."
)


def _load_faqs() -> list[dict]:
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["faqs"]


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _tokens_match(query_token: str, candidate_token: str) -> bool:
    """Exact match, or a lightweight stem/prefix match for word variants
    (automate/automation, comply/compliance, etc.) without a full stemming
    library. Short tokens (<4 chars) require an exact match to avoid
    coincidental prefix collisions."""
    if query_token == candidate_token:
        return True
    if len(query_token) < 4 or len(candidate_token) < 4:
        return False

    shorter_len = min(len(query_token), len(candidate_token))
    common_prefix = 0
    for a, b in zip(query_token, candidate_token):
        if a != b:
            break
        common_prefix += 1
    return common_prefix >= max(3, round(shorter_len * 0.6))


def _score_faq(query_tokens: set[str], faq: dict) -> int:
    """Each query token contributes AT MOST once per FAQ (2 points if it hits
    the question, else 1 if it only hits the answer, else 0) - a token
    matching in both question and answer of the SAME faq must not count
    twice, or a single near-universal word like "project" (present in both
    the question and answer of almost every FAQ) can inflate an otherwise
    unrelated entry past the confidence threshold on its own."""
    question_tokens = _tokenize(faq["question"])
    answer_tokens = _tokenize(faq["answer"])

    score = 0
    for qt in query_tokens:
        if any(_tokens_match(qt, ft) for ft in question_tokens):
            score += 2
        elif any(_tokens_match(qt, ft) for ft in answer_tokens):
            score += 1
    return score


def find_best_faqs(message: str, top_n: int = 3) -> list[dict]:
    """Rank FAQ entries by fuzzy token overlap against BOTH question and
    answer content, returning only entries that clear _MIN_CONFIDENT_SCORE -
    never an arbitrary/best-of-a-bad-lot match.

    Only used now as the AI-unavailable fallback (see get_faq_response) -
    the normal path passes the full FAQ list to the AI instead of relying on
    this to pre-select context."""
    query_tokens = _tokenize(message)
    if not query_tokens:
        return []

    scored = [
        (score, faq)
        for faq in _load_faqs()
        if (score := _score_faq(query_tokens, faq)) >= _MIN_CONFIDENT_SCORE
    ]

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [faq for _, faq in scored[:top_n]]


def _build_faq_prompt(message: str, recent_context: str | None) -> str:
    faq_context = "\n\n".join(f"Q: {faq['question']}\nA: {faq['answer']}" for faq in _load_faqs())
    return (
        (f"Recent conversation so far:\n{recent_context}\n\n" if recent_context else "")
        + f"XQORA's full FAQ list:\n{faq_context}\n\n"
        + f"Latest user message: {message}\n\n"
        + "Find whichever FAQ entry (or entries) above actually answers this, even if the "
        "user's wording is quite different from the question text, and give a natural, "
        "concise answer grounded strictly in that content. If more than one entry is "
        "relevant, combine them naturally. If the latest message is a short reply "
        "continuing the conversation above, answer in that context instead of treating it "
        "as a fresh, standalone question. If nothing in the list actually covers it, say so "
        "rather than guessing."
    )


def _fallback_response(message: str, recent_context: str | None) -> str:
    """Best-effort answer via the local keyword matcher, used only when the
    AI call itself is unavailable (see get_faq_response)."""
    matches = find_best_faqs(message)
    if not matches and recent_context:
        matches = find_best_faqs(f"{recent_context}\n{message}")
    if not matches:
        return _FALLBACK_MESSAGE
    return matches[0]["answer"]


def get_faq_response(message: str, recent_context: str | None = None) -> str:
    prompt = _build_faq_prompt(message, recent_context)
    try:
        return get_ai_response(prompt, system_prompt=FAQ_SYSTEM_PROMPT)
    except AIServiceError:
        logger.warning("AI FAQ call failed; falling back to keyword-matched answer")
        return _fallback_response(message, recent_context)
