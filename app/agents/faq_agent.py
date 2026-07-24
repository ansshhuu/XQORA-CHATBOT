"""
FAQ agent: finds the best-matching FAQ entries, then uses ai_service to
phrase a natural answer grounded strictly in that content.

Accepts optional recent_context (recent chat history, same shape orchestrator
already builds for intent_agent) so a short follow-up reply with no FAQ
keywords of its own can still be matched and answered in context, instead of
falling back to a generic "don't have that info" reply.

Matching considers BOTH the FAQ's question and its answer (not just the
question), uses fuzzy/stem-tolerant token matching (so "automate" connects to
"automation", "comply" to "compliance"), and requires a minimum relevance
score before accepting a match at all - a single incidental shared filler
word ("help", "project", "support" all appear in several FAQ questions) is
not enough on its own to "win". This mirrors recommend_agent.py's matching
fix for the same class of bug; duplicated locally rather than shared via a
common module to keep this fix scoped to this file.

A true embedding-based semantic search was considered per the original bug
report, but isn't practical right now: Groq (the only currently-working AI
provider - see ai_service.py) has no embeddings endpoint, and Gemini's
embedding endpoint is unusable until its quota issue is resolved. Adding a
local embedding model would mean a heavy new dependency (sentence-transformers
+ torch) for a ~40-entry FAQ file. If a working embedding provider becomes
available later, that would be a stronger long-term fix than this scoring
approach.
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
    never an arbitrary/best-of-a-bad-lot match."""
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


def get_faq_response(message: str, recent_context: str | None = None) -> str:
    matches = find_best_faqs(message)

    if not matches and recent_context:
        matches = find_best_faqs(f"{recent_context}\n{message}")

    if not matches:
        return _FALLBACK_MESSAGE

    context = "\n\n".join(f"Q: {m['question']}\nA: {m['answer']}" for m in matches)
    prompt = (
        (f"Recent conversation so far:\n{recent_context}\n\n" if recent_context else "")
        + f"FAQ context:\n{context}\n\n"
        + f"Latest user message: {message}\n\n"
        + "Using ONLY the FAQ context above, give a natural, concise answer. If the "
        "latest message is a short reply continuing the conversation above, answer in "
        "that context instead of treating it as a fresh, standalone question."
    )
    try:
        return get_ai_response(prompt, system_prompt=FAQ_SYSTEM_PROMPT)
    except AIServiceError:
        logger.warning("AI phrasing failed for FAQ; returning best-match answer directly")
        return matches[0]["answer"]
