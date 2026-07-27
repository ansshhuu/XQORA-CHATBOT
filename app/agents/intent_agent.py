"""
Intent classification + hard topic guardrail.

Guardrail is enforced in multiple layers, not left to the model alone:
  1. _is_near_empty() / _GREETING_RE / _is_fuzzy_greeting() - pure code
     pre-checks for empty input, plain greetings, and typo'd greetings ("hie",
     "hwello"), short-circuited to a friendly canned reply before any AI call
     or off-topic judgment runs. Deterministic on purpose: the AI's judgment
     on short, ambiguous, typo'd text is not reliably consistent call to call.
  2. hard_topic_filter() - pure code regex/keyword gate, runs before any AI call.
  3. hard_escalate_signal() - pure code pre-check forcing escalate for a
     narrow set of phrase combinations (a government tender tied to an actual
     XQORA service, lawsuits, litigation) that the AI was observed classifying
     as faq inconsistently call to call - same non-determinism problem as
     greeting detection, on the escalate/faq boundary instead.
  3b. _is_unrelated_professional_service() - pure code pre-check forcing
      off_topic for asks that sound formal/serious but are for a professional
      service entirely outside XQORA's actual catalog (legal/compliance
      audits, tax filing, HR consulting, accounting) - escalating these would
      imply the team can help, which is misleading, so they're refused the
      same as any other off_topic request instead. Checked before
      hard_escalate_signal so a stray "audit"/"tender"-adjacent word doesn't
      let one override the other.
  4. INTENT_SYSTEM_PROMPT - instructs the model to classify off-topic requests
     as off_topic, to prefer escalate over off_topic for genuine business needs
     it can't confidently answer, and (given recent_context) to read a short
     reply as continuing the conversation rather than judging it standalone.
  5. Fail-closed parsing - any model output outside the 4 allowed labels is
     treated as off_topic (covers "off_topic" itself and garbage output).

Greetings and near-empty input reuse the IntentResult(blocked=True,
refusal_message=...) shape as a general "return this text directly, skip
every specialist agent" signal, not just for genuine refusals, so the
orchestrator's existing blocked-handling short-circuits to them too.
"""
import itertools
import logging
import random
import re
from dataclasses import dataclass

from app.prompt import (
    BARE_CATEGORY_RESPONSE,
    CHECK_IN_RESPONSE,
    CLARIFY_BARE_WORD_RESPONSE,
    EMPTY_INPUT_PROMPT,
    GREETING_RESPONSE,
    GUARDRAIL_REFUSAL_MESSAGE,
    INTENT_SYSTEM_PROMPT,
    SELF_IDENTITY_RESPONSE,
)
from app.services.ai_service import AIServiceError, get_ai_response
from app.utils import GREETING_WORDS

logger = logging.getLogger("xqora.intent_agent")

VALID_INTENTS = {"faq", "recommend", "lead", "escalate"}

ON_TOPIC_KEYWORDS = [
    "xqora", "internship", "intern", "service", "services", "pricing", "price",
    "cost", "quote", "quotation", "contact", "faq", "support", "maintenance",
    "website", "web development", "software development", "ai automation",
    "automation", "cloud", "chatbot", "project", "budget", "hire", "apply",
    "resume", "email", "phone", "technical support",
]


OFF_TOPIC_PATTERNS = [
    r"\bwrite\s+(me|us)?\s*(a|an|the)?\s*(python|java|javascript|c\+\+|sql|regex|html|css)?\s*(code|script|program|function|algorithm)\b",
    r"\bdebug\s+(this|my)\s+code\b",
    r"\brecipe\b",
    r"\bhow\s+(do|to)\s+(i\s+)?cook\b",
    r"\btell\s+me\s+a\s+joke\b",
    r"\bwrite\s+(me|us)?\s*(a|an)?\s*(poem|song|lyrics|story)\b",
    r"\btranslate\s+.+\s+(to|into)\s+\w+\b",
    r"\bsolve\s+(this|the)\s+(math|equation|problem)\b",
    r"\bhomework\b",
    r"\bcapital\s+of\b",
    r"\bpresident\s+of\b",
    r"\bprime\s+minister\b",
    r"\bweather\s+(in|today|forecast)\b",
    r"\bstock\s+price\b",
    r"\blatest\s+news\b",
    r"\bwho\s+(won|win)\b",
    r"\bwhat\s+time\s+is\s+it\b",
    r"\bwhat'?s\s+the\s+time\b",
    r"\bcurrent\s+time\b",
]
_OFF_TOPIC_RE = [re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS]


_HARD_ESCALATE_PATTERNS = [
    r"\blawsuit\b",
    r"\blitigation\b",
]
_HARD_ESCALATE_RE = [re.compile(p, re.IGNORECASE) for p in _HARD_ESCALATE_PATTERNS]

_TENDER_RE = re.compile(r"\btender\b", re.IGNORECASE)


_XQORA_SERVICE_KEYWORDS = [
    "website", "web app", "web application", "webapp", "web development",
    "software", "custom software", "software development",
    "ai solution", "ai automation", "artificial intelligence", "chatbot", "automation",
    "cloud", "aws", "azure", "google cloud", "gcp", "server", "hosting", "infrastructure",
    "migrate", "migration", "deployment", "mainframe",
    "mobile app", "mobile application", "android", "ios", "app development",
    "digital solution",
    "student project", "academic project", "internship",
    "erp", "crm", "enterprise system", "inventory system", "hr portal",
    "technical support", "maintenance", "bug fix",
    "platform", "system", "application", "portal", "integration", "database",
    "project",
]


def _mentions_xqora_service(message: str) -> bool:
    lower = message.lower()
    return any(keyword in lower for keyword in _XQORA_SERVICE_KEYWORDS)


def hard_escalate_signal(message: str) -> bool:
    """True if the message matches one of the narrow high-signal escalate
    conditions above, before any AI classification runs."""
    lower = message.lower()
    if any(pattern.search(lower) for pattern in _HARD_ESCALATE_RE):
        return True
    return bool(_TENDER_RE.search(lower)) and _mentions_xqora_service(lower)



_UNRELATED_PROFESSIONAL_SERVICE_PATTERNS = [
    r"\b(legal|compliance|regulatory)\b.*\baudit\b",
    r"\baudit\b.*\b(legal|compliance|regulatory)\b",
    r"\bfile\s+(my|our)\s+tax(es)?\b",
    r"\btax\s+(advice|filing|return|preparation|consultant)\b",
    r"\baccounting\s+(services|advice|help|firm)\b",
    r"\bhr\s+consult(ing|ant)\b",
    r"\blegal\s+advice\b",
]
_UNRELATED_PROFESSIONAL_SERVICE_RE = [
    re.compile(p, re.IGNORECASE) for p in _UNRELATED_PROFESSIONAL_SERVICE_PATTERNS
]


def _is_unrelated_professional_service(message: str) -> bool:
    lower = message.lower()
    return any(pattern.search(lower) for pattern in _UNRELATED_PROFESSIONAL_SERVICE_RE)


def _stretch(word: str) -> str:
    """Turn "hello" into a regex allowing any letter to repeat, so texting-style
    stretches like "heyyy" or "helllllllloooooooo" still match "hey"/"hello"."""
    return "".join(f"{re.escape(ch)}+" for ch, _ in itertools.groupby(word))


_GREETING_TIME_WORDS = ["morning", "afternoon", "evening"]


_simple_greeting_alt = "|".join(_stretch(w) for w in sorted(GREETING_WORDS))
_time_alt = "|".join(_stretch(w) for w in _GREETING_TIME_WORDS)
_good_greeting_pattern = _stretch("good") + r"\s+(?:" + _time_alt + ")"

_GREETING_RE = re.compile(
    rf"^\s*(?:{_simple_greeting_alt}|{_good_greeting_pattern})[\s!.,]*$",
    re.IGNORECASE,
)

_GREETING_RESPONSES = (
    GREETING_RESPONSE,
    "Hey! What's up, anything I can help with?",
    "Hi there! What brings you by today?",
    "Hey, good to see you! What's on your mind?",
    "Hello! Got a project or question for me?",
)

_NON_WORD_RE = re.compile(r"[^\w]")
_TRAILING_PUNCT_RE = re.compile(r"[!?.,]+$")

_FUZZY_GREETING_WORDS = ("hi", "hey", "hello", "howdy", "greetings")
_FUZZY_GREETING_MAX_LEN = 10


def _levenshtein(a: str, b: str) -> int:
    """Standard edit distance (insertions/deletions/substitutions). Small
    inputs only (bounded by _FUZZY_GREETING_MAX_LEN below), so the plain
    O(n*m) DP table is plenty - no dependency needed."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def _is_fuzzy_greeting(message: str) -> bool:
    stripped = _TRAILING_PUNCT_RE.sub("", message.strip().lower())
    if not stripped or len(stripped) > _FUZZY_GREETING_MAX_LEN:
        return False

    for word in _FUZZY_GREETING_WORDS:
        threshold = 1 if len(word) <= 3 else 2
        if _levenshtein(stripped, word) <= threshold:
            return True
    return False


_BARE_CATEGORY_WORDS = {
    "faq", "faqs", "pricing", "price", "prices", "cost", "costs", "quote",
    "quotes", "quotation",
}


def _is_bare_category_word(message: str) -> bool:
    stripped = _TRAILING_PUNCT_RE.sub("", message.strip().lower())
    return stripped in _BARE_CATEGORY_WORDS


_BARE_WORD_SHAPE_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{1,29}")
_TRAILING_SOFT_PUNCT_RE = re.compile(r"[.!]+$")


def _bare_unclassified_word(message: str) -> str | None:
    """Returns the stripped word if `message` is a single bare word - no
    verb/question structure, not a recognized greeting, category, or
    on-topic/service keyword - that on its own gives the AI classifier
    nothing to go on. Recent_context-aware AI classification would otherwise
    tend to just carry forward whatever topic the conversation was already
    on (see INTENT_SYSTEM_PROMPT's short-reply-continues-context rule),
    which is right for an actual short reply ("yeah a lot") but wrong for an
    unrelated stray word like a name ("nikhil") that happens to arrive after
    an unrelated topic. Returns None if this doesn't apply (multi-word
    input, contains "?", or the word itself is already meaningful on its
    own), so the caller falls through to normal classification.

    Never used to name-match against a stored lead - that's a session-scoped
    check orchestrator.py already owns (_own_info_reply) and runs before
    classify_intent is even called, so a bare word that IS this session's
    own stored name is handled as a self-recall there and never reaches
    this function."""
    stripped = message.strip()
    if "?" in stripped:
        return None
    stripped = _TRAILING_SOFT_PUNCT_RE.sub("", stripped).strip()
    if not _BARE_WORD_SHAPE_RE.fullmatch(stripped):
        return None

    lower = stripped.lower()
    if lower in GREETING_WORDS or lower in _BARE_CATEGORY_WORDS:
        return None
    if lower in ON_TOPIC_KEYWORDS or _mentions_xqora_service(lower):
        return None
    return stripped



_SELF_IDENTITY_RE = re.compile(
    r"^\s*(who\s+are\s+you|what\s+are\s+you|what\s+is\s+xqora\s+assistant|"
    r"what('?s|\s+is)\s+your\s+name|are\s+you\s+(a\s+)?(bot|human|ai|real|"
    r"assistant)|what\s+do\s+you\s+do|tell\s+me\s+about\s+yourself)\s*[?!.]*\s*$",
    re.IGNORECASE,
)


def _is_self_identity_question(message: str) -> bool:
    return bool(_SELF_IDENTITY_RE.match(message.strip()))



_CHECK_IN_RE = re.compile(
    r"^\s*(hey[,!]?\s*)?(are\s+you\s+(still\s+)?there|you\s+(still\s+)?there|"
    r"(is\s+)?(anyone|anybody)\s+(there|here))\s*[?!.]*\s*$",
    re.IGNORECASE,
)


def _is_check_in_message(message: str) -> bool:
    return bool(_CHECK_IN_RE.match(message.strip()))



_EXPLICIT_LEAD_RE = re.compile(
    r"\b(hire\s+(you|xqora|us)|get\s+(me\s+)?(a\s+)?quot(e|ation)|want\s+(a\s+)?quot(e|ation)|"
    r"contact\s+me|contact\s+us|reach\s+out\s+to\s+me|get\s+in\s+touch|sign\s+(me\s+)?up|"
    r"book\s+a\s+call|schedule\s+a\s+(call|meeting|demo)|talk\s+to\s+(a\s+)?(human|someone|"
    r"the\s+team|your\s+team|a\s+rep)|connect\s+me|get\s+started|start\s+(a|my)\s+project|"
    r"get\s+a\s+proposal|become\s+a\s+client|work\s+with\s+(you|xqora))\b",
    re.IGNORECASE,
)


def _explicit_lead_signal(message: str) -> bool:
    return bool(_EXPLICIT_LEAD_RE.search(message.strip().lower()))


@dataclass
class IntentResult:
    intent: str
    blocked: bool
    refusal_message: str | None = None


def _is_near_empty(message: str) -> bool:
    """True for "", whitespace-only, or punctuation-only input ("...", "??")."""
    return len(_NON_WORD_RE.sub("", message.strip())) == 0


def hard_topic_filter(message: str) -> bool:
    """Return True if message must be BLOCKED as off-topic, before any AI call.

    ORDERING RISK: this runs BEFORE the context-aware AI classifier ever sees
    the message (see classify_intent below), and it is context-blind by
    design - it only ever looks at the current message in isolation, never
    recent_context. That's currently low-risk because OFF_TOPIC_PATTERNS is a
    short, conservative list of fairly specific multi-word phrases (e.g.
    "write me a script", "capital of"), so a short contextual reply like
    "yeah a lot of requests" has no realistic way to match one. But if this
    pattern list is ever broadened or made more aggressive, a short
    contextual reply COULD coincidentally match a pattern here and get
    blocked before the context-aware classifier gets a chance to clear it.
    Any future edit to OFF_TOPIC_PATTERNS should keep patterns specific
    enough to avoid that, or this function will need its own context
    parameter to guard against it.
    """
    lower = message.lower()
    if any(keyword in lower for keyword in ON_TOPIC_KEYWORDS):
        return False
    return any(pattern.search(lower) for pattern in _OFF_TOPIC_RE)


def classify_intent(
    message: str, recent_context: str | None = None, previous_intent: str | None = None
) -> IntentResult:
    """recent_context, when given, is the last few turns of this session's
    chat history formatted as "User: ...\\nBot: ..." lines (oldest first).
    It lets the AI recognize a short reply like "yeah a lot" as continuing
    whatever the bot just asked about, instead of judging it standalone and
    falling back to off_topic. previous_intent, when given, is that same
    session's last logged intent (faq/recommend/lead/escalate) - a
    deterministic fallback for when the AI's own context-reading isn't
    reliable (see the off_topic override below), not just more text fed into
    the same AI call. The pure code pre-checks below (near-empty, greeting,
    hard_topic_filter) intentionally stay context-blind: they're a fast
    surface-form gate, not a meaning judgment, so they don't need it."""
    if _is_near_empty(message):
        return IntentResult(intent="greeting", blocked=True, refusal_message=EMPTY_INPUT_PROMPT)

    if _GREETING_RE.match(message.strip()) or _is_fuzzy_greeting(message):
        return IntentResult(intent="greeting", blocked=True, refusal_message=random.choice(_GREETING_RESPONSES))

    if _is_bare_category_word(message):
        return IntentResult(intent="faq", blocked=True, refusal_message=BARE_CATEGORY_RESPONSE)

    if _is_self_identity_question(message):
        return IntentResult(intent="faq", blocked=True, refusal_message=SELF_IDENTITY_RESPONSE)

    if _is_check_in_message(message):
        return IntentResult(intent="greeting", blocked=True, refusal_message=CHECK_IN_RESPONSE)

    if hard_topic_filter(message):
        return IntentResult(intent="off_topic", blocked=True, refusal_message=GUARDRAIL_REFUSAL_MESSAGE)

    if _is_unrelated_professional_service(message):
        return IntentResult(intent="off_topic", blocked=True, refusal_message=GUARDRAIL_REFUSAL_MESSAGE)

    if hard_escalate_signal(message):
        return IntentResult(intent="escalate", blocked=False)

    bare_word = _bare_unclassified_word(message)
    if bare_word is not None:
        return IntentResult(
            intent="off_topic", blocked=True, refusal_message=CLARIFY_BARE_WORD_RESPONSE.format(word=bare_word)
        )

    if recent_context:
        classification_input = f"Recent conversation so far:\n{recent_context}\n\nLatest user message: {message}"
    else:
        classification_input = message

    try:
        raw = get_ai_response(classification_input, system_prompt=INTENT_SYSTEM_PROMPT)
    except AIServiceError:
        logger.warning("Intent classification AI call failed; defaulting to faq")
        return IntentResult(intent="faq", blocked=False)

    label = re.sub(r"[^a-z_]", "", raw.strip().lower())
    if label not in VALID_INTENTS:
        label = "off_topic"

    if label == "off_topic" and recent_context and previous_intent in VALID_INTENTS:
       
        try:
            retry_raw = get_ai_response(classification_input, system_prompt=INTENT_SYSTEM_PROMPT)
            retry_label = re.sub(r"[^a-z_]", "", retry_raw.strip().lower())
        except AIServiceError:
            retry_label = "off_topic"
        if retry_label in VALID_INTENTS and retry_label != "off_topic":
            label = retry_label

    if label == "lead" and not _explicit_lead_signal(message):
       
        label = previous_intent if previous_intent in VALID_INTENTS and previous_intent != "lead" else "recommend"

    if label == "off_topic":
        return IntentResult(intent="off_topic", blocked=True, refusal_message=GUARDRAIL_REFUSAL_MESSAGE)
    return IntentResult(intent=label, blocked=False)
