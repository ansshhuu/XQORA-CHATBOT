"""
Small shared helpers used by more than one agent. Kept dependency-free (no
imports from app.agents.* or app.orchestrator) so agents can import from here
without coupling to each other's internals.
"""
import re

GREETING_WORDS = {"hi", "hey", "hello", "yo", "sup", "howdy", "greetings"}

_TRAILING_PUNCTUATION_RE = re.compile(r"[!.,?]+$")


def is_bare_greeting(text: str) -> bool:
    """True if `text` is just a common greeting word, optionally with
    trailing punctuation ("hi", "hey!", "Hello."). Deliberately simple/exact
    match, not stretched-spelling aware like intent_agent's own greeting
    regex - this is a narrow, reusable check for other agents (e.g. lead
    collection) to recognize an obvious greeting isn't a real field value,
    not a replacement for intent_agent's fuller greeting detection.
    """
    stripped = _TRAILING_PUNCTUATION_RE.sub("", text.strip().lower())
    return stripped in GREETING_WORDS
