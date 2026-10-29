"""Section 3.3: the shared quality signal.

Four binary signals are averaged into one score in [0, 1]. That single
score drives both Method 1 (reinforce vs. weaken) and Method 3 (the
extraction gate).

The paper treats the four signals as a reference design. The heuristics
below are simple and transparent on purpose; swap in your own by
subclassing ResponseQualityEvaluator, or set evaluator_mode="llm" to have
the model grade its own response.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .constraints import BehavioralConstraint

SIGNAL_NAMES = (
    "asked_questions",
    "appropriate_length",
    "followed_constraints",
    "shows_self_awareness",
)

_VAGUE_WORDS = {
    "it", "this", "that", "thing", "stuff", "something", "them", "those",
    "these", "help", "fix", "issue", "problem",
}
_HEDGES = (
    "i'm not sure", "i am not sure", "not certain", "i think", "i believe",
    "might", "may ", "could be", "depends", "it depends", "likely",
    "probably", "uncertain", "i don't know", "i do not know", "can't be sure",
    "cannot be sure", "as far as i know", "limitation", "i can't", "i cannot",
    "hard to say", "roughly", "approximately", "not guaranteed",
)
_SPECULATIVE = (
    "will ", "should i", "predict", "future", "best ", "which is better",
    "what do you think", "opinion", "recommend", "is it safe", "diagnos",
    "invest", "legal", "medical", "how long", "how much", "guarantee",
    "why ",
)
_NEGATION_RULE = re.compile(
    r"\b(?:avoid|never|don't|do not|stop)\s+(?:using\s+|saying\s+|the\s+word\s+)?['\"]?([a-z0-9' ]{3,40}?)['\"]?(?:[.,;]|$)"
)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def is_ambiguous(user_message: str) -> bool:
    """Short or vague requests that a good answer should clarify first."""
    words = [w.lower() for w in _words(user_message)]
    if not words:
        return True
    if len(words) <= 2:
        greetings = {"hi", "hey", "hello", "yo", "thanks", "thank", "you", "sup"}
        return not set(words) <= greetings
    if len(words) <= 6:
        vague = sum(1 for w in words if w in _VAGUE_WORDS)
        return vague >= 2
    return False


def needs_humility(user_message: str) -> bool:
    lower = user_message.lower() + " "
    return any(marker in lower for marker in _SPECULATIVE)


def length_bounds(user_message: str) -> tuple[int, int]:
    """Acceptable response length in words for a given request."""
    n = len(_words(user_message))
    low = 2 if n <= 3 else 5
    high = max(60, n * 25)
    lower = user_message.lower()
    if any(k in lower for k in ("explain", "detail", "step by step", "write", "essay", "list", "compare")):
        high = max(high, 600)
    return low, min(high, 900)


def violated_rules(response: str, constraints: list[BehavioralConstraint]) -> list[str]:
    """Rules of the form 'avoid X' / 'never X' that the response broke."""
    lower = response.lower()
    broken = []
    for c in constraints:
        for match in _NEGATION_RULE.finditer(c.text.lower()):
            phrase = match.group(1).strip()
            if len(phrase) >= 3 and phrase in lower:
                broken.append(c.text)
                break
    return broken


@dataclass
class QualitySignals:
    asked_questions: bool
    appropriate_length: bool
    followed_constraints: bool
    shows_self_awareness: bool
    quality_score: float = 0.0
    deserves_reinforcement: bool = False
    # Extra, unscored observations used by the mirror loop's trait nudges.
    contains_question: bool = False
    word_count: int = 0
    length_ratio: float = 0.0
    hedged: bool = False
    source: str = "heuristic"
    notes: list[str] = field(default_factory=list)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        return {
            "asked_questions": self.asked_questions,
            "appropriate_length": self.appropriate_length,
            "followed_constraints": self.followed_constraints,
            "shows_self_awareness": self.shows_self_awareness,
            "quality_score": round(self.quality_score, 4),
            "deserves_reinforcement": self.deserves_reinforcement,
            "contains_question": self.contains_question,
            "word_count": self.word_count,
            "length_ratio": round(self.length_ratio, 3),
            "hedged": self.hedged,
            "source": self.source,
            "notes": list(self.notes),
        }


def finalize(signals: QualitySignals, threshold: float) -> QualitySignals:
    """Paper Section 5.2: score = mean of the four signals; gate at threshold."""
    values = [getattr(signals, name) for name in SIGNAL_NAMES]
    signals.quality_score = sum(bool(v) for v in values) / len(values)
    signals.deserves_reinforcement = signals.quality_score >= threshold
    return signals


class ResponseQualityEvaluator:
    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold

    def evaluate_quality(
        self,
        response: str,
        user_message: str,
        constraints: list[BehavioralConstraint] | None = None,
    ) -> QualitySignals:
        constraints = constraints or []
        notes: list[str] = []
        lower = response.lower()
        word_count = len(_words(response))
        contains_question = "?" in response
        hedged = any(h in lower for h in _HEDGES)

        # asked_questions: clarification when ambiguous. A clear request
        # doesn't need a question, so it passes unless the reply is only a
        # question and gives nothing else.
        if is_ambiguous(user_message):
            asked = contains_question
            if not asked:
                notes.append("ambiguous request but no clarifying question")
        else:
            stripped = response.strip()
            only_question = stripped.endswith("?") and word_count <= 12 and stripped.count(".") == 0
            asked = not only_question
            if only_question:
                notes.append("clear request answered with only a question")

        low, high = length_bounds(user_message)
        appropriate = low <= word_count <= high
        if not appropriate:
            notes.append(f"length {word_count} words outside {low}-{high}")
        ratio = word_count / high if high else 0.0

        broken = violated_rules(response, constraints)
        followed = not broken
        if broken:
            notes.append(f"broke rule: {broken[0]}")

        if needs_humility(user_message):
            aware = hedged
            if not aware:
                notes.append("speculative question answered with no hedging")
        else:
            aware = True

        signals = QualitySignals(
            asked_questions=asked,
            appropriate_length=appropriate,
            followed_constraints=followed,
            shows_self_awareness=aware,
            contains_question=contains_question,
            word_count=word_count,
            length_ratio=ratio,
            hedged=hedged,
            source="heuristic",
            notes=notes,
        )
        return finalize(signals, self.threshold)


JUDGE_PROMPT = """You are grading an AI assistant's reply. Answer with JSON only, no other text.

User message:
<<<{user}>>>

Active rules the assistant was given:
{rules}

Assistant reply:
<<<{reply}>>>

Return exactly this JSON object with true or false for each key:
{{"asked_questions": <true if the reply asked for clarification when the request was ambiguous, or the request was clear enough not to need it>,
 "appropriate_length": <true if the length fits the request>,
 "followed_constraints": <true if the reply is consistent with the active rules>,
 "shows_self_awareness": <true if it acknowledges uncertainty or limits where relevant, or none was needed>}}"""


def parse_judge_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not all(name in data for name in SIGNAL_NAMES):
        return None
    return {name: bool(data[name]) for name in SIGNAL_NAMES}


class LLMJudgeEvaluator(ResponseQualityEvaluator):
    """Asks the model to grade its own reply; falls back to heuristics."""

    def __init__(self, llm, threshold: float = 0.65):
        super().__init__(threshold)
        self.llm = llm

    def evaluate_quality(self, response, user_message, constraints=None) -> QualitySignals:
        base = super().evaluate_quality(response, user_message, constraints)
        rules = "\n".join(f"- {c.text}" for c in (constraints or [])) or "(none)"
        prompt = JUDGE_PROMPT.format(user=user_message, rules=rules, reply=response)
        try:
            raw = self.llm.chat([{"role": "user", "content": prompt}], system="You grade replies and output JSON only.")
        except Exception as exc:  # network or model failure: keep heuristics
            base.notes.append(f"judge unavailable ({type(exc).__name__}); used heuristics")
            return base
        parsed = parse_judge_json(raw)
        if parsed is None:
            base.notes.append("judge reply was not valid JSON; used heuristics")
            return base
        for name, value in parsed.items():
            setattr(base, name, value)
        base.source = "llm"
        return finalize(base, self.threshold)
