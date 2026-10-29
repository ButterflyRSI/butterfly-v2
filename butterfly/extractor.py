"""Method 3: Quality-Gated Constraint Extraction.

The inversion: a good response (score >= threshold) extracts NOTHING and
routes to reinforcement instead. Only a poor response triggers learning,
and the learning reads the agent's self-evaluation of that response,
not the response itself.
"""

from __future__ import annotations

import re

from .constraints import normalize, similarity

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_LEADING_JUNK = re.compile(r"^\s*(?:[-*\u2022>]+|\d+[.)]|rule\s*\d*\s*:)\s*", re.IGNORECASE)
_SUBJECT_PREFIX = re.compile(
    r"^(?:in (?:the )?future,?\s*|next time,?\s*|going forward,?\s*)?"
    r"(?:i|you|we|the (?:assistant|response|reply|answer|agent)|my (?:response|reply|answer))\s+",
    re.IGNORECASE,
)


class ConstraintExtractor:
    def __init__(
        self,
        trigger_words: tuple[str, ...] = ("should", "must", "need to", "avoid", "ensure"),
        min_words: int = 3,
        max_words: int = 30,
        dedupe_similarity: float = 0.8,
    ):
        self.trigger_words = tuple(w.lower() for w in trigger_words)
        self.min_words = min_words
        self.max_words = max_words
        self.dedupe_similarity = dedupe_similarity
        self._patterns = [
            re.compile(r"\b" + re.escape(w).replace(r"\ ", r"\s+") + r"\b", re.IGNORECASE)
            for w in self.trigger_words
        ]

    def extract_constraints(self, self_eval_text: str, quality_signals) -> list[str]:
        if quality_signals.get("deserves_reinforcement"):
            return []  # Quality gate: skip extraction on good performance
        candidates = self._find_candidate_sentences(self_eval_text or "")
        return self._dedupe(candidates)

    def _has_trigger(self, sentence: str) -> bool:
        return any(p.search(sentence) for p in self._patterns)

    def _clean(self, sentence: str) -> str:
        s = _LEADING_JUNK.sub("", sentence.strip())
        s = s.strip().strip('"').strip()
        s = _SUBJECT_PREFIX.sub("", s)
        if s:
            s = s[0].upper() + s[1:]
            if s[-1] not in ".!":
                s += "."
        return s

    def _is_valid_constraint(self, sentence: str) -> bool:
        words = sentence.split()
        if not self.min_words <= len(words) <= self.max_words:
            return False
        if sentence.rstrip().endswith("?"):
            return False
        if not self._has_trigger(sentence):
            return False
        lower = sentence.lower()
        # Self-praise like "I should be proud" or meta noise isn't a behavior rule.
        if any(p in lower for p in ("as an ai", "language model", "overall the response")):
            return False
        return True

    def _find_candidate_sentences(self, text: str) -> list[str]:
        out = []
        for raw in _SENTENCE_SPLIT.split(text):
            if not raw.strip() or raw.strip().endswith("?"):
                continue
            cleaned = self._clean(raw)
            if cleaned and self._is_valid_constraint(cleaned):
                out.append(cleaned)
        return out

    def _dedupe(self, candidates: list[str]) -> list[str]:
        kept: list[str] = []
        for c in candidates:
            if any(
                normalize(c) == normalize(k) or similarity(c, k) >= self.dedupe_similarity
                for k in kept
            ):
                continue
            kept.append(c)
        return kept
