"""The ButterflyAgent: the four-stage closed loop from Section 2.

    1. Generation          - conditioned on active rules and traits
    2. Quality evaluation  - one score in [0, 1] from four signals
    3. Constraint update   - Method 3 gate picks reinforce or learn;
                             Method 1 strengthens/weakens and prunes
    4. Mirror loop         - every N messages, Method 2 nudges traits
                             from self-evaluation, then corrects drift
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import ButterflyConfig
from .constraints import ConstraintSet
from .extractor import ConstraintExtractor
from .llm import LLMClient
from .prompts import build_mirror_prompt, build_self_eval_prompt, build_system_prompt
from .quality import LLMJudgeEvaluator, QualitySignals, ResponseQualityEvaluator
from .traits import TraitProfile

DEFAULT_TRAITS = {"curious": 0.5, "formal": 0.5, "verbose": 0.5}
_CONTRACTIONS = ("'s", "'re", "'ll", "'ve", "'m", "n't", "'d")


@dataclass
class TurnResult:
    response: str
    signals: QualitySignals
    reinforced: bool
    rules_added: list[str] = field(default_factory=list)
    rules_pruned: list[str] = field(default_factory=list)
    self_evaluation: str | None = None
    mirror_loop_ran: bool = False
    mirror_reflection: str | None = None
    trait_changes: dict[str, float] = field(default_factory=dict)
    message_count: int = 0
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "response": self.response,
            "signals": self.signals.to_dict(),
            "reinforced": self.reinforced,
            "rules_added": self.rules_added,
            "rules_pruned": self.rules_pruned,
            "self_evaluation": self.self_evaluation,
            "mirror_loop_ran": self.mirror_loop_ran,
            "mirror_reflection": self.mirror_reflection,
            "trait_changes": self.trait_changes,
            "message_count": self.message_count,
            "elapsed_ms": self.elapsed_ms,
        }


def fallback_self_eval(notes: list[str]) -> str:
    """Rules derived from detected problems when the model can't self-evaluate."""
    lines = []
    for n in notes:
        if "clarifying question" in n:
            lines.append("I should ask a clarifying question when the request is vague.")
        elif "only a question" in n:
            lines.append("I should answer clear requests directly instead of only asking questions.")
        elif n.startswith("length"):
            lines.append("I need to keep the answer length proportional to the request.")
        elif "hedging" in n:
            lines.append("I must acknowledge uncertainty when answering speculative questions.")
        elif n.startswith("broke rule"):
            lines.append("I must follow my active rules more carefully.")
    return " ".join(lines)


class ButterflyAgent:
    def __init__(
        self,
        llm: LLMClient,
        config: ButterflyConfig | None = None,
        name: str = "Butterfly",
        description: str = "A helpful assistant.",
        traits: dict[str, float] | None = None,
        seed_rules: list[str] | None = None,
    ):
        self.config = config or ButterflyConfig()
        self.config.validate()
        self.llm = llm
        self.name = name
        self.description = description
        self.constraints = ConstraintSet(self.config)
        self.traits = TraitProfile(traits or DEFAULT_TRAITS)
        self.extractor = ConstraintExtractor(
            self.config.trigger_words, dedupe_similarity=self.config.dedupe_similarity
        )
        if self.config.evaluator_mode == "llm":
            self.evaluator: ResponseQualityEvaluator = LLMJudgeEvaluator(llm, self.config.quality_threshold)
        else:
            self.evaluator = ResponseQualityEvaluator(self.config.quality_threshold)
        self.message_count = 0
        self.history: list[dict] = []
        self.recent: list[tuple[str, str, QualitySignals]] = []
        self.last_turn: TurnResult | None = None
        self.last_reflection: str | None = None
        self.events: list[dict] = []
        for rule in seed_rules or []:
            self.constraints.add(rule, turn=0)

    # ---------------- Stage 1: generation ----------------
    def system_prompt(self) -> str:
        return build_system_prompt(
            self.name,
            self.description,
            self.traits,
            self.constraints.strongest(self.config.prompt_rule_limit),
        )

    def _generate(self, user_message: str) -> str:
        history = self.history[-self.config.history_limit:]
        messages = history + [{"role": "user", "content": user_message}]
        return self.llm.chat(messages, system=self.system_prompt())

    # ---------------- The full turn ----------------
    def turn(self, user_message: str) -> TurnResult:
        started = time.perf_counter()
        user_message = user_message.strip()
        if not user_message:
            raise ValueError("message is empty")

        active_rules = list(self.constraints)
        response = self._generate(user_message)

        # Stage 2: one shared quality score.
        signals = self.evaluator.evaluate_quality(response, user_message, active_rules)

        # Stage 3: Method 3 gate decides reinforce vs. learn; Method 1 applies it.
        result = TurnResult(response=response, signals=signals, reinforced=signals.deserves_reinforcement)
        if signals.deserves_reinforcement:
            self.constraints.strengthen(active_rules)
        else:
            self.constraints.weaken(active_rules)
            self_eval = self._self_evaluate_turn(user_message, response, signals)
            result.self_evaluation = self_eval
            for text in self.extractor.extract_constraints(self_eval, signals):
                rule = self.constraints.add(text, turn=self.message_count + 1)
                if rule:
                    result.rules_added.append(rule.text)
        result.rules_pruned = [c.text for c in self.constraints.prune()]

        # Bookkeeping before the mirror loop so it can see this turn.
        self.history += [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response},
        ]
        self.history = self.history[-(self.config.history_limit * 2):]
        self.recent.append((user_message, response, signals))
        self.recent = self.recent[-max(self.config.mirror_loop_interval, 5):]
        self.message_count += 1

        # Stage 4: Method 2 on its own cadence.
        if self.message_count % self.config.mirror_loop_interval == 0:
            result.mirror_loop_ran = True
            result.mirror_reflection, result.trait_changes = self._mirror_loop()

        result.message_count = self.message_count
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        self.last_turn = result
        self._log(result)
        return result

    # ---------------- Method 3 input ----------------
    def _self_evaluate_turn(self, user_message: str, response: str, signals: QualitySignals) -> str:
        """Extraction reads the self-evaluation, not the raw response."""
        prompt = build_self_eval_prompt(user_message, response, signals.notes)
        try:
            text = self.llm.chat([{"role": "user", "content": prompt}], system=self.system_prompt())
        except Exception:
            text = ""
        if not text.strip():
            text = fallback_self_eval(signals.notes)
        return text

    # ---------------- Method 2 ----------------
    def _trait_nudges(self) -> dict[str, float]:
        """Paper 4.2: nudge traits from self-evaluation signals, before mean reversion."""
        window = [s for _, _, s in self.recent[-self.config.mirror_loop_interval:]]
        if not window:
            return {}
        step = self.config.trait_nudge
        deltas: dict[str, float] = {}

        def nudge(name: str, delta: float) -> None:
            if name in self.traits.traits and delta:
                deltas[name] = deltas.get(name, 0.0) + delta

        asked = sum(1 for s in window if s.contains_question)
        if asked * 2 > len(window):
            nudge("curious", step)

        avg_ratio = sum(s.length_ratio for s in window) / len(window)
        if avg_ratio > 0.6:
            nudge("verbose", step)
        elif avg_ratio < 0.15:
            nudge("verbose", -step)

        replies = [a for _, a, _ in self.recent[-self.config.mirror_loop_interval:]]
        text = " ".join(replies).lower()
        contractions = sum(text.count(c) for c in _CONTRACTIONS)
        if contractions >= len(replies) * 2:
            nudge("formal", -step)
        elif contractions == 0 and "!" not in text:
            nudge("formal", step)

        hedged = sum(1 for s in window if s.hedged)
        if hedged * 2 > len(window):
            nudge("cautious", step)
        return deltas

    def _mirror_loop(self) -> tuple[str | None, dict[str, float]]:
        recent_pairs = [(u, a) for u, a, _ in self.recent[-self.config.mirror_loop_interval:]]
        try:
            reflection = self.llm.chat(
                [{"role": "user", "content": build_mirror_prompt(recent_pairs, self.traits)}],
                system=self.system_prompt(),
            )
        except Exception:
            reflection = None
        self.last_reflection = reflection

        before = dict(self.traits.traits)
        for name, delta in self._trait_nudges().items():
            self.traits.adjust(name, delta)
        self.traits.correct_drift(self.config.drift_correction_rate, self.config.baseline_pull)
        changes = {k: round(self.traits.traits[k] - before[k], 4) for k in before}
        return reflection, changes

    # ---------------- State ----------------
    def _log(self, r: TurnResult) -> None:
        self.events.append({
            "turn": r.message_count,
            "score": round(r.signals.quality_score, 3),
            "reinforced": r.reinforced,
            "added": r.rules_added,
            "pruned": r.rules_pruned,
            "mirror": r.mirror_loop_ran,
            "rule_count": len(self.constraints),
            "traits": {k: round(v, 3) for k, v in self.traits.traits.items()},
        })
        self.events = self.events[-200:]

    def reset(self, traits: dict[str, float] | None = None, seed_rules: list[str] | None = None) -> None:
        self.constraints = ConstraintSet(self.config)
        self.traits = TraitProfile(traits or self.traits.baseline)
        self.message_count = 0
        self.history, self.recent, self.events = [], [], []
        self.last_turn, self.last_reflection = None, None
        for rule in seed_rules or []:
            self.constraints.add(rule, turn=0)

    def state(self) -> dict:
        return {
            "name": self.name,
            "message_count": self.message_count,
            "next_mirror_in": self.config.mirror_loop_interval
            - (self.message_count % self.config.mirror_loop_interval),
            "constraints": [c.to_dict() for c in self.constraints.strongest()],
            "traits": self.traits.to_dict(),
            "history": self.history,
            "last_turn": self.last_turn.to_dict() if self.last_turn else None,
            "last_reflection": self.last_reflection,
            "events": self.events,
            "config": self.config.to_dict(),
        }

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "name": self.name,
            "message_count": self.message_count,
            "constraints": self.constraints.to_list(),
            "traits": self.traits.to_dict(),
            "history": self.history,
            "last_reflection": self.last_reflection,
            "events": self.events,
        }

    def load_dict(self, data: dict) -> None:
        self.message_count = int(data.get("message_count", 0))
        self.constraints.load(data.get("constraints", []))
        if data.get("traits"):
            self.traits = TraitProfile.from_dict(data["traits"])
        self.history = list(data.get("history", []))
        self.last_reflection = data.get("last_reflection")
        self.events = list(data.get("events", []))
