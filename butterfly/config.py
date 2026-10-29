"""Reference parameters for the Butterfly Framework.

Every default here comes straight from the white paper (Section 11.2).
Two extra controls, ``max_constraints`` and ``baseline_pull``, are
opt-in safety bounds and are OFF by default so the paper's behavior is
reproduced exactly unless you turn them on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class ButterflyConfig:
    # --- Method 1: Positive Reinforcement Constraint System ---
    strengthen_amount: float = 0.1      # +0.1 per reinforced turn
    weaken_amount: float = 0.05         # -0.05 per failed turn (deliberate 2:1 asymmetry)
    max_strength: float = 2.0           # cap so no single rule dominates
    min_strength: float = 0.1           # floor applied by weaken()
    prune_floor: float = 0.2            # rules at or below this are removed
    initial_strength: float = 1.0       # strength of a newly extracted rule

    # --- Method 3: Quality-Gated Constraint Extraction ---
    quality_threshold: float = 0.65     # score >= this suppresses extraction
    trigger_words: tuple[str, ...] = ("should", "must", "need to", "avoid", "ensure")
    dedupe_similarity: float = 0.8      # Jaccard overlap treated as a duplicate rule

    # --- Method 2: Mirror Loop with Personality Drift Correction ---
    mirror_loop_interval: int = 3       # fire every N messages
    drift_correction_rate: float = 0.15 # pull 15% toward the group mean
    trait_nudge: float = 0.05           # size of a single self-evaluation nudge

    # --- Prompt assembly ---
    history_limit: int = 20             # messages of history sent to the model
    prompt_rule_limit: int = 12         # strongest N rules placed in the system prompt

    # --- Optional homeostasis bounds (off by default) ---
    max_constraints: int | None = None  # cap on rule set size; weakest dropped first
    baseline_pull: float = 0.0          # extra pull of each trait toward its own baseline

    # --- Quality evaluator ---
    evaluator_mode: str = "heuristic"   # "heuristic" or "llm" (model grades itself)

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["trigger_words"] = list(self.trigger_words)
        return data

    @classmethod
    def from_dict(cls, data: dict | None) -> "ButterflyConfig":
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in known}
        if "trigger_words" in clean:
            clean["trigger_words"] = tuple(clean["trigger_words"])
        return cls(**clean)

    def validate(self) -> None:
        if not 0 < self.quality_threshold <= 1:
            raise ValueError("quality_threshold must be in (0, 1]")
        if self.mirror_loop_interval < 1:
            raise ValueError("mirror_loop_interval must be >= 1")
        if not 0 <= self.drift_correction_rate <= 1:
            raise ValueError("drift_correction_rate must be in [0, 1]")
        if not 0 <= self.baseline_pull <= 1:
            raise ValueError("baseline_pull must be in [0, 1]")
        if self.min_strength > self.prune_floor:
            raise ValueError("min_strength must not exceed prune_floor")
        if self.max_constraints is not None and self.max_constraints < 1:
            raise ValueError("max_constraints must be >= 1 when set")
        if self.evaluator_mode not in ("heuristic", "llm"):
            raise ValueError("evaluator_mode must be 'heuristic' or 'llm'")
