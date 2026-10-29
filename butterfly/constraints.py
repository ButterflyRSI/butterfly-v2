"""Method 1: Positive Reinforcement Constraint System.

Rules are objects with a numeric strength instead of on/off switches.
They gain strength when active during a good response, lose strength
during a bad one, and are pruned once they fall to the floor.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from .config import ButterflyConfig

_WORD = re.compile(r"[a-z0-9']+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    return " ".join(_WORD.findall(text.lower()))


def token_set(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def similarity(a: str, b: str) -> float:
    """Jaccard overlap of word sets, used to catch near-duplicate rules."""
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class BehavioralConstraint:
    text: str
    strength: float = 1.0
    successes: int = 0
    failures: int = 0
    created_turn: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def strengthen(self, amount: float = 0.1, cap: float = 2.0) -> None:
        self.strength = min(cap, self.strength + amount)
        self.successes += 1

    def weaken(self, amount: float = 0.05, floor: float = 0.1) -> None:
        self.strength = max(floor, self.strength - amount)
        self.failures += 1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "strength": round(self.strength, 4),
            "successes": self.successes,
            "failures": self.failures,
            "created_turn": self.created_turn,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BehavioralConstraint":
        return cls(
            text=data["text"],
            strength=float(data.get("strength", 1.0)),
            successes=int(data.get("successes", 0)),
            failures=int(data.get("failures", 0)),
            created_turn=int(data.get("created_turn", 0)),
            id=data.get("id") or uuid.uuid4().hex[:8],
        )


def prune_constraints(
    constraints: list[BehavioralConstraint], floor: float = 0.2
) -> list[BehavioralConstraint]:
    """Paper Section 3.2: keep only rules strictly above the floor.

    A small epsilon absorbs float drift, so a rule weakened to exactly
    the floor (e.g. 1.0 minus 16 x 0.05) is pruned as the paper intends.
    """
    return [c for c in constraints if c.strength > floor + 1e-9]


class ConstraintSet:
    """The active rule set plus the bookkeeping around it."""

    def __init__(self, config: ButterflyConfig | None = None):
        self.config = config or ButterflyConfig()
        self.items: list[BehavioralConstraint] = []

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def is_duplicate(self, text: str) -> bool:
        """Checks against STORED rules, not just the incoming batch."""
        norm = normalize(text)
        for c in self.items:
            if normalize(c.text) == norm:
                return True
            if similarity(c.text, text) >= self.config.dedupe_similarity:
                return True
        return False

    def add(self, text: str, turn: int = 0) -> BehavioralConstraint | None:
        text = text.strip()
        if not text or self.is_duplicate(text):
            return None
        rule = BehavioralConstraint(
            text=text, strength=self.config.initial_strength, created_turn=turn
        )
        self.items.append(rule)
        return rule

    def strengthen(self, rules: list[BehavioralConstraint] | None = None) -> None:
        for c in rules if rules is not None else self.items:
            c.strengthen(self.config.strengthen_amount, self.config.max_strength)

    def weaken(self, rules: list[BehavioralConstraint] | None = None) -> None:
        for c in rules if rules is not None else self.items:
            c.weaken(self.config.weaken_amount, self.config.min_strength)

    def prune(self) -> list[BehavioralConstraint]:
        """Remove rules at the floor, then enforce max_constraints if set.

        Returns the rules that were removed.
        """
        kept = prune_constraints(self.items, self.config.prune_floor)
        removed = [c for c in self.items if c not in kept]
        cap = self.config.max_constraints
        if cap is not None and len(kept) > cap:
            # Drop the weakest; on ties drop the newest so established rules survive.
            ordered = sorted(kept, key=lambda c: (c.strength, -c.created_turn), reverse=True)
            survivors = ordered[:cap]
            removed += [c for c in kept if c not in survivors]
            kept = [c for c in kept if c in survivors]
        self.items = kept
        return removed

    def strongest(self, n: int | None = None) -> list[BehavioralConstraint]:
        ordered = sorted(self.items, key=lambda c: c.strength, reverse=True)
        return ordered if n is None else ordered[:n]

    def to_list(self) -> list[dict]:
        return [c.to_dict() for c in self.items]

    def load(self, data: list[dict]) -> None:
        self.items = [BehavioralConstraint.from_dict(d) for d in data]
