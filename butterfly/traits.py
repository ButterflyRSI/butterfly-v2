"""Method 2 state: the personality trait profile.

Traits are values in [0, 1]. The mirror loop nudges them based on
self-evaluation, then pulls every trait part of the way toward the
group mean so no single trait runs away while relative differences
between traits are preserved.
"""

from __future__ import annotations


class UnknownTraitError(KeyError):
    """Raised instead of silently creating a trait from a typo."""


class TraitProfile:
    def __init__(self, traits: dict[str, float], baseline: dict[str, float] | None = None):
        if not traits:
            raise ValueError("a trait profile needs at least one trait")
        self.traits = {k: self._clamp(v) for k, v in traits.items()}
        # Baseline is the persona's configured starting point. It is only
        # used when the optional baseline_pull control is turned on.
        self.baseline = dict(baseline) if baseline else dict(self.traits)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def get(self, name: str) -> float:
        if name not in self.traits:
            raise UnknownTraitError(name)
        return self.traits[name]

    def adjust(self, name: str, delta: float) -> float:
        if name not in self.traits:
            raise UnknownTraitError(
                f"unknown trait '{name}'; known traits: {sorted(self.traits)}"
            )
        self.traits[name] = self._clamp(self.traits[name] + delta)
        return self.traits[name]

    def mean(self) -> float:
        return sum(self.traits.values()) / len(self.traits)

    def correct_drift(self, rate: float = 0.15, baseline_pull: float = 0.0) -> dict[str, float]:
        """Pull each trait toward the group mean by ``rate``.

        With baseline_pull > 0, each trait is additionally pulled toward its
        own baseline. That gives the profile a natural way home after a long
        one-sided run of nudges, which mean-reversion alone cannot do (if every
        trait is pinned at 1.0, the mean is also 1.0).

        Returns the per-trait change applied.
        """
        group_mean = self.mean()
        changes = {}
        for name, value in self.traits.items():
            new = value + (group_mean - value) * rate
            if baseline_pull:
                new += (self.baseline.get(name, new) - new) * baseline_pull
            new = self._clamp(new)
            changes[name] = round(new - value, 6)
            self.traits[name] = new
        return changes

    def describe(self) -> str:
        """Plain-language description used in the system prompt."""
        def level(v: float) -> str:
            if v >= 0.75:
                return "very"
            if v >= 0.55:
                return "fairly"
            if v >= 0.35:
                return "moderately"
            if v >= 0.15:
                return "not very"
            return "not at all"

        return ", ".join(f"{level(v)} {name}" for name, v in self.traits.items())

    def to_dict(self) -> dict:
        return {
            "traits": {k: round(v, 4) for k, v in self.traits.items()},
            "baseline": {k: round(v, 4) for k, v in self.baseline.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TraitProfile":
        return cls(data["traits"], data.get("baseline"))
