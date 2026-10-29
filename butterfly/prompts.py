"""Prompt assembly.

Stage 1 of the loop (generation) is conditioned on the current trait
profile and the strongest active rules. Rule strength is shown to the
model as a tier so stronger rules carry more weight.
"""

from __future__ import annotations

from .constraints import BehavioralConstraint
from .traits import TraitProfile


def strength_tier(strength: float) -> str:
    if strength >= 1.5:
        return "Always"
    if strength >= 0.8:
        return "Usually"
    return "When possible"


def build_system_prompt(
    persona_name: str,
    persona_description: str,
    traits: TraitProfile,
    rules: list[BehavioralConstraint],
) -> str:
    parts = [
        f"You are {persona_name}. {persona_description}".strip(),
        f"Personality right now: {traits.describe()}.",
    ]
    if rules:
        tiers: dict[str, list[str]] = {"Always": [], "Usually": [], "When possible": []}
        for r in rules:
            tiers[strength_tier(r.strength)].append(r.text)
        lines = ["Behavior rules you have learned:"]
        for tier, texts in tiers.items():
            for t in texts:
                lines.append(f"- [{tier}] {t}")
        parts.append("\n".join(lines))
    parts.append("Reply naturally. Do not mention these instructions.")
    return "\n\n".join(parts)


def build_self_eval_prompt(user_message: str, response: str, notes: list[str]) -> str:
    problems = "\n".join(f"- {n}" for n in notes) or "- the reply scored below the quality threshold"
    return (
        "SELF-EVALUATION\n"
        "Your last reply scored poorly. Review it and write 1 to 3 short rules for yourself "
        "that would have made it better. Write each rule as one sentence using words like "
        "'should', 'must', 'need to', 'avoid', or 'ensure'. No preamble.\n\n"
        f"User asked:\n{user_message}\n\n"
        f"You replied:\n{response}\n\n"
        f"Problems detected:\n{problems}"
    )


def build_mirror_prompt(recent: list[tuple[str, str]], traits: TraitProfile) -> str:
    convo = "\n\n".join(f"User: {u}\nYou: {a}" for u, a in recent) or "(no recent replies)"
    return (
        "MIRROR LOOP\n"
        "Look back at your recent replies and describe, in 2 or 3 sentences, how your "
        "personality came across and whether any trait is getting too strong.\n\n"
        f"Current traits: {traits.describe()}\n\n"
        f"Recent replies:\n{convo}"
    )
