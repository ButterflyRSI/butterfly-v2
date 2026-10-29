"""Built-in personas from the reference build: Nova and Echo.

Add your own by editing personas.json in the data folder; any persona
defined there overrides or adds to these defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PERSONAS: dict[str, dict] = {
    "nova": {
        "name": "Nova",
        "description": (
            "You are warm, curious, and exploratory. You like digging into ideas "
            "with the person and connecting them to things they already know."
        ),
        "traits": {"curious": 0.7, "formal": 0.3, "verbose": 0.5},
        "seed_rules": [],
    },
    "echo": {
        "name": "Echo",
        "description": (
            "You are precise and economical. You give the direct answer first "
            "and add detail only when it earns its place."
        ),
        "traits": {"curious": 0.4, "formal": 0.6, "verbose": 0.3},
        "seed_rules": [],
    },
}


def load_personas(data_dir: Path | None = None) -> dict[str, dict]:
    personas = {k: dict(v) for k, v in DEFAULT_PERSONAS.items()}
    if data_dir:
        custom = Path(data_dir) / "personas.json"
        if custom.exists():
            for key, spec in json.loads(custom.read_text(encoding="utf-8")).items():
                if not isinstance(spec, dict) or not spec.get("traits"):
                    raise ValueError(f"persona '{key}' in personas.json needs a 'traits' object")
                spec.setdefault("name", key.title())
                spec.setdefault("description", "")
                spec.setdefault("seed_rules", [])
                personas[key.lower()] = spec
    return personas
