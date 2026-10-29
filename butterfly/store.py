"""Saves each persona's rules, traits, and history to JSON so they survive restarts."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

_SAFE = re.compile(r"^[a-z0-9_-]{1,40}$")


class StateStore:
    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if not _SAFE.match(key):
            raise ValueError(f"invalid persona key: {key!r}")
        return self.dir / f"{key}.json"

    def load(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Keep a copy of the damaged file instead of silently losing it.
            path.replace(path.with_suffix(".corrupt.json"))
            return None

    def save(self, key: str, data: dict) -> None:
        path = self._path(key)
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)  # atomic, so a crash never leaves half a file

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()
