"""FastAPI server (Section 10.8), extended with per-persona state,
persistence, health checks, and the built React UI.

Run:  python -m butterfly          (see README for options)
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import ButterflyAgent
from .config import ButterflyConfig
from .llm import LLMClient, LLMError, client_from_env
from .personas import load_personas
from .store import StateStore
from .tts import KokoroTTS, TTSUnavailable

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "frontend" / "dist"


class ChatRequest(BaseModel):
    persona: str = "nova"
    message: str = Field(min_length=1, max_length=8000)


class TTSRequest(BaseModel):
    persona: str = "nova"
    text: str = Field(min_length=1, max_length=4000)
    voice: str | None = None
    speed: float | None = Field(default=None, ge=0.5, le=2.0)


class AgentPool:
    """One agent per persona, each with its own lock and saved state."""

    def __init__(self, llm: LLMClient, store: StateStore, personas: dict, config: ButterflyConfig):
        self.llm = llm
        self.store = store
        self.personas = personas
        self.config = config
        self.agents: dict[str, ButterflyAgent] = {}
        self.locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def get(self, key: str) -> tuple[ButterflyAgent, threading.Lock]:
        key = key.lower()
        if key not in self.personas:
            raise HTTPException(404, f"Unknown persona '{key}'. Choose one of: {', '.join(self.personas)}")
        with self._guard:
            if key not in self.agents:
                spec = self.personas[key]
                agent = ButterflyAgent(
                    self.llm,
                    ButterflyConfig.from_dict(self.config.to_dict()),
                    name=spec["name"],
                    description=spec["description"],
                    traits=spec["traits"],
                    seed_rules=spec.get("seed_rules"),
                )
                saved = self.store.load(key)
                if saved:
                    agent.load_dict(saved)
                self.agents[key] = agent
                self.locks[key] = threading.Lock()
            return self.agents[key], self.locks[key]


def create_app(
    llm: LLMClient | None = None,
    data_dir: str | Path | None = None,
    config: ButterflyConfig | None = None,
    tts=None,
) -> FastAPI:
    data_dir = Path(data_dir or os.getenv("BUTTERFLY_DATA", ROOT / "data"))
    llm = llm or client_from_env()
    config = config or ButterflyConfig()
    config.validate()
    pool = AgentPool(llm, StateStore(data_dir), load_personas(data_dir), config)
    tts = tts or KokoroTTS()

    app = FastAPI(title="Butterfly", version="2.0.0")
    app.state.pool = pool
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return pool.llm.health()

    @app.get("/api/personas")
    def personas():
        return [
            {"key": k, "name": v["name"], "description": v["description"], "voice": v.get("voice")}
            for k, v in pool.personas.items()
        ]

    @app.get("/api/state/{persona}")
    def state(persona: str):
        agent, lock = pool.get(persona)
        with lock:
            return agent.state()

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        agent, lock = pool.get(req.persona)
        with lock:
            try:
                result = agent.turn(req.message)
            except LLMError as exc:
                raise HTTPException(502, str(exc)) from exc
            pool.store.save(req.persona.lower(), agent.to_dict())
            return {"turn": result.to_dict(), "state": agent.state()}

    @app.post("/api/reset/{persona}")
    def reset(persona: str):
        agent, lock = pool.get(persona)
        spec = pool.personas[persona.lower()]
        with lock:
            agent.reset(traits=spec["traits"], seed_rules=spec.get("seed_rules"))
            pool.store.delete(persona.lower())
            return agent.state()

    @app.get("/api/tts")
    def tts_status():
        return tts.status()

    @app.post("/api/tts")
    def speak(req: TTSRequest):
        key = req.persona.lower()
        if key not in pool.personas:
            raise HTTPException(404, f"Unknown persona '{key}'")
        default = pool.personas[key].get("voice") or {}
        voice = req.voice or default.get("name", "af_heart")
        speed = req.speed if req.speed is not None else default.get("speed", 1.0)
        try:
            audio = tts.synthesize(req.text, voice=voice, speed=speed)
        except TTSUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc
        return Response(content=audio, media_type="audio/wav")

    if DIST.exists():
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

        @app.get("/")
        def index():
            return FileResponse(DIST / "index.html")

    return app
