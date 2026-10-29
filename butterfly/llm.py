"""Generation backends.

Butterfly is model-agnostic (Section 2): any client with a
``chat(messages, system=None) -> str`` method works.

- OllamaClient: local open-weight models (the reference build).
- OpenAICompatibleClient: any server speaking the OpenAI chat API,
  e.g. LM Studio, llama.cpp server, vLLM, or a hosted provider.
- MockLLM: deterministic, no network. Used by the tests and by demo mode
  so the UI works before a model is installed.
"""

from __future__ import annotations

import os
import re
from typing import Protocol

import httpx


class LLMClient(Protocol):
    name: str

    def chat(self, messages: list[dict], system: str | None = None) -> str: ...

    def health(self) -> dict: ...


class LLMError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434",
                 timeout: float = 180.0, temperature: float = 0.7):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.name = f"ollama:{model}"

    def chat(self, messages: list[dict], system: str | None = None) -> str:
        payload_messages = ([{"role": "system", "content": system}] if system else []) + messages
        try:
            r = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": payload_messages,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Can't reach Ollama at {self.base_url}. Start it, or run in demo mode."
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:200]
            if exc.response.status_code == 404:
                raise LLMError(f"Model '{self.model}' isn't installed. Run: ollama pull {self.model}") from exc
            raise LLMError(f"Ollama error {exc.response.status_code}: {detail}") from exc
        except httpx.TimeoutException as exc:
            raise LLMError("Ollama took too long to answer. Try a smaller model.") from exc
        return r.json().get("message", {}).get("content", "").strip()

    def health(self) -> dict:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as exc:
            return {"ok": False, "backend": self.name, "error": f"Ollama not reachable: {type(exc).__name__}"}
        installed = any(m == self.model or m.split(":")[0] == self.model for m in models)
        return {
            "ok": installed,
            "backend": self.name,
            "models": models,
            "error": None if installed else f"Model '{self.model}' not installed. Run: ollama pull {self.model}",
        }


class OpenAICompatibleClient:
    def __init__(self, model: str, base_url: str, api_key: str | None = None,
                 timeout: float = 120.0, temperature: float = 0.7):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.name = f"openai-compatible:{model}"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def chat(self, messages: list[dict], system: str | None = None) -> str:
        payload_messages = ([{"role": "system", "content": system}] if system else []) + messages
        try:
            r = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json={"model": self.model, "messages": payload_messages, "temperature": self.temperature},
                timeout=self.timeout,
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Model server error: {exc}") from exc
        choices = r.json().get("choices") or [{}]
        return (choices[0].get("message", {}).get("content") or "").strip()

    def health(self) -> dict:
        try:
            r = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=5)
            r.raise_for_status()
            return {"ok": True, "backend": self.name, "error": None}
        except Exception as exc:
            return {"ok": False, "backend": self.name, "error": f"Model server not reachable: {type(exc).__name__}"}


class MockLLM:
    """Deterministic stand-in for a model.

    Replies are rule-aware so demo mode visibly reacts to learned rules:
    a vague message gets an answer with no question (a "bad" turn that
    triggers learning) until a rule about asking questions is active.
    Pass ``script`` to return fixed replies in order (used by tests).
    """

    name = "mock"

    def __init__(self, script: list[str] | None = None):
        self.script = list(script or [])
        self.calls: list[dict] = []

    def health(self) -> dict:
        return {"ok": True, "backend": "mock (demo mode)", "error": None}

    def chat(self, messages: list[dict], system: str | None = None) -> str:
        self.calls.append({"messages": messages, "system": system})
        last = messages[-1]["content"] if messages else ""

        if "Return exactly this JSON object" in last:
            return ('{"asked_questions": true, "appropriate_length": true, '
                    '"followed_constraints": true, "shows_self_awareness": true}')
        if last.startswith("SELF-EVALUATION"):
            return self._self_eval(last)
        if last.startswith("MIRROR LOOP"):
            return ("Looking back over my recent replies, my tone stayed mostly consistent. "
                    "I could balance curiosity with getting to the point.")
        if self.script:
            return self.script.pop(0)
        return self._reply(last, system or "")

    @staticmethod
    def _self_eval(prompt: str) -> str:
        lines = []
        if "no clarifying question" in prompt:
            lines.append("I should ask a clarifying question when the request is vague.")
        if "length" in prompt and "outside" in prompt:
            lines.append("I need to keep the answer length proportional to the request.")
        if "no hedging" in prompt:
            lines.append("I must acknowledge uncertainty when answering speculative questions.")
        if "broke rule" in prompt:
            lines.append("I must follow my active rules more carefully.")
        if not lines:
            lines.append("I should check that my answer directly addresses what was asked.")
        return " ".join(lines)

    @staticmethod
    def _reply(user: str, system: str) -> str:
        words = re.findall(r"[A-Za-z0-9']+", user)
        sys_lower = system.lower()
        wants_questions = "clarifying question" in sys_lower
        wants_hedge = "uncertainty" in sys_lower
        topic = " ".join(words[:6]) or "that"
        if len(words) <= 6 and wants_questions:
            return f"Happy to help with {topic}. Can you tell me a bit more about what you need?"
        if len(words) <= 6:
            return f"Sure, here is a quick take on {topic}."
        hedge = "I might be wrong, but " if wants_hedge else ""
        return (f"{hedge}{'here' if hedge else 'Here'}'s my answer about {topic}. "
                "The short version: it comes down to your goals and constraints, "
                "so weigh the tradeoffs before deciding.")


def client_from_env() -> LLMClient:
    """Pick a backend from environment variables.

    BUTTERFLY_BACKEND = ollama (default) | openai | mock
    BUTTERFLY_MODEL   = model name (default llama3.1)
    OLLAMA_URL        = default http://localhost:11434
    OPENAI_BASE_URL / OPENAI_API_KEY for the openai backend
    """
    backend = os.getenv("BUTTERFLY_BACKEND", "ollama").lower()
    model = os.getenv("BUTTERFLY_MODEL", "llama3.1")
    if backend == "mock":
        return MockLLM()
    if backend == "openai":
        base = os.getenv("OPENAI_BASE_URL")
        if not base:
            raise LLMError("Set OPENAI_BASE_URL to use the openai backend.")
        return OpenAICompatibleClient(model, base, os.getenv("OPENAI_API_KEY"))
    return OllamaClient(model, os.getenv("OLLAMA_URL", "http://localhost:11434"))
