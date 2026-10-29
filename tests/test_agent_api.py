import pytest
from fastapi.testclient import TestClient

from butterfly.agent import ButterflyAgent
from butterfly.config import ButterflyConfig
from butterfly.llm import LLMError, MockLLM
from butterfly.server import create_app

GOOD = "Sure. Which file is broken, and what error do you see?"  # passes all four for "fix it"
BAD = "Yes, definitely, it is done."                               # fails 2+ for "should I fix it"


class FakeLLM:
    """Generation replies come from `replies`; self-eval and mirror are canned."""

    name = "fake"

    def __init__(self, replies, self_eval="I should ask a clarifying question when the request is vague."):
        self.replies = list(replies)
        self.self_eval = self_eval
        self.prompts = []

    def health(self):
        return {"ok": True, "backend": "fake", "error": None}

    def chat(self, messages, system=None):
        last = messages[-1]["content"]
        self.prompts.append((last, system))
        if last.startswith("SELF-EVALUATION"):
            return self.self_eval
        if last.startswith("MIRROR LOOP"):
            return "I asked a lot of questions recently."
        return self.replies.pop(0) if self.replies else GOOD


def test_good_turn_reinforces_and_extracts_nothing():
    agent = ButterflyAgent(FakeLLM([GOOD]), seed_rules=["Ensure answers stay on topic."])
    r = agent.turn("fix it")
    assert r.reinforced and r.rules_added == []
    assert r.self_evaluation is None
    assert agent.constraints.items[0].strength == pytest.approx(1.1)


def test_bad_turn_weakens_and_learns_from_self_eval_not_response():
    llm = FakeLLM([BAD + " You should restart the computer."])
    agent = ButterflyAgent(llm, seed_rules=["Ensure answers stay on topic."])
    r = agent.turn("should I fix it")
    assert not r.reinforced
    assert agent.constraints.items[0].strength == pytest.approx(0.95)
    # The rule comes from the self-evaluation, not "You should restart" in the reply.
    assert r.rules_added == ["Should ask a clarifying question when the request is vague."]
    assert any(p.startswith("SELF-EVALUATION") for p, _ in llm.prompts)


def test_repeated_bad_turns_do_not_bloat_rules():
    agent = ButterflyAgent(FakeLLM([BAD] * 5))
    for _ in range(5):
        agent.turn("should I fix it")
    assert len(agent.constraints) == 1


def test_learned_rule_appears_in_next_system_prompt():
    llm = FakeLLM([BAD, GOOD])
    agent = ButterflyAgent(llm)
    agent.turn("should I fix it")
    agent.turn("fix it")
    gen_system = [s for p, s in llm.prompts if p == "fix it"][0]
    assert "clarifying question" in gen_system


def test_mirror_loop_cadence():
    agent = ButterflyAgent(FakeLLM([GOOD] * 9))
    fired = [agent.turn("fix it").mirror_loop_ran for _ in range(9)]
    assert [i + 1 for i, f in enumerate(fired) if f] == [3, 6, 9]


def test_mirror_loop_nudges_then_corrects():
    agent = ButterflyAgent(FakeLLM([GOOD] * 3), traits={"curious": 0.5, "formal": 0.5, "verbose": 0.5})
    for _ in range(3):
        r = agent.turn("fix it")
    assert r.mirror_reflection == "I asked a lot of questions recently."
    # Every reply asked a question, so curiosity was nudged up before mean reversion.
    assert agent.traits.get("curious") > 0.5
    assert r.trait_changes["curious"] > 0


def test_traits_missing_from_profile_are_skipped_not_created():
    agent = ButterflyAgent(FakeLLM([GOOD] * 3), traits={"warm": 0.5, "formal": 0.5})
    for _ in range(3):
        agent.turn("fix it")
    assert set(agent.traits.traits) == {"warm", "formal"}


def test_self_eval_failure_falls_back_to_detected_problems():
    class Flaky(FakeLLM):
        def chat(self, messages, system=None):
            if messages[-1]["content"].startswith("SELF-EVALUATION"):
                raise RuntimeError("model down")
            return super().chat(messages, system)

    agent = ButterflyAgent(Flaky([BAD]))
    r = agent.turn("should I fix it")
    assert r.rules_added  # learned from the fallback text built from the notes


def test_empty_message_rejected():
    with pytest.raises(ValueError):
        ButterflyAgent(FakeLLM([])).turn("   ")


def test_persistence_roundtrip():
    agent = ButterflyAgent(FakeLLM([BAD, GOOD, GOOD]))
    for msg in ("should I fix it", "fix it", "fix it"):
        agent.turn(msg)
    clone = ButterflyAgent(FakeLLM([]))
    clone.load_dict(agent.to_dict())
    assert clone.to_dict() == agent.to_dict()


# ---------------- Homeostasis simulations ----------------

def test_long_run_good_behavior_stays_bounded():
    agent = ButterflyAgent(FakeLLM([GOOD] * 200), seed_rules=["Ensure answers stay on topic."])
    for _ in range(200):
        agent.turn("fix it")
    assert all(0.0 <= v <= 1.0 for v in agent.traits.traits.values())
    assert agent.constraints.items[0].strength <= 2.0


def test_sustained_failure_prunes_everything_eventually():
    llm = FakeLLM([BAD] * 40, self_eval="Nothing useful here.")
    agent = ButterflyAgent(llm, seed_rules=["Ensure answers stay on topic."])
    for _ in range(40):
        agent.turn("should I fix it")
    assert len(agent.constraints) == 0


def test_max_constraints_bounds_rule_growth_under_varied_failures():
    evals = [f"I should handle case number {i} with extra care and detail." for i in range(30)]

    class Varied(FakeLLM):
        def chat(self, messages, system=None):
            if messages[-1]["content"].startswith("SELF-EVALUATION"):
                return evals.pop(0)
            return super().chat(messages, system)

    agent = ButterflyAgent(Varied([BAD] * 30), ButterflyConfig(max_constraints=5))
    for _ in range(30):
        agent.turn("should I fix it")
    assert len(agent.constraints) <= 5


# ---------------- API ----------------

@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(llm=MockLLM(), data_dir=tmp_path))


def test_api_health_and_personas(client):
    assert client.get("/api/health").json()["ok"] is True
    keys = [p["key"] for p in client.get("/api/personas").json()]
    assert keys == ["nova", "echo"]


def test_api_chat_returns_turn_and_state(client):
    body = client.post("/api/chat", json={"persona": "nova", "message": "should I fix it"}).json()
    assert "response" in body["turn"]
    assert body["state"]["message_count"] == 1
    assert body["turn"]["signals"]["quality_score"] <= 0.5
    assert body["turn"]["rules_added"]


def test_api_state_persists_across_restart(tmp_path):
    c1 = TestClient(create_app(llm=MockLLM(), data_dir=tmp_path))
    c1.post("/api/chat", json={"persona": "echo", "message": "should I fix it"})
    c2 = TestClient(create_app(llm=MockLLM(), data_dir=tmp_path))
    state = c2.get("/api/state/echo").json()
    assert state["message_count"] == 1 and state["constraints"]


def test_api_reset(client):
    client.post("/api/chat", json={"persona": "nova", "message": "should I fix it"})
    state = client.post("/api/reset/nova").json()
    assert state["message_count"] == 0 and state["constraints"] == []


def test_api_unknown_persona_404(client):
    assert client.post("/api/chat", json={"persona": "zed", "message": "hi"}).status_code == 404


def test_api_model_error_is_502(tmp_path):
    class Down(MockLLM):
        def chat(self, *a, **k):
            raise LLMError("Can't reach Ollama")

    c = TestClient(create_app(llm=Down(), data_dir=tmp_path))
    r = c.post("/api/chat", json={"persona": "nova", "message": "hello there friend"})
    assert r.status_code == 502 and "Ollama" in r.json()["detail"]


def test_custom_personas_file(tmp_path):
    (tmp_path / "personas.json").write_text(
        '{"sage": {"name": "Sage", "traits": {"calm": 0.8}, "seed_rules": ["Avoid slang in replies."]}}'
    )
    c = TestClient(create_app(llm=MockLLM(), data_dir=tmp_path))
    state = c.get("/api/state/sage").json()
    assert state["traits"]["traits"] == {"calm": 0.8}
    assert state["constraints"][0]["text"] == "Avoid slang in replies."
