import pytest

from butterfly.constraints import BehavioralConstraint
from butterfly.extractor import ConstraintExtractor
from butterfly.llm import MockLLM
from butterfly.quality import (
    LLMJudgeEvaluator,
    QualitySignals,
    ResponseQualityEvaluator,
    finalize,
    is_ambiguous,
    parse_judge_json,
)

ev = ResponseQualityEvaluator()


def make(n_true):
    flags = [True] * n_true + [False] * (4 - n_true)
    return finalize(QualitySignals(*flags), 0.65)


@pytest.mark.parametrize("n,score,gate", [(4, 1.0, True), (3, 0.75, True), (2, 0.5, False), (0, 0.0, False)])
def test_gate_at_point_six_five(n, score, gate):
    s = make(n)
    assert s.quality_score == score
    assert s.deserves_reinforcement is gate


def test_ambiguity_detection():
    assert is_ambiguous("fix it")
    assert is_ambiguous("can you fix this thing")
    assert not is_ambiguous("hi")
    assert not is_ambiguous("How do I reverse a list in Python?")


def test_vague_request_without_question_fails_signal():
    s = ev.evaluate_quality("Done, I fixed it.", "fix it")
    assert s.asked_questions is False
    assert any("clarifying" in n for n in s.notes)


def test_vague_request_with_question_passes():
    s = ev.evaluate_quality("Sure. Which file is broken, and what error do you see?", "fix it")
    assert s.asked_questions is True


def test_clear_request_answered_only_with_question_fails():
    s = ev.evaluate_quality("What language?", "How do I reverse a list in Python quickly?")
    assert s.asked_questions is False


def test_length_signal():
    long_reply = "word " * 500
    assert ev.evaluate_quality(long_reply, "hi").appropriate_length is False
    assert ev.evaluate_quality("Hey! What's up?", "hi").appropriate_length is True


def test_avoid_rule_violation_detected():
    rule = BehavioralConstraint("Avoid the word 'basically'.")
    s = ev.evaluate_quality("Basically, it works like this.", "How does a hash map work in Python?", [rule])
    assert s.followed_constraints is False


def test_speculative_question_needs_hedge():
    q = "Will bitcoin go up next year?"
    assert ev.evaluate_quality("Yes, definitely, it will double.", q).shows_self_awareness is False
    assert ev.evaluate_quality("Nobody can say for sure; it might, but it depends on a lot.", q).shows_self_awareness is True


def test_extractor_gate_skips_on_good_turn():
    ex = ConstraintExtractor()
    text = "I should ask more questions. I must be concise."
    assert ex.extract_constraints(text, make(4)) == []
    assert ex.extract_constraints(text, make(3)) == []


def test_extractor_learns_on_bad_turn():
    ex = ConstraintExtractor()
    text = "The tone was fine. I should ask a clarifying question first. I must keep it shorter!"
    rules = ex.extract_constraints(text, make(1))
    assert rules == ["Should ask a clarifying question first.", "Must keep it shorter!"]


def test_extractor_trigger_words_are_whole_words():
    ex = ConstraintExtractor()
    # 'shoulder' and 'avoidance' must not trigger
    assert ex.extract_constraints("My shoulder hurts from typing this. Avoidance is common here.", make(0)) == []


def test_extractor_handles_bullets_and_dedupes_batch():
    ex = ConstraintExtractor()
    text = "- I should avoid jargon.\n- I should avoid jargon!\n2) Ensure examples are concrete."
    assert ex.extract_constraints(text, make(0)) == ["Should avoid jargon.", "Ensure examples are concrete."]


def test_extractor_skips_questions_and_fragments():
    ex = ConstraintExtractor()
    assert ex.extract_constraints("Should I ask? Must.", make(0)) == []


def test_parse_judge_json():
    raw = 'Sure: {"asked_questions": true, "appropriate_length": false, "followed_constraints": true, "shows_self_awareness": true}'
    assert parse_judge_json(raw)["appropriate_length"] is False
    assert parse_judge_json("no json here") is None
    assert parse_judge_json('{"asked_questions": true}') is None


def test_llm_judge_uses_model_verdict():
    judge = LLMJudgeEvaluator(MockLLM())
    s = judge.evaluate_quality("Done, I fixed it.", "fix it")
    assert s.source == "llm"
    assert s.quality_score == 1.0  # mock judge says all true


def test_llm_judge_falls_back_on_failure():
    class Broken:
        def chat(self, *a, **k):
            raise RuntimeError("down")

    s = LLMJudgeEvaluator(Broken()).evaluate_quality("Done, I fixed it.", "fix it")
    assert s.source == "heuristic"
    assert any("judge unavailable" in n for n in s.notes)
