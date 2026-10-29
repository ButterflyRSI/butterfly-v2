import pytest

from butterfly.config import ButterflyConfig
from butterfly.constraints import BehavioralConstraint, ConstraintSet, prune_constraints
from butterfly.traits import TraitProfile, UnknownTraitError


# ---------------- Method 1 ----------------

def test_strengthen_caps_at_two():
    c = BehavioralConstraint("Ask a question.")
    for _ in range(50):
        c.strengthen()
    assert c.strength == pytest.approx(2.0)
    assert c.successes == 50


def test_weaken_floors_at_point_one():
    c = BehavioralConstraint("Ask a question.")
    for _ in range(50):
        c.weaken()
    assert c.strength == pytest.approx(0.1)
    assert c.failures == 50


def test_asymmetry_alternating_trends_upward():
    c = BehavioralConstraint("Ask a question.")
    for i in range(20):
        c.strengthen() if i % 2 == 0 else c.weaken()
    assert c.strength == pytest.approx(1.5)


def test_fresh_rule_pruned_after_sixteen_failures():
    cs = ConstraintSet(ButterflyConfig())
    cs.add("Ensure the answer is short.")
    failures = 0
    while len(cs):
        cs.weaken()
        cs.prune()
        failures += 1
        assert failures < 100
    assert failures == 16


def test_prune_is_strictly_above_floor():
    rules = [BehavioralConstraint("a b c", 0.2), BehavioralConstraint("d e f", 0.21)]
    assert [c.text for c in prune_constraints(rules)] == ["d e f"]


def test_dedupe_against_stored_rules():
    cs = ConstraintSet(ButterflyConfig())
    assert cs.add("Ask a clarifying question when the request is vague.")
    assert cs.add("ask a clarifying question when the request is vague") is None
    assert cs.add("Ask a clarifying question when a request is vague.") is None  # near-duplicate
    assert len(cs) == 1


def test_max_constraints_drops_weakest():
    cs = ConstraintSet(ButterflyConfig(max_constraints=2))
    a = cs.add("Rule one about tone here.")
    b = cs.add("Rule two about length here.")
    c = cs.add("Rule three about questions here.")
    a.strength, b.strength, c.strength = 1.5, 0.5, 1.0
    removed = cs.prune()
    assert [r.text for r in removed] == [b.text]
    assert {r.text for r in cs} == {a.text, c.text}


def test_constraint_roundtrip():
    c = BehavioralConstraint("Avoid jargon.", 1.3, 4, 2, 7)
    assert BehavioralConstraint.from_dict(c.to_dict()).to_dict() == c.to_dict()


# ---------------- Method 2 ----------------

def test_adjust_clamps():
    t = TraitProfile({"curious": 0.98})
    t.adjust("curious", 0.5)
    assert t.get("curious") == 1.0
    t.adjust("curious", -5)
    assert t.get("curious") == 0.0


def test_adjust_unknown_trait_raises_instead_of_creating():
    t = TraitProfile({"curious": 0.5})
    with pytest.raises(UnknownTraitError):
        t.adjust("curiuos", 0.05)
    assert "curiuos" not in t.traits


def test_drift_correction_preserves_mean_and_order():
    t = TraitProfile({"curious": 0.9, "formal": 0.2, "verbose": 0.5})
    before_mean = t.mean()
    t.correct_drift(0.15)
    assert t.mean() == pytest.approx(before_mean)
    v = t.traits
    assert v["curious"] > v["verbose"] > v["formal"]
    assert v["curious"] == pytest.approx(0.9 + (before_mean - 0.9) * 0.15)


def test_drift_correction_shrinks_spread():
    t = TraitProfile({"a": 1.0, "b": 0.0})
    t.correct_drift(0.15)
    assert t.traits["a"] - t.traits["b"] == pytest.approx(0.85)  # spread shrinks by 15%


def test_baseline_pull_returns_pinned_profile_home():
    t = TraitProfile({"a": 1.0, "b": 1.0}, baseline={"a": 0.5, "b": 0.3})
    for _ in range(40):
        t.correct_drift(0.15, baseline_pull=0.1)
    # Mean reversion and baseline pull settle together near the baseline,
    # off the ceiling, with the original ordering intact.
    a, b = t.traits["a"], t.traits["b"]
    assert 0.4 < a < 0.55 and 0.3 < b < 0.45 and a > b


def test_no_baseline_pull_leaves_pinned_profile_pinned():
    t = TraitProfile({"a": 1.0, "b": 1.0}, baseline={"a": 0.5, "b": 0.3})
    t.correct_drift(0.15)
    assert t.traits == {"a": 1.0, "b": 1.0}


def test_describe_and_roundtrip():
    t = TraitProfile({"curious": 0.8, "formal": 0.1})
    assert t.describe() == "very curious, not at all formal"
    assert TraitProfile.from_dict(t.to_dict()).traits == t.traits


def test_config_validation():
    with pytest.raises(ValueError):
        ButterflyConfig(mirror_loop_interval=0).validate()
    with pytest.raises(ValueError):
        ButterflyConfig(quality_threshold=1.5).validate()
    cfg = ButterflyConfig(max_constraints=5)
    assert ButterflyConfig.from_dict(cfg.to_dict()) == cfg
