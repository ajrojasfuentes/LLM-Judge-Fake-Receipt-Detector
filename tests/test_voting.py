"""Tests for the VotingEngine."""

import pytest
from judges.base_judge import JudgeResult
from judges.voting import VotingEngine


def make_result(label: str, confidence: float, judge_id: str = "j1") -> JudgeResult:
    return JudgeResult(
        judge_id=judge_id,
        judge_name=judge_id,
        receipt_id="r001",
        label=label,
        confidence=confidence,
        reasons=["r1", "r2"],
        skill_results={k: "pass" for k in [
            "math_consistency", "typography", "visual_authenticity",
            "layout_structure", "contextual_validation",
        ]},
        flags=[],
        risk_level="low",
    )


def test_majority_fake_2_of_3():
    engine = VotingEngine(strategy="majority")
    results = [
        make_result("FAKE", 90.0, "j1"),
        make_result("FAKE", 80.0, "j2"),
        make_result("REAL", 70.0, "j3"),
    ]
    verdict = engine.aggregate(results)
    assert verdict.label == "FAKE"
    assert verdict.tally == "FAKE (2/3)"


def test_majority_real_2_of_3():
    engine = VotingEngine(strategy="majority")
    results = [
        make_result("REAL", 90.0, "j1"),
        make_result("REAL", 85.0, "j2"),
        make_result("FAKE", 60.0, "j3"),
    ]
    verdict = engine.aggregate(results)
    assert verdict.label == "REAL"


def test_majority_uncertain_when_no_majority():
    engine = VotingEngine(strategy="majority")
    results = [
        make_result("FAKE", 60.0, "j1"),
        make_result("REAL", 60.0, "j2"),
        make_result("UNCERTAIN", 40.0, "j3"),
    ]
    verdict = engine.aggregate(results)
    assert verdict.label == "UNCERTAIN"


def test_uncertain_threshold_forces_uncertain():
    engine = VotingEngine(strategy="majority", uncertain_threshold=2)
    results = [
        make_result("UNCERTAIN", 40.0, "j1"),
        make_result("UNCERTAIN", 35.0, "j2"),
        make_result("FAKE", 80.0, "j3"),
    ]
    verdict = engine.aggregate(results)
    assert verdict.label == "UNCERTAIN"


def test_confidence_weighted_vote():
    engine = VotingEngine(strategy="confidence_weighted")
    results = [
        make_result("FAKE", 90.0, "j1"),
        make_result("REAL", 30.0, "j2"),
        make_result("FAKE", 70.0, "j3"),
    ]
    verdict = engine.aggregate(results)
    assert verdict.label == "FAKE"


def test_flags_are_unioned():
    from judges.base_judge import JudgeResult
    r1 = make_result("FAKE", 80.0, "j1")
    r1.flags = ["TOTAL_MISMATCH"]
    r2 = make_result("FAKE", 75.0, "j2")
    r2.flags = ["FONT_INCONSISTENCY"]
    r3 = make_result("REAL", 60.0, "j3")
    r3.flags = ["TOTAL_MISMATCH"]

    engine = VotingEngine()
    verdict = engine.aggregate([r1, r2, r3])
    assert "TOTAL_MISMATCH" in verdict.all_flags
    assert "FONT_INCONSISTENCY" in verdict.all_flags
    # No duplicates
    assert verdict.all_flags.count("TOTAL_MISMATCH") == 1
