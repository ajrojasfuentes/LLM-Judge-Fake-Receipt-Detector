"""Tests for judge parsing and validation logic."""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from judges.base_judge import BaseJudge, JudgeResult


class DummyJudge(BaseJudge):
    """Concrete judge for testing the base class."""
    def __init__(self, fake_response: str):
        super().__init__(
            judge_id="test_judge",
            judge_name="Test Judge",
            persona_description="Test persona",
            focus_skills=None,
        )
        self._fake_response = fake_response

    def _call_api(self, prompt: str, image_path: Path) -> str:
        return self._fake_response


VALID_JSON = json.dumps({
    "label": "FAKE",
    "confidence": 85.0,
    "reasons": ["Total mismatch", "Font inconsistency"],
    "skill_results": {
        "math_consistency": "fail",
        "typography": "fail",
        "visual_authenticity": "uncertain",
        "layout_structure": "pass",
        "contextual_validation": "pass",
    },
    "flags": ["TOTAL_MISMATCH", "FONT_INCONSISTENCY"],
    "risk_level": "high",
})


def test_valid_json_parsed_correctly():
    judge = DummyJudge(fake_response=VALID_JSON)
    result = judge._parse_response(VALID_JSON, "receipt_001")
    assert result.label == "FAKE"
    assert result.confidence == 85.0
    assert len(result.reasons) == 2
    assert result.risk_level == "high"
    assert result.is_valid()


def test_json_inside_markdown_fence():
    wrapped = f"```json\n{VALID_JSON}\n```"
    judge = DummyJudge(fake_response=wrapped)
    result = judge._parse_response(wrapped, "receipt_001")
    assert result.label == "FAKE"
    assert result.is_valid()


def test_invalid_label_defaults_to_uncertain():
    bad = json.dumps({
        "label": "MAYBE",
        "confidence": 50.0,
        "reasons": ["reason 1", "reason 2"],
        "skill_results": {k: "pass" for k in
                          ["math_consistency", "typography", "visual_authenticity",
                           "layout_structure", "contextual_validation"]},
        "flags": [],
        "risk_level": "low",
    })
    judge = DummyJudge(fake_response=bad)
    result = judge._parse_response(bad, "receipt_001")
    assert result.label == "UNCERTAIN"


def test_confidence_clamped():
    bad = json.dumps({
        "label": "REAL",
        "confidence": 150.0,
        "reasons": ["r1", "r2"],
        "skill_results": {k: "pass" for k in
                          ["math_consistency", "typography", "visual_authenticity",
                           "layout_structure", "contextual_validation"]},
        "flags": [],
        "risk_level": "low",
    })
    judge = DummyJudge(fake_response=bad)
    result = judge._parse_response(bad, "r1")
    assert result.confidence == 100.0


def test_unapproved_flags_are_filtered():
    bad = json.dumps({
        "label": "FAKE",
        "confidence": 70.0,
        "reasons": ["r1", "r2"],
        "skill_results": {k: "fail" for k in
                          ["math_consistency", "typography", "visual_authenticity",
                           "layout_structure", "contextual_validation"]},
        "flags": ["TOTAL_MISMATCH", "MADE_UP_FLAG", "ANOTHER_BAD_FLAG"],
        "risk_level": "medium",
    })
    judge = DummyJudge(fake_response=bad)
    result = judge._parse_response(bad, "r1")
    assert "MADE_UP_FLAG" not in result.flags
    assert "TOTAL_MISMATCH" in result.flags


def test_completely_invalid_json_returns_error_result():
    judge = DummyJudge(fake_response="This is not JSON at all!!")
    result = judge._parse_response("This is not JSON at all!!", "r1")
    assert result.label == "UNCERTAIN"
    assert result.parse_error is not None
