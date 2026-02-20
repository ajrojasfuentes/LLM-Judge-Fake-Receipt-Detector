"""
BaseJudge: Abstract base class for all LLM judge implementations.
Defines the contract every judge must fulfil.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model for a single judge's output
# ---------------------------------------------------------------------------

VALID_LABELS = {"REAL", "FAKE", "UNCERTAIN"}
VALID_SKILL_RESULTS = {"pass", "fail", "uncertain"}
VALID_RISK_LEVELS = {"low", "medium", "high"}

APPROVED_FLAGS = {
    "TOTAL_MISMATCH", "TAX_ERROR", "LINE_ITEM_ERROR", "FONT_INCONSISTENCY",
    "TEXT_OVERLAY", "COPY_PASTE_ARTIFACT", "COMPRESSION_ANOMALY",
    "MISSING_FIELDS", "TEMPLATE_LAYOUT", "IMPLAUSIBLE_DATE",
    "IMPLAUSIBLE_STORE", "CURRENCY_MISMATCH", "PAYMENT_INCONSISTENCY",
    "ERASED_CONTENT", "RESOLUTION_MISMATCH", "SUSPICIOUS_ROUND_TOTAL",
}


@dataclass
class JudgeResult:
    """Structured output from a single LLM judge."""

    judge_id: str
    judge_name: str
    receipt_id: str

    # Required fields (from output schema)
    label: str                              # "REAL" | "FAKE" | "UNCERTAIN"
    confidence: float                       # [0.0 – 100.0]
    reasons: list[str]                      # 2–4 short observations
    skill_results: dict[str, str]           # per-skill pass/fail/uncertain
    flags: list[str]                        # approved flag codes
    risk_level: str                         # "low" | "medium" | "high"

    # Meta
    raw_response: str = field(default="", repr=False)
    parse_error: str | None = None

    def is_valid(self) -> bool:
        return (
            self.label in VALID_LABELS
            and 0.0 <= self.confidence <= 100.0
            and 2 <= len(self.reasons) <= 4
            and self.risk_level in VALID_RISK_LEVELS
        )

    def to_dict(self) -> dict:
        return {
            "judge_id": self.judge_id,
            "judge_name": self.judge_name,
            "receipt_id": self.receipt_id,
            "label": self.label,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "skill_results": self.skill_results,
            "flags": self.flags,
            "risk_level": self.risk_level,
            "parse_error": self.parse_error,
        }

    @classmethod
    def error_result(cls, judge_id: str, judge_name: str, receipt_id: str, error: str) -> "JudgeResult":
        """Return a fallback UNCERTAIN result when parsing fails completely."""
        return cls(
            judge_id=judge_id,
            judge_name=judge_name,
            receipt_id=receipt_id,
            label="UNCERTAIN",
            confidence=0.0,
            reasons=["Failed to parse judge response"],
            skill_results={
                k: "uncertain"
                for k in ["math_consistency", "typography", "visual_authenticity",
                           "layout_structure", "contextual_validation"]
            },
            flags=[],
            risk_level="low",
            parse_error=error,
        )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseJudge(ABC):
    """
    Abstract base judge. Subclasses implement `_call_api` to hit the specific
    model endpoint. The base class handles prompt building, JSON parsing,
    validation, and retry logic.
    """

    MAX_RETRIES = 3

    def __init__(self, judge_id: str, judge_name: str, persona_description: str,
                 focus_skills: list[str] | None = None):
        from skills import Rubric
        self.judge_id = judge_id
        self.judge_name = judge_name
        self.persona_description = persona_description
        self.focus_skills = focus_skills
        self._rubric = Rubric()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def judge(self, receipt_id: str, image_path: Path) -> JudgeResult:
        """
        Main entry point. Builds the prompt, calls the API, parses the response.
        Retries up to MAX_RETRIES times on invalid JSON.
        """
        prompt = self._rubric.build_prompt(
            receipt_id=receipt_id,
            persona_name=self.judge_name,
            persona_description=self.persona_description,
            focus_skills=self.focus_skills,
        )

        last_error = ""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                raw = self._call_api(prompt=prompt, image_path=image_path)
                result = self._parse_response(raw, receipt_id)
                if result.is_valid():
                    return result
                last_error = f"Validation failed: label={result.label}, confidence={result.confidence}"
            except Exception as exc:
                last_error = str(exc)

        return JudgeResult.error_result(
            judge_id=self.judge_id,
            judge_name=self.judge_name,
            receipt_id=receipt_id,
            error=f"Max retries exceeded. Last error: {last_error}",
        )

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _call_api(self, prompt: str, image_path: Path) -> str:
        """
        Call the LLM API and return the raw string response.
        Must be implemented by each model-specific subclass.
        """
        ...

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str, receipt_id: str) -> JudgeResult:
        """Extract and validate JSON from the raw LLM response."""
        json_str = self._extract_json(raw)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            return JudgeResult.error_result(
                self.judge_id, self.judge_name, receipt_id,
                f"JSON decode error: {exc}",
            )

        # Normalise and validate fields
        label = str(data.get("label", "UNCERTAIN")).upper()
        if label not in VALID_LABELS:
            label = "UNCERTAIN"

        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(100.0, confidence))

        reasons = data.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        reasons = [str(r) for r in reasons[:4]]
        if len(reasons) < 2:
            reasons += ["No reason provided"] * (2 - len(reasons))

        skill_results = data.get("skill_results", {})
        for k in ["math_consistency", "typography", "visual_authenticity",
                  "layout_structure", "contextual_validation"]:
            if skill_results.get(k) not in VALID_SKILL_RESULTS:
                skill_results[k] = "uncertain"

        flags = [f for f in data.get("flags", []) if f in APPROVED_FLAGS]

        risk_level = str(data.get("risk_level", "low")).lower()
        if risk_level not in VALID_RISK_LEVELS:
            risk_level = "low"

        return JudgeResult(
            judge_id=self.judge_id,
            judge_name=self.judge_name,
            receipt_id=receipt_id,
            label=label,
            confidence=confidence,
            reasons=reasons,
            skill_results=skill_results,
            flags=flags,
            risk_level=risk_level,
            raw_response=raw,
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Attempt to extract a JSON object from the raw response text.
        Handles markdown code fences and extra surrounding text.
        """
        # Strip markdown fences
        text = re.sub(r"```(?:json)?", "", text).strip()
        # Find first { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return text
