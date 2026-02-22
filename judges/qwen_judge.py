"""Qwen judge implementation with optional multi-image VLM input."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from huggingface_hub import InferenceClient

from .base_judge import BaseJudge


class QwenJudge(BaseJudge):
    MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"

    def __init__(
        self,
        judge_id: str,
        judge_name: str,
        persona_description: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        focus_skills: list[str] | None = None,
        forensic_mode: str = "FULL",
    ):
        super().__init__(
            judge_id=judge_id,
            judge_name=judge_name,
            persona_description=persona_description,
            focus_skills=focus_skills,
            forensic_mode=forensic_mode,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = InferenceClient(api_key=os.environ["HF_TOKEN"])

    def _call_api(self, prompt: str, image_paths: list[Path]) -> str:
        content = []
        for image_path in image_paths:
            image_b64 = self._encode_image(image_path)
            mime = self._get_mime(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                }
            )
        content.append({"type": "text", "text": prompt})

        response = self._client.chat.completions.create(
            model=self.MODEL_ID,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _get_mime(image_path: Path) -> str:
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
            ".webp": "image/webp",
        }.get(image_path.suffix.lower(), "image/jpeg")


def make_forensic_accountant() -> QwenJudge:
    return QwenJudge(
        judge_id="judge_1",
        judge_name="Forensic Accountant",
        persona_description=(
            "You are a Forensic Accountant specialized in detecting financial document fraud. "
            "Your primary focus is mathematical consistency and numerical anomalies. "
            "You are highly precise and conservative — you only report FAKE if the evidence is clear."
        ),
        temperature=0.2,
        max_tokens=1024,
        focus_skills=["math_consistency", "contextual_validation"],
        forensic_mode="READING",
    )


def make_document_examiner() -> QwenJudge:
    return QwenJudge(
        judge_id="judge_2",
        judge_name="Document Examiner",
        persona_description=(
            "You are a Forensic Document Examiner specialized in detecting visual forgeries. "
            "Your primary focus is typographic anomalies, visual artifacts, and layout inconsistencies. "
            "You are thorough and detail-oriented, trained to catch subtle image-level manipulations."
        ),
        temperature=0.6,
        max_tokens=1024,
        focus_skills=["typography_analysis", "visual_authenticity", "layout_structure"],
        forensic_mode="GRAPHIC",
    )
