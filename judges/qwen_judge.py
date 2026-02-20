"""
QwenJudge: Judge implementation using Qwen2.5-VL via HuggingFace Inference API.
Supports two personas (Forensic Accountant and Document Examiner)
via different temperatures and focus skills.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from huggingface_hub import InferenceClient

from .base_judge import BaseJudge


class QwenJudge(BaseJudge):
    """
    Judge backed by Qwen/Qwen2.5-VL-72B-Instruct via HuggingFace InferenceClient.

    Two instances are created (different personas + temperatures):
      - judge_1: Forensic Accountant, T=0.1
      - judge_2: Document Examiner,   T=0.7
    """

    MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"

    def __init__(
        self,
        judge_id: str,
        judge_name: str,
        persona_description: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        focus_skills: list[str] | None = None,
    ):
        super().__init__(
            judge_id=judge_id,
            judge_name=judge_name,
            persona_description=persona_description,
            focus_skills=focus_skills,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = InferenceClient(
            provider="huggingface",
            api_key=os.environ["HF_TOKEN"],
        )

    def _call_api(self, prompt: str, image_path: Path) -> str:
        """
        Call Qwen2.5-VL with the receipt image (base64-encoded) and the judge prompt.
        Uses the chat completions endpoint with vision support.
        """
        image_b64 = self._encode_image(image_path)
        mime = self._get_mime(image_path)

        response = self._client.chat.completions.create(
            model=self.MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{image_b64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _get_mime(image_path: Path) -> str:
        ext = image_path.suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
        }.get(ext, "image/jpeg")


# ---------------------------------------------------------------------------
# Factory functions for the two Qwen personas
# ---------------------------------------------------------------------------

def make_forensic_accountant() -> QwenJudge:
    return QwenJudge(
        judge_id="judge_1",
        judge_name="Forensic Accountant",
        persona_description=(
            "You are a Forensic Accountant specialized in detecting financial document fraud. "
            "Your primary focus is mathematical consistency and numerical anomalies. "
            "You are highly precise and conservative — you only report FAKE if the evidence is clear."
        ),
        temperature=0.1,
        max_tokens=1024,
        focus_skills=["math_consistency", "contextual_validation"],
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
        temperature=0.7,
        max_tokens=1024,
        focus_skills=["typography_analysis", "visual_authenticity", "layout_structure"],
    )
