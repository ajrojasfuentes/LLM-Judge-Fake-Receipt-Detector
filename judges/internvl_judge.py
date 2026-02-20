"""
InternVLJudge: Judge implementation using InternVL2.5-78B via HuggingFace Inference API.
Acts as the third independent "Holistic Auditor" judge.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from huggingface_hub import InferenceClient

from .base_judge import BaseJudge


class InternVLJudge(BaseJudge):
    """
    Judge backed by OpenGVLab/InternVL2_5-78B via HuggingFace InferenceClient.
    Serves as the Holistic Auditor — applies all 5 skills for a cross-validated verdict.
    """

    MODEL_ID = "OpenGVLab/InternVL2_5-78B"

    def __init__(
        self,
        judge_id: str = "judge_3",
        judge_name: str = "Holistic Auditor",
        persona_description: str = (
            "You are a Holistic Document Auditor with broad expertise in receipt authentication. "
            "You apply all forensic skills and provide a cross-validated, independent assessment. "
            "You treat mathematical, typographic, visual, structural, and contextual evidence with equal weight."
        ),
        temperature: float = 0.3,
        max_tokens: int = 1024,
        focus_skills: list[str] | None = None,
    ):
        super().__init__(
            judge_id=judge_id,
            judge_name=judge_name,
            persona_description=persona_description,
            focus_skills=focus_skills,  # None → all skills with equal weight
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = InferenceClient(
            provider="huggingface",
            api_key=os.environ["HF_TOKEN"],
        )

    def _call_api(self, prompt: str, image_path: Path) -> str:
        """
        Call InternVL2.5 with the receipt image and the judge prompt.
        Uses HuggingFace InferenceClient chat completions with vision.
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
