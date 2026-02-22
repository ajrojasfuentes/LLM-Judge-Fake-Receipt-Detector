"""Qwen judge implementation with multi-image support."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Sequence

from huggingface_hub import InferenceClient

from .base_judge import BaseJudge


class QwenJudge(BaseJudge):
    MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"

    def __init__(
        self,
        judge_id: str,
        judge_name: str,
        persona_description: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        focus_skills: list[str] | None = None,
        model_id: str | None = None,
    ):
        super().__init__(
            judge_id=judge_id,
            judge_name=judge_name,
            persona_description=persona_description,
            focus_skills=focus_skills,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_id = model_id or self.MODEL_ID
        self._client = InferenceClient(api_key=os.environ["HF_TOKEN"])

    def _call_api(self, prompt: str, image_paths: Sequence[Path]) -> str:
        content = []
        for img in image_paths:
            image_b64 = self._encode_image(img)
            mime = self._get_mime(img)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                }
            )

        content.append({"type": "text", "text": prompt})

        response = self._client.chat.completions.create(
            model=self.model_id,
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
        ext = image_path.suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")
