"""GLM judge implementation with multi-image support."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Sequence

from huggingface_hub import InferenceClient

from .base_judge import BaseJudge


class GLMJudge(BaseJudge):
    MODEL_ID = "zai-org/GLM-4.5V"

    def __init__(
        self,
        judge_id: str = "judge_3",
        judge_name: str = "Holistic Auditor",
        persona_description: str = (
            "You are a Holistic Document Auditor with broad expertise in receipt authentication. "
            "You apply all forensic skills and provide a cross-validated, independent assessment."
        ),
        temperature: float = 0.3,
        max_tokens: int = 1024,
        focus_skills: list[str] | None = None,
        timeout_s: float = 120.0,
        model_id: str | None = None,
    ):
        super().__init__(
            judge_id=judge_id,
            judge_name=judge_name,
            persona_description=persona_description,
            focus_skills=focus_skills,
        )

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("Missing HF_TOKEN environment variable.")

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_id = model_id or self.MODEL_ID
        self._client = InferenceClient(api_key=hf_token, timeout=timeout_s)

    def _call_api(self, prompt: str, image_paths: Sequence[Path]) -> str:
        user_content = [{"type": "text", "text": prompt}]
        for img in image_paths:
            image_b64 = self._encode_image(img)
            mime = self._get_mime(img)
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                }
            )

        messages = [
            {"role": "system", "content": self.persona_description},
            {"role": "user", "content": user_content},
        ]

        try:
            completion = self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = completion.choices[0].message.content
            return content if isinstance(content, str) else str(content)
        except AttributeError:
            completion = self._client.chat_completion(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = completion.choices[0].message.content
            return content if isinstance(content, str) else str(content)

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
