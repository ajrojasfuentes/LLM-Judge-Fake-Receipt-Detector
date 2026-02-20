"""
GLMJudge: Judge implementation using GLM-4.1V-9B-Thinking via Hugging Face Inference Providers (router).
Acts as the third independent "Holistic Auditor" judge.

Requires:
- HF_TOKEN env var with permissions to call Inference Providers.
Optionally:
- Set routing to "novita" (provider) or "fastest"/"cheapest"/"preferred" (policies).
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from huggingface_hub import InferenceClient

from .base_judge import BaseJudge


class GLMJudge(BaseJudge):
    """
    Judge backed by zai-org/GLM-4.5V via Hugging Face Inference Providers.
    Serves as the Holistic Auditor — applies all skills for a cross-validated verdict.
    """

    MODEL_ID = "zai-org/GLM-4.5V"

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
        timeout_s: float = 120.0,
    ):
        super().__init__(
            judge_id=judge_id,
            judge_name=judge_name,
            persona_description=persona_description,
            focus_skills=focus_skills,  # None → all skills with equal weight
        )

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("Missing HF_TOKEN environment variable.")

        self.temperature = temperature
        self.max_tokens = max_tokens

        # routing can be a provider name (e.g., "novita") or a policy ("fastest", "cheapest", "preferred")
        # We'll form the routed model id accordingly: "<model>:<routing>"
        self._routed_model = self.MODEL_ID

        self._client = InferenceClient(
            api_key=hf_token,
            timeout=timeout_s,
        )

    def _call_api(self, prompt: str, image_path: Path) -> str:
        """
        Call GLM-4.5V with the receipt image and the judge prompt via chat completions (VLM).
        Accepts a local image by sending it as a data URL (base64).
        """
        image_b64 = self._encode_image(image_path)
        mime = self._get_mime(image_path)

        messages = [
            {"role": "system", "content": self.persona_description},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                ],
            },
        ]

        # Preferred API (matches Hugging Face Inference Providers docs and your example)
        try:
            completion = self._client.chat.completions.create(
                model=self._routed_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = completion.choices[0].message.content
            return content if isinstance(content, str) else str(content)
        except AttributeError:
            # Backward-compat fallback for older huggingface_hub versions
            completion = self._client.chat_completion(
                model=self._routed_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = completion.choices[0].message.content
            return content if isinstance(content, str) else str(content)

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
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")