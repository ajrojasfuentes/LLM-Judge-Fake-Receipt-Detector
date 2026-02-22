"""Compatibility and orchestration layer for forensic evidence packs.

Wraps `forensics.build_evidence_pack` and returns a compact context object
consumable by judges and high-level pipeline code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forensics import build_evidence_pack

GRAPHIC_ARTIFACTS = [
    "mela/mela_heat.png",
    "mela/mela_overlay.png",
    "mela/mela_rois.png",
    "noise/noise_overlay.png",
    "noise/noise_heat.png",
    "copymove/copymove_rois.png",
]


@dataclass
class ForensicContext:
    """Forensic bundle used to condition prompts and multi-image VLM input."""

    mode: str
    evidence_pack: dict[str, Any]
    primary_image_path: Path
    supporting_image_paths: list[Path]

    def to_prompt_section(self) -> str:
        """Return a deterministic evidence section for judge prompts."""
        pack_json = json.dumps(self.evidence_pack, indent=2, ensure_ascii=False)
        return (
            "## Forensic Evidence Pack\n"
            f"Mode: {self.mode}\n"
            "Use these pre-computed signals as high-priority evidence while validating against the receipt image.\n\n"
            f"{pack_json}"
        )


class ForensicPipeline:
    """Runs forensic analysis with mode-specific artifact selection for judges."""

    def __init__(self, output_dir: str | Path = "forensics/evidence"):
        self.output_dir = Path(output_dir)

    def analyze(
        self,
        image_path: str | Path,
        *,
        ocr_txt_path: str | Path | None = None,
        mode: str = "FULL",
    ) -> ForensicContext:
        image_path = Path(image_path)
        mode = mode.upper()

        image_out_dir = self.output_dir / image_path.stem
        pack = build_evidence_pack(
            image_path=image_path,
            output_dir=image_out_dir,
            ocr_txt=ocr_txt_path,
            mode=mode,
        )

        supporting = self._collect_supporting_images(image_out_dir, mode)
        return ForensicContext(
            mode=mode,
            evidence_pack=pack,
            primary_image_path=image_path,
            supporting_image_paths=supporting,
        )

    @staticmethod
    def _collect_supporting_images(image_out_dir: Path, mode: str) -> list[Path]:
        if mode not in {"GRAPHIC", "FULL"}:
            return []

        found: list[Path] = []
        for rel in GRAPHIC_ARTIFACTS:
            p = image_out_dir / rel
            if p.exists():
                found.append(p)
        return found
