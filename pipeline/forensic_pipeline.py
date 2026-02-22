"""Compatibility adapter over the active `forensics` module.

Legacy callers in `pipeline/*` can keep importing `ForensicPipeline`, but the
implementation now delegates to `forensics.pipeline.build_evidence_pack`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from forensics.pipeline import build_evidence_pack


@dataclass
class ForensicContext:
    receipt_id: str
    mode: str
    evidence_pack: Dict[str, Any]

    @property
    def errors(self) -> Dict[str, str]:
        return self.evidence_pack.get("errors", {})

    def to_prompt_section(self) -> str:
        pack = self.evidence_pack
        lines = [
            "=== FORENSIC PRE-ANALYSIS (SYSTEM-GENERATED) ===",
            f"Mode: {self.mode.upper()}",
        ]

        if "forensic" in pack:
            fz = pack["forensic"]
            mela = fz.get("mela", {})
            noise = fz.get("noise", {})
            cm = fz.get("copy_move", {})
            lines += [
                f"MELA suspicious_ratio: {mela.get('suspicious_ratio', 'n/a')}",
                f"Noise max_tile_z: {noise.get('max_tile_z', 'n/a')}",
                f"Copy-move pairs_found: {cm.get('pairs_found', 'n/a')}",
            ]

        reading = pack.get("reading")
        if isinstance(reading, dict):
            ar = reading.get("arithmetic_report") or {}
            lines += [
                f"OCR quality_score: {reading.get('quality_score', 'n/a')}",
                f"Arithmetic consistent: {ar.get('arithmetic_consistent', 'n/a')}",
                f"Arithmetic best_explanation: {ar.get('best_explanation', 'n/a')}",
            ]

        if self.errors:
            lines.append(f"Errors: {self.errors}")

        lines.append("=== END FORENSIC PRE-ANALYSIS ===")
        return "\n".join(lines)


class ForensicPipeline:
    def __init__(self, output_dir: str = "forensics/evidence", save_images: bool = True, verbose: bool = False):
        self.output_dir = output_dir
        self.save_images = save_images
        self.verbose = verbose

    def analyze(self, image_path: Path, ocr_txt_path: Optional[str | Path] = None, mode: str = "full") -> ForensicContext:
        pack = build_evidence_pack(
            image_path=image_path,
            output_dir=None,  # default forensics/evidence/<image_id>
            ocr_txt=ocr_txt_path,
            mode=mode,
        )
        return ForensicContext(
            receipt_id=Path(image_path).stem,
            mode=mode,
            evidence_pack=pack,
        )
