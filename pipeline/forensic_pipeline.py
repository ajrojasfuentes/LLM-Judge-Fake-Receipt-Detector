"""
ForensicPipeline — orchestrates modular forensic analysis tools and produces
a rich structured context for the VLM judges.

Flow
----
  1. crop_receipt        — detect & deskew receipt; produce shared grayscale
  2. mela_analyze        — enhanced Multi-Quality ELA with ROIs & percentiles
  3. noise_analyze       — block noise-variance analysis with ROIs
  4. frequency_analyze   — DCT/FFT frequency analysis with ROIs
  5. cpi_analyze         — dense block copy-paste detection gated by above ROIs
  6. OCR extraction      — parse paired .txt transcription; arithmetic check
  7. ForensicContext      — structured result with rich prompt section for judges

Usage
-----
    from pipeline.forensic_pipeline import ForensicPipeline
    fp = ForensicPipeline(output_dir="outputs/forensic")
    ctx = fp.analyze("receipt.png", ocr_txt_path="receipt.txt")
    prompt_text = ctx.to_prompt_section()
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Ensure project root is on the path for cross-module imports
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from forensic.utils import load_image_rgb, crop_receipt, ROI
from forensic.mela import mela_analyze, MELAResult
from forensic.noisemap import noise_analyze, NoiseResult
from forensic.frequencydct import frequency_analyze, FrequencyResult
from forensic.cpi import cpi_analyze, CPIResult, CPIPair


# ─────────────────────────────────────────────────────────────────────────────
# ForensicContext — structured forensic results for the judges
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ForensicContext:
    """
    Structured forensic analysis results for a single receipt image.
    Passed to each judge as additional context to enrich their prompt.
    """

    image_path: str

    # ── Receipt crop / deskew ────────────────────────────────────────────────
    crop_method: Optional[str] = None               # "contour_warp" | "nonwhite_bbox" | "none"
    receipt_area_ratio: Optional[float] = None      # cropped area / original area

    # ── Multi-ELA (enhanced) ─────────────────────────────────────────────────
    multi_ela_suspicious_ratio: Optional[float] = None
    multi_ela_mean_variance: Optional[float] = None
    multi_ela_max_variance: Optional[float] = None
    multi_ela_divergent_blocks: Optional[int] = None
    multi_ela_total_blocks: Optional[int] = None
    multi_ela_blocks_divergent_ratio: Optional[float] = None
    multi_ela_percentiles: Optional[Dict[str, float]] = None   # p50, p75, p90, p95, p99
    multi_ela_tail_mass_pct: Optional[float] = None
    multi_ela_components_count: Optional[int] = None
    multi_ela_largest_component_area_pct: Optional[float] = None
    multi_ela_top_rois: Optional[List[Dict[str, Any]]] = None  # serialisable list
    multi_ela_threshold_method: Optional[str] = None

    # ── Noise map (enhanced) ─────────────────────────────────────────────────
    noise_anomalous_ratio: Optional[float] = None
    noise_anomalous_blocks: Optional[int] = None
    noise_total_blocks: Optional[int] = None
    noise_percentiles: Optional[Dict[str, float]] = None
    noise_top_rois: Optional[List[Dict[str, Any]]] = None

    # ── Frequency / DCT (enhanced) ───────────────────────────────────────────
    freq_anomalous_blocks: Optional[int] = None
    freq_anomalous_ratio: Optional[float] = None
    freq_hf_mean: Optional[float] = None
    freq_percentiles: Optional[Dict[str, float]] = None
    freq_top_rois: Optional[List[Dict[str, Any]]] = None

    # ── CPI — Dense block copy-paste inside ─────────────────────────────────
    cpi_confidence: Optional[float] = None
    cpi_level: Optional[str] = None                            # "LOW" | "MOD" | "HIGH"
    cpi_best_shift_dx: Optional[int] = None
    cpi_best_shift_dy: Optional[int] = None
    cpi_inlier_ratio: Optional[float] = None
    cpi_clone_area_pct: Optional[float] = None
    cpi_num_hypotheses: Optional[int] = None
    cpi_verified_pairs: Optional[int] = None
    cpi_top_pairs: Optional[List[Dict[str, Any]]] = None       # serialisable list

    # ── OCR structured fields ────────────────────────────────────────────────
    ocr_cleaned_text: Optional[str] = None
    ocr_company: Optional[List[str]] = None
    ocr_dates: Optional[List[str]] = None
    ocr_totals: Optional[List[str]] = None
    ocr_items: Optional[List[str]] = None
    ocr_amounts: Optional[List[float]] = None
    ocr_quality: Optional[float] = None

    # ── Arithmetic verification ──────────────────────────────────────────────
    ocr_arithmetic_report: Optional[Dict[str, Any]] = None

    # ── Saved forensic image paths (for optional multi-image judge input) ────
    saved_images: Dict[str, str] = field(default_factory=dict)

    # ── Backward-compat copy-move fields (deprecated) ───────────────────────
    cm_num_matches: Optional[int] = None
    cm_num_clusters: Optional[int] = None
    cm_confidence: Optional[float] = None

    # ── Errors ───────────────────────────────────────────────────────────────
    errors: Dict[str, str] = field(default_factory=dict)

    # ── Prompt formatting ────────────────────────────────────────────────────

    def to_prompt_section(self) -> str:
        """Convert forensic context into a compact, actionable text block for judges."""
        lines: List[str] = []
        lines.append("=== FORENSIC PRE-ANALYSIS (automated signals) ===")
        lines.append(
            "The following signals were computed from the image BEFORE your visual "
            "inspection. Use them to focus your attention on suspicious regions."
        )
        lines.append("")

        # ── Crop info ──
        if self.crop_method is not None:
            ratio_str = f"{self.receipt_area_ratio:.0%}" if self.receipt_area_ratio is not None else "n/a"
            lines.append(
                f"[Receipt Pre-processing]\n"
                f"  Crop method: {self.crop_method} | Receipt area ratio: {ratio_str}"
            )
            lines.append("")

        # ── Multi-ELA ──
        if self.multi_ela_suspicious_ratio is not None:
            level = self._interpret_mela()
            pct = self.multi_ela_percentiles or {}
            p90_str = f"{pct.get('p90', 0):.2f}"
            p95_str = f"{pct.get('p95', 0):.2f}"
            p99_str = f"{pct.get('p99', 0):.2f}"
            div_str = ""
            if self.multi_ela_divergent_blocks is not None and self.multi_ela_total_blocks:
                div_str = (
                    f"\n  Divergent blocks    : {self.multi_ela_divergent_blocks}/"
                    f"{self.multi_ela_total_blocks}"
                    f" ({self.multi_ela_blocks_divergent_ratio or 0:.1%})"
                )
            comp_str = ""
            if self.multi_ela_components_count is not None:
                comp_str = (
                    f"\n  Components          : {self.multi_ela_components_count}"
                    f" | Largest: {self.multi_ela_largest_component_area_pct or 0:.2f}%"
                )
            tail_str = ""
            if self.multi_ela_tail_mass_pct is not None:
                tail_str = f"\n  Tail mass (>p95)    : {self.multi_ela_tail_mass_pct:.2f}%"

            lines.append(
                f"[Multi-ELA — Cross-Quality Compression Variance]: {level}\n"
                f"  Suspicious pixel ratio : {self.multi_ela_suspicious_ratio:.2%}"
                f"{div_str}"
                f"\n  Percentiles (p90/p95/p99): {p90_str} / {p95_str} / {p99_str}"
                f"{tail_str}"
                f"{comp_str}"
            )

            # Top ROIs
            if self.multi_ela_top_rois:
                lines.append("  Top suspicious ROIs:")
                for i, roi in enumerate(self.multi_ela_top_rois[:3]):
                    bbox = roi.get("bbox", [0, 0, 0, 0])
                    score = roi.get("score", 0)
                    notes = roi.get("notes", "")
                    lines.append(
                        f"    ROI-{i+1}: (x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]})"
                        f"  score={score:.2f}  [{notes}]"
                    )

            lines.append(
                f"  → Regions with high cross-quality variance have 'compression memory':\n"
                f"    likely derived from a JPEG source (pasted content, prior editing).\n"
                f"    CPI forgeries leave this signature at paste boundaries."
            )
        else:
            lines.append("[Multi-ELA] Not available.")
        lines.append("")

        # ── Noise Map ──
        if self.noise_anomalous_ratio is not None:
            level = self._interpret_noise()
            pct = self.noise_percentiles or {}
            lines.append(
                f"[Noise Map — Block Variance Analysis]: {level}\n"
                f"  Anomalous blocks : {self.noise_anomalous_blocks}/{self.noise_total_blocks} "
                f"({self.noise_anomalous_ratio:.1%})\n"
                f"  Percentiles (p90/p95): {pct.get('p90', 0):.2f} / {pct.get('p95', 0):.2f}"
            )
            if self.noise_top_rois:
                for i, roi in enumerate(self.noise_top_rois[:2]):
                    bbox = roi.get("bbox", [0, 0, 0, 0])
                    lines.append(
                        f"  ROI-{i+1}: (x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]})"
                    )
            lines.append(
                "  → Anomalous noise blocks indicate regions with different compression\n"
                "    history (e.g., pasted from a different source)."
            )
        else:
            lines.append("[Noise Map] Not available.")
        lines.append("")

        # ── Frequency / DCT ──
        if self.freq_anomalous_blocks is not None:
            level = self._interpret_freq()
            pct = self.freq_percentiles or {}
            lines.append(
                f"[Frequency Analysis (DCT/FFT)]: {level}\n"
                f"  Anomalous DCT blocks : {self.freq_anomalous_blocks} "
                f"({self.freq_anomalous_ratio or 0:.1%})\n"
                f"  Global HF mean ratio : {self.freq_hf_mean:.4f}\n"
                f"  Percentiles (p90/p95): {pct.get('p90', 0):.4f} / {pct.get('p95', 0):.4f}\n"
                f"  → High anomalous count suggests sharp artificial edges or frequency\n"
                f"    discontinuities from text insertion or copy-paste."
            )
        else:
            lines.append("[Frequency Analysis] Not available.")
        lines.append("")

        # ── CPI — Dense Block Copy-Paste ──
        if self.cpi_level is not None:
            level_str = self.cpi_level
            icon = "⚠" if level_str == "HIGH" else ("⚡" if level_str == "MOD" else "✓")
            shift_str = (
                f"dx={self.cpi_best_shift_dx:+d}, dy={self.cpi_best_shift_dy:+d}"
                if (self.cpi_best_shift_dx is not None and self.cpi_best_shift_dy is not None)
                else "n/a"
            )
            conf_str = f"{self.cpi_confidence:.3f}" if self.cpi_confidence is not None else "n/a"
            inlier_str = f"{self.cpi_inlier_ratio:.3f}" if self.cpi_inlier_ratio is not None else "n/a"
            area_str = f"{self.cpi_clone_area_pct:.2f}%" if self.cpi_clone_area_pct is not None else "n/a"
            hyp_str = str(self.cpi_num_hypotheses) if self.cpi_num_hypotheses is not None else "0"
            vp_str = str(self.cpi_verified_pairs) if self.cpi_verified_pairs is not None else "0"
            lines.append(
                f"[Copy-Paste Evidence (Dense Blocks)]: {icon} {level_str}\n"
                f"  Confidence  : {conf_str} | Inlier ratio: {inlier_str}\n"
                f"  Best shift  : {shift_str} | Clone area: {area_str}\n"
                f"  Hypotheses  : {hyp_str} | Verified pairs: {vp_str}"
            )
            if self.cpi_top_pairs:
                for i, pair in enumerate(self.cpi_top_pairs[:3]):
                    dst = pair.get("dest_bbox", [0, 0, 0, 0])
                    src = pair.get("src_bbox", [0, 0, 0, 0])
                    shift = pair.get("shift", [0, 0])
                    ir = pair.get("inlier_ratio", 0)
                    total_z = "TOTAL ZONE" if pair.get("overlaps_total_zone") else ""
                    tax_z = "TAX ZONE" if pair.get("overlaps_tax_zone") else ""
                    zone_note = " | ".join(z for z in [total_z, tax_z] if z)
                    lines.append(
                        f"  Pair-{i+1}: dest=(x={dst[0]},y={dst[1]},w={dst[2]},h={dst[3]}) "
                        f"src=(x={src[0]},y={src[1]},w={src[2]},h={src[3]})"
                        f" shift=({shift[0]:+d},{shift[1]:+d}) inliers={ir:.2f}"
                        + (f" [{zone_note}]" if zone_note else "")
                    )
            if self.cpi_level == "HIGH":
                lines.append(
                    "  → HIGH confidence: region likely copied and re-pasted inside document.\n"
                    "    Check destination ROI carefully — especially if it overlaps totals/prices."
                )
            elif self.cpi_level == "MOD":
                lines.append(
                    "  → MODERATE confidence: some repeated block patterns detected.\n"
                    "    Cross-check with visual inspection of the total/price regions."
                )
            else:
                lines.append("  → LOW: no strong copy-paste evidence found.")
        elif self.cm_confidence is not None:
            # Backward-compat: old ORB-based copy-move
            level = self._interpret_copy_move_legacy()
            lines.append(
                f"[Copy-Move Detection (ORB keypoints)]\n"
                f"  Matches: {self.cm_num_matches} | Clusters: {self.cm_num_clusters} "
                f"| Confidence: {self.cm_confidence:.2f}  {level}"
            )
        else:
            lines.append("[Copy-Paste Detection] Not available.")
        lines.append("")

        # ── OCR ──
        if self.ocr_quality is not None:
            lines.append(
                f"[OCR Transcription (from paired .txt file)]\n"
                f"  Quality score : {self.ocr_quality:.0%}"
            )
            if self.ocr_company:
                lines.append(f"  Store header  : {' | '.join(self.ocr_company[:3])}")
            if self.ocr_dates:
                lines.append(f"  Date(s) found : {', '.join(self.ocr_dates[:3])}")
            if self.ocr_totals:
                lines.append(f"  Total fields  : {', '.join(self.ocr_totals[:4])}")
            if self.ocr_amounts:
                formatted = [f"{a:.2f}" for a in self.ocr_amounts[:10]]
                lines.append(f"  All amounts   : [{', '.join(formatted)}]")
        else:
            lines.append("[OCR Transcription] Not available (no paired .txt file).")
        lines.append("")

        # ── Arithmetic verification ──
        if self.ocr_arithmetic_report is not None:
            lines.append(self._format_arithmetic_report(self.ocr_arithmetic_report))
        else:
            lines.append("[Arithmetic Verification] Not available.")
        lines.append("")

        # ── Errors ──
        if self.errors:
            lines.append(f"[Analysis Errors] {self.errors}")
            lines.append("")

        lines.append("=== END FORENSIC PRE-ANALYSIS ===")
        return "\n".join(lines)

    # ── Interpretation helpers ───────────────────────────────────────────────

    def _interpret_mela(self) -> str:
        r = self.multi_ela_suspicious_ratio or 0.0
        if r > 0.10:
            return "⚠ HIGH — significant cross-quality variance detected"
        elif r > 0.03:
            return "⚡ MODERATE — some suspicious areas"
        return "✓ LOW — consistent compression behaviour"

    def _interpret_noise(self) -> str:
        r = self.noise_anomalous_ratio or 0.0
        if r > 0.15:
            return "⚠ HIGH"
        elif r > 0.05:
            return "⚡ MODERATE"
        return "✓ LOW"

    def _interpret_freq(self) -> str:
        n = self.freq_anomalous_blocks or 0
        ratio = self.freq_anomalous_ratio or 0.0
        if n > 50 or ratio > 0.15:
            return "⚠ HIGH"
        elif n > 20 or ratio > 0.05:
            return "⚡ MODERATE"
        return "✓ LOW"

    def _interpret_copy_move_legacy(self) -> str:
        c = self.cm_confidence or 0.0
        if c > 0.5:
            return "⚠ HIGH — copy-move strongly indicated"
        elif c > 0.2:
            return "⚡ MODERATE"
        return "✓ LOW"

    @staticmethod
    def _format_arithmetic_report(report: Dict[str, Any]) -> str:
        lines = ["[Arithmetic Verification (from OCR)]"]
        item_count = report.get("item_count", 0)
        item_amounts = report.get("item_amounts", [])
        item_sum = report.get("item_sum", 0.0)
        stated_total = report.get("stated_total")
        tax_amount = report.get("tax_amount")
        discrepancy = report.get("discrepancy")
        discrepancy_pct = report.get("discrepancy_pct")
        arithmetic_consistent = report.get("arithmetic_consistent")

        if item_count == 0:
            lines.append("  No item lines with prices could be parsed from OCR.")
            return "\n".join(lines)

        amounts_str = " | ".join(f"{a:.2f}" for a in item_amounts[:8])
        if len(item_amounts) > 8:
            amounts_str += f" ... (+{len(item_amounts) - 8} more)"
        lines.append(f"  Item lines parsed ({item_count}): {amounts_str}")
        lines.append(f"  Sum of item amounts : {item_sum:.2f}")

        if stated_total is not None:
            lines.append(f"  Stated TOTAL field  : {stated_total:.2f}")
        else:
            lines.append("  Stated TOTAL field  : NOT FOUND in OCR")

        if tax_amount is not None:
            lines.append(f"  Tax/GST detected    : {tax_amount:.2f}")

        if discrepancy is not None and stated_total is not None:
            sign = "+" if discrepancy >= 0 else ""
            pct_str = f" ({sign}{discrepancy_pct:.1f}%)" if discrepancy_pct is not None else ""
            lines.append(f"  Difference (total - sum): {sign}{discrepancy:.2f}{pct_str}")
            if arithmetic_consistent:
                lines.append(
                    "  → ✓ ARITHMETIC CONSISTENT (within expected tax/rounding tolerance)."
                )
            else:
                lines.append(
                    "  → ⚠ ARITHMETIC DISCREPANCY — stated total differs from item sum\n"
                    "    by more than expected. Strong signal of a manipulated total."
                )
        else:
            lines.append("  → Could not complete arithmetic cross-check (insufficient data).")

        lines.append(
            "  NOTE: OCR accuracy may be imperfect. Cross-verify with the image."
        )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# ForensicPipeline
# ─────────────────────────────────────────────────────────────────────────────

class ForensicPipeline:
    """
    Orchestrates all modular forensic analysis tools, collects results, and
    produces a ForensicContext ready for injection into the VLM judge prompt.

    Usage
    -----
        fp = ForensicPipeline(output_dir="outputs/forensic")
        ctx = fp.analyze("receipt.png", ocr_txt_path="receipt.txt")
        prompt_text = ctx.to_prompt_section()
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs/forensic",
        # MELA params
        mela_qualities: Tuple[int, ...] = (60, 70, 80, 85, 90, 95),
        mela_block_sizes: Tuple[int, ...] = (8, 16, 32),
        mela_threshold_method: str = "percentile",
        mela_threshold_percentile: float = 98.0,
        mela_edge_suppression_k: float = 0.5,
        mela_max_rois: int = 5,
        # Noise params
        noise_block_size: int = 32,
        noise_max_rois: int = 5,
        # Frequency params
        freq_block_size: int = 32,
        freq_max_rois: int = 5,
        # CPI params
        cpi_block_size: int = 16,
        cpi_stride: int = 8,
        cpi_dbscan_eps: float = 5.0,
        cpi_dbscan_min_samples: int = 20,
        cpi_ncc_threshold: float = 0.85,
        cpi_max_hypotheses: int = 3,
        # General
        save_images: bool = True,
        verbose: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.save_images = save_images
        self.verbose = verbose

        self._mela_qualities = mela_qualities
        self._mela_block_sizes = mela_block_sizes
        self._mela_threshold_method = mela_threshold_method
        self._mela_threshold_percentile = mela_threshold_percentile
        self._mela_edge_suppression_k = mela_edge_suppression_k
        self._mela_max_rois = mela_max_rois

        self._noise_block_size = noise_block_size
        self._noise_max_rois = noise_max_rois

        self._freq_block_size = freq_block_size
        self._freq_max_rois = freq_max_rois

        self._cpi_block_size = cpi_block_size
        self._cpi_stride = cpi_stride
        self._cpi_dbscan_eps = cpi_dbscan_eps
        self._cpi_dbscan_min_samples = cpi_dbscan_min_samples
        self._cpi_ncc_threshold = cpi_ncc_threshold
        self._cpi_max_hypotheses = cpi_max_hypotheses

    # ── Public interface ─────────────────────────────────────────────────────

    def analyze(
        self,
        image_path: Union[str, Path],
        ocr_txt_path: Optional[Union[str, Path]] = None,
    ) -> ForensicContext:
        """Run full forensic analysis on a single receipt image.

        Parameters
        ----------
        image_path : str or Path
            Path to the receipt image (PNG, JPEG, etc.).
        ocr_txt_path : str or Path, optional
            Path to the paired OCR transcription (.txt file).

        Returns
        -------
        ForensicContext
        """
        image_path = Path(image_path)
        ctx = ForensicContext(image_path=str(image_path))
        out_dir = self.output_dir / image_path.stem if self.save_images else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # 0. Load image
        # ------------------------------------------------------------------
        try:
            image_rgb = load_image_rgb(image_path)
        except Exception as exc:
            ctx.errors["image_load"] = f"{type(exc).__name__}: {exc}"
            return ctx

        prefix = image_path.stem

        # ------------------------------------------------------------------
        # 1. Crop / deskew
        # ------------------------------------------------------------------
        try:
            crop_result = crop_receipt(image_rgb, output_dir=out_dir, prefix=prefix)
            ctx.crop_method = crop_result.crop_method
            ctx.receipt_area_ratio = crop_result.receipt_area_ratio
            ctx.saved_images.update({
                f"crop_{k}": v for k, v in crop_result.saved_images.items()
            })
            cropped_rgb = crop_result.cropped_rgb
            cropped_gray = crop_result.cropped_gray
        except Exception as exc:
            ctx.errors["crop"] = f"{type(exc).__name__}: {exc}"
            cropped_rgb = image_rgb
            cropped_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            ctx.crop_method = "none"

        # ------------------------------------------------------------------
        # 2. Multi-ELA
        # ------------------------------------------------------------------
        mela_result: Optional[MELAResult] = None
        try:
            mela_result = mela_analyze(
                image_rgb=cropped_rgb,
                cropped_gray=cropped_gray,
                output_dir=out_dir,
                prefix=prefix,
                qualities=self._mela_qualities,
                block_sizes=self._mela_block_sizes,
                threshold_method=self._mela_threshold_method,
                threshold_percentile=self._mela_threshold_percentile,
                edge_suppression_k=self._mela_edge_suppression_k,
                max_rois=self._mela_max_rois,
            )
            ctx.multi_ela_suspicious_ratio = mela_result.suspicious_ratio
            ctx.multi_ela_mean_variance = mela_result.mean_variance
            ctx.multi_ela_max_variance = mela_result.max_variance
            ctx.multi_ela_divergent_blocks = mela_result.divergent_blocks
            ctx.multi_ela_total_blocks = mela_result.total_blocks
            ctx.multi_ela_blocks_divergent_ratio = mela_result.blocks_divergent_ratio
            ctx.multi_ela_percentiles = mela_result.percentiles
            ctx.multi_ela_tail_mass_pct = mela_result.tail_mass_pct
            ctx.multi_ela_components_count = mela_result.components_count
            ctx.multi_ela_largest_component_area_pct = mela_result.largest_component_area_pct
            ctx.multi_ela_threshold_method = mela_result.threshold_method
            ctx.multi_ela_top_rois = [
                {
                    "bbox": list(r.bbox),
                    "score": round(r.score, 4),
                    "area_pct": round(r.area_pct, 4),
                    "source": r.source,
                    "notes": r.notes,
                }
                for r in mela_result.top_rois
            ]
            ctx.saved_images.update({
                f"mela_{k}": v for k, v in mela_result.saved_images.items()
            })
        except Exception as exc:
            ctx.errors["mela"] = f"{type(exc).__name__}: {exc}"
            if self.verbose:
                print(f"[forensic] MELA error: {exc}")

        # ------------------------------------------------------------------
        # 3. Noise map
        # ------------------------------------------------------------------
        noise_result: Optional[NoiseResult] = None
        try:
            noise_result = noise_analyze(
                cropped_gray=cropped_gray,
                output_dir=out_dir,
                prefix=prefix,
                block_size=self._noise_block_size,
                max_rois=self._noise_max_rois,
            )
            ctx.noise_anomalous_blocks = noise_result.anomalous_blocks
            ctx.noise_total_blocks = noise_result.total_blocks
            ctx.noise_anomalous_ratio = noise_result.anomalous_ratio
            ctx.noise_percentiles = noise_result.percentiles
            ctx.noise_top_rois = [
                {
                    "bbox": list(r.bbox),
                    "score": round(r.score, 4),
                    "area_pct": round(r.area_pct, 4),
                    "source": r.source,
                    "notes": r.notes,
                }
                for r in noise_result.top_rois
            ]
            ctx.saved_images.update({
                f"noise_{k}": v for k, v in noise_result.saved_images.items()
            })
        except Exception as exc:
            ctx.errors["noise"] = f"{type(exc).__name__}: {exc}"
            if self.verbose:
                print(f"[forensic] Noise error: {exc}")

        # ------------------------------------------------------------------
        # 4. Frequency / DCT
        # ------------------------------------------------------------------
        freq_result: Optional[FrequencyResult] = None
        try:
            freq_result = frequency_analyze(
                cropped_gray=cropped_gray,
                output_dir=out_dir,
                prefix=prefix,
                block_size=self._freq_block_size,
                max_rois=self._freq_max_rois,
            )
            ctx.freq_anomalous_blocks = freq_result.anomalous_blocks
            ctx.freq_anomalous_ratio = freq_result.anomalous_ratio
            ctx.freq_hf_mean = freq_result.global_hf_mean
            ctx.freq_percentiles = freq_result.percentiles
            ctx.freq_top_rois = [
                {
                    "bbox": list(r.bbox),
                    "score": round(r.score, 4),
                    "area_pct": round(r.area_pct, 4),
                    "source": r.source,
                    "notes": r.notes,
                }
                for r in freq_result.top_rois
            ]
            ctx.saved_images.update({
                f"freq_{k}": v for k, v in freq_result.saved_images.items()
            })
        except Exception as exc:
            ctx.errors["freq"] = f"{type(exc).__name__}: {exc}"
            if self.verbose:
                print(f"[forensic] Frequency error: {exc}")

        # ------------------------------------------------------------------
        # 5. OCR extraction
        # ------------------------------------------------------------------
        ocr_text: Optional[str] = None
        if ocr_txt_path is not None:
            try:
                ocr_result = self._extract_ocr(Path(ocr_txt_path))
                ctx.ocr_quality = ocr_result["quality_score"]
                ctx.ocr_cleaned_text = ocr_result["cleaned_text"]
                ctx.ocr_company = ocr_result["company_lines"]
                ctx.ocr_dates = ocr_result["date_candidates"]
                ctx.ocr_totals = ocr_result["total_candidates"]
                ctx.ocr_items = ocr_result["item_candidates"]
                ctx.ocr_amounts = ocr_result["all_amounts"]
                ctx.ocr_arithmetic_report = self._compute_arithmetic_report(ocr_result)
                ocr_text = ocr_result.get("cleaned_text")
            except Exception as exc:
                ctx.errors["ocr"] = f"{type(exc).__name__}: {exc}"
                if self.verbose:
                    print(f"[forensic] OCR error: {exc}")

        # ------------------------------------------------------------------
        # 6. CPI — dense block copy-paste inside
        # ------------------------------------------------------------------
        try:
            cpi_result = cpi_analyze(
                cropped_gray=cropped_gray,
                mela_rois=mela_result.top_rois if mela_result else None,
                noise_rois=noise_result.top_rois if noise_result else None,
                freq_rois=freq_result.top_rois if freq_result else None,
                mela_suspicious_ratio=mela_result.suspicious_ratio if mela_result else 0.0,
                noise_anomalous_ratio=(
                    noise_result.anomalous_ratio if noise_result else 0.0
                ),
                freq_anomalous_ratio=freq_result.anomalous_ratio if freq_result else 0.0,
                ocr_text=ocr_text,
                output_dir=out_dir,
                prefix=prefix,
                block_size=self._cpi_block_size,
                stride=self._cpi_stride,
                dbscan_eps=self._cpi_dbscan_eps,
                dbscan_min_samples=self._cpi_dbscan_min_samples,
                ncc_threshold=self._cpi_ncc_threshold,
                max_hypotheses=self._cpi_max_hypotheses,
            )
            ctx.cpi_confidence = cpi_result.confidence
            ctx.cpi_level = cpi_result.level
            if cpi_result.best_shift is not None:
                ctx.cpi_best_shift_dx, ctx.cpi_best_shift_dy = cpi_result.best_shift
            ctx.cpi_inlier_ratio = cpi_result.inlier_ratio
            ctx.cpi_clone_area_pct = cpi_result.clone_area_pct
            ctx.cpi_num_hypotheses = cpi_result.num_hypotheses
            ctx.cpi_verified_pairs = cpi_result.verified_pairs
            ctx.cpi_top_pairs = [
                {
                    "dest_bbox": list(p.dest_bbox),
                    "src_bbox": list(p.src_bbox),
                    "shift": list(p.shift),
                    "inlier_ratio": round(p.inlier_ratio, 4),
                    "mean_ncc": round(p.mean_ncc, 4),
                    "verified_pairs": p.verified_pairs,
                    "clone_area_pct": round(p.clone_area_pct, 4),
                    "overlaps_total_zone": p.overlaps_total_zone,
                    "overlaps_tax_zone": p.overlaps_tax_zone,
                    "cluster_size": p.cluster_size,
                    "penalised": p.penalised,
                }
                for p in cpi_result.top_pairs
            ]
            # Backward-compat aliases
            ctx.cm_confidence = cpi_result.confidence
            ctx.cm_num_clusters = cpi_result.num_hypotheses
            ctx.cm_num_matches = cpi_result.verified_pairs
            ctx.saved_images.update({
                f"cpi_{k}": v for k, v in cpi_result.saved_images.items()
            })
        except Exception as exc:
            ctx.errors["cpi"] = f"{type(exc).__name__}: {exc}"
            if self.verbose:
                print(f"[forensic] CPI error: {exc}")

        return ctx

    def analyze_batch(
        self,
        receipts: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> Dict[str, ForensicContext]:
        """Run forensic analysis on a batch of receipts.

        Parameters
        ----------
        receipts : list of dict
            Each dict must have "id", "image_path", and optionally "ocr_txt_path".
        verbose : bool
            Print progress.

        Returns
        -------
        Dict mapping receipt ID → ForensicContext.
        """
        results: Dict[str, ForensicContext] = {}
        total = len(receipts)

        for idx, receipt in enumerate(receipts):
            rid = receipt["id"]
            img_path = receipt.get("image_path")
            txt_path = receipt.get("ocr_txt_path")

            if verbose:
                print(f"  [forensic] [{idx+1}/{total}] {rid} ...", end=" ", flush=True)

            if img_path is None:
                ctx = ForensicContext(image_path="")
                ctx.errors["image_path"] = "image_path is None — image not found on disk"
                results[rid] = ctx
                if verbose:
                    print("SKIPPED (no image)")
                continue

            ctx = self.analyze(img_path, ocr_txt_path=txt_path)
            results[rid] = ctx

            if verbose:
                signals = []
                if ctx.multi_ela_suspicious_ratio is not None:
                    signals.append(f"MELA={ctx.multi_ela_suspicious_ratio:.0%}")
                if ctx.cpi_level is not None:
                    signals.append(f"CPI={ctx.cpi_level}({ctx.cpi_confidence:.2f})")
                elif ctx.cm_confidence is not None:
                    signals.append(f"CM={ctx.cm_confidence:.2f}")
                if ctx.ocr_arithmetic_report is not None:
                    consistent = ctx.ocr_arithmetic_report.get("arithmetic_consistent")
                    if consistent is False:
                        signals.append("ARITH=⚠")
                    elif consistent is True:
                        signals.append("ARITH=✓")
                if ctx.errors:
                    signals.append(f"errors={list(ctx.errors.keys())}")
                print(" | ".join(signals) or "ok")

        return results

    # ── OCR helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_ocr(txt_path: Path) -> Dict[str, Any]:
        """Load and structure the paired OCR text file."""
        raw = txt_path.read_text(encoding="utf-8", errors="replace")

        # Cleaning: normalise whitespace, remove control characters
        cleaned = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", " ", raw)
        cleaned = re.sub(r" {2,}", " ", cleaned).strip()

        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]

        # Quality score: ratio of alphanumeric+punctuation chars
        total_chars = max(len(cleaned), 1)
        useful_chars = sum(1 for c in cleaned if c.isalnum() or c in ".,:-/$%")
        quality_score = min(useful_chars / total_chars * 1.5, 1.0)

        # Structural bonus
        has_total = any(any(k in ln.lower() for k in ("total", "amount", "nett")) for ln in lines)
        has_date = bool(re.search(r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}", cleaned))
        has_items = sum(1 for ln in lines if re.search(r"\d+\.\d{2}", ln)) > 2
        quality_score = min(quality_score + 0.05 * has_total + 0.05 * has_date + 0.05 * has_items, 1.0)

        # Company: first non-empty lines before dates/amounts appear
        company_lines: List[str] = []
        price_re = re.compile(r"\d+\.\d{2}")
        date_re = re.compile(r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}")
        for ln in lines[:6]:
            if price_re.search(ln) or date_re.search(ln):
                break
            company_lines.append(ln)

        # Dates
        date_candidates: List[str] = []
        for ln in lines:
            if date_re.search(ln):
                date_candidates.append(ln[:60])

        # Totals
        total_kws = ("total", "grand", "subtotal", "amount", "nett", "net", "tax", "gst", "vat")
        total_candidates: List[str] = []
        for ln in lines:
            if any(k in ln.lower() for k in total_kws) and price_re.search(ln):
                total_candidates.append(ln[:80])

        # Item lines: lines with a price that are NOT total lines
        item_candidates: List[str] = []
        for ln in lines:
            if price_re.search(ln) and not any(k in ln.lower() for k in total_kws):
                item_candidates.append(ln[:80])

        # All monetary amounts
        all_amounts: List[float] = sorted(
            set(float(m) for m in price_re.findall(cleaned)),
        )

        return {
            "quality_score": quality_score,
            "cleaned_text": cleaned[:4000],   # cap for prompt size
            "company_lines": company_lines[:5],
            "date_candidates": date_candidates[:5],
            "total_candidates": total_candidates[:8],
            "item_candidates": item_candidates[:20],
            "all_amounts": all_amounts[:20],
        }

    @staticmethod
    def _compute_arithmetic_report(structured: Dict[str, Any]) -> Dict[str, Any]:
        """Compute arithmetic verification from structured OCR fields."""
        price_re = re.compile(r"\d+\.\d{2}")

        # Item amounts: rightmost price on each item line
        item_amounts: List[float] = []
        for line in structured.get("item_candidates", []):
            amounts = [float(a) for a in price_re.findall(line)]
            if amounts:
                item_amounts.append(amounts[-1])

        # Stated total: prefer lines with "total" / "amount"
        total_priority = ["grand total", "total", "amount due", "amount", "nett", "net"]
        stated_total: Optional[float] = None
        total_candidates = structured.get("total_candidates", [])

        for keyword in total_priority:
            for line in reversed(total_candidates):
                if keyword.lower() in line.lower():
                    amounts = [float(a) for a in price_re.findall(line)]
                    if amounts:
                        stated_total = amounts[-1]
                        break
            if stated_total is not None:
                break

        # Tax amount
        tax_amount: Optional[float] = None
        for line in total_candidates:
            if any(k in line.lower() for k in ("tax", "gst", "vat")):
                amounts = [float(a) for a in price_re.findall(line)]
                if amounts:
                    tax_amount = amounts[-1]
                    break

        item_sum = round(sum(item_amounts), 2)
        discrepancy: Optional[float] = None
        discrepancy_pct: Optional[float] = None
        arithmetic_consistent: Optional[bool] = None

        if stated_total is not None and item_sum > 0:
            # Subtract tax from the discrepancy when a tax line is present
            effective_sum = round(item_sum + (tax_amount or 0.0), 2)
            discrepancy = round(stated_total - effective_sum, 2)
            discrepancy_pct = (
                round(abs(discrepancy) / stated_total * 100, 1) if stated_total > 0 else None
            )
            # Tolerance of 10% to allow for minor rounding; >10% is suspicious
            arithmetic_consistent = (discrepancy_pct is None or discrepancy_pct < 10.0)

        return {
            "item_count": len(item_amounts),
            "item_amounts": [round(a, 2) for a in item_amounts[:10]],
            "item_sum": item_sum,
            "stated_total": stated_total,
            "tax_amount": tax_amount,
            "discrepancy": discrepancy,
            "discrepancy_pct": discrepancy_pct,
            "arithmetic_consistent": arithmetic_consistent,
        }
