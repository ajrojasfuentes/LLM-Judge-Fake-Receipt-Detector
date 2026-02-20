"""
ForensicPipeline: Pre-processing layer that runs ForensicAnalyzer before the LLM judges.

This module bridges the forensic_analysis.py toolkit (image-level signal extraction)
with the multi-judge voting pipeline. It produces a structured "forensic context" block
that is appended to the judge's text prompt, providing amplified forgery signals to
the VLMs before they inspect the image.

=== Why this matters for the Find-It-Again dataset ===
The dominant forgery type (CPI — Copy-Paste Inside, ~78%) leaves subtle pixel-level
and frequency-domain traces that are difficult to spot by visual inspection alone:
  • Multi-ELA reveals regions with abnormal cross-quality compression variance.
  • Noise map highlights blocks with anomalous variance (likely pasted regions).
  • DCT/FFT exposes high-frequency anomalies typical of copy-paste boundaries.
  • Copy-move detection specifically targets CPI patterns.
  • OCR extraction + arithmetic verification cross-checks stated amounts.

These signals are converted to a textual "FORENSIC PRE-ANALYSIS" section that
is prepended to the judge's existing prompt.

=== Usage ===
    from pipeline.forensic_pipeline import ForensicPipeline
    fp = ForensicPipeline()
    context = fp.analyze(image_path, ocr_txt_path)   # → ForensicContext
    prompt_suffix = context.to_prompt_section()       # → str for appending to prompt
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ForensicAnalyzer lives at the project root
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Data class — structured forensic context for the judges
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ForensicContext:
    """
    Structured forensic analysis results for a single receipt image.
    Passed to each judge as additional context to enrich their prompt.
    """

    image_path: str

    # Multi-ELA metrics (cross-quality compression variance)
    multi_ela_suspicious_ratio: Optional[float] = None   # fraction of pixels above variance threshold
    multi_ela_mean_variance: Optional[float] = None      # global mean cross-quality variance
    multi_ela_max_variance: Optional[float] = None       # maximum per-pixel variance
    multi_ela_divergent_blocks: Optional[int] = None     # blocks with variance > 2σ
    multi_ela_total_blocks: Optional[int] = None

    # Noise map metrics
    noise_anomalous_ratio: Optional[float] = None   # fraction of blocks anomalous
    noise_anomalous_blocks: Optional[int] = None
    noise_total_blocks: Optional[int] = None

    # Frequency (DCT) metrics
    freq_anomalous_blocks: Optional[int] = None
    freq_hf_mean: Optional[float] = None

    # Copy-move metrics
    cm_num_matches: Optional[int] = None
    cm_num_clusters: Optional[int] = None
    cm_confidence: Optional[float] = None           # 0.0–1.0

    # OCR structured fields (from the paired .txt file)
    ocr_cleaned_text: Optional[str] = None
    ocr_company: Optional[List[str]] = None
    ocr_dates: Optional[List[str]] = None
    ocr_totals: Optional[List[str]] = None
    ocr_items: Optional[List[str]] = None
    ocr_amounts: Optional[List[float]] = None
    ocr_quality: Optional[float] = None             # 0.0–1.0

    # Arithmetic verification report (computed from OCR structured fields)
    ocr_arithmetic_report: Optional[Dict[str, Any]] = None

    # Any errors during forensic analysis
    errors: Dict[str, str] = field(default_factory=dict)

    def to_prompt_section(self) -> str:
        """
        Convert the forensic context into a text block for inclusion in the judge prompt.

        The output is formatted as a clear, structured section that the VLM can parse
        alongside the image to calibrate its attention on suspicious regions.
        """
        lines: List[str] = []
        lines.append("=== FORENSIC PRE-ANALYSIS (computed signals) ===")
        lines.append(
            "The following signals were automatically computed from the image "
            "BEFORE your visual inspection. Use them to focus your attention."
        )
        lines.append("")

        # ── Multi-ELA ──
        if self.multi_ela_suspicious_ratio is not None:
            mela_level = self._interpret_multi_ela()
            divergent_info = ""
            if self.multi_ela_divergent_blocks is not None and self.multi_ela_total_blocks:
                divergent_info = (
                    f"\n  Divergent blocks    : {self.multi_ela_divergent_blocks}/"
                    f"{self.multi_ela_total_blocks}"
                )
            lines.append(
                f"[Multi-ELA — Cross-Quality Compression Variance]\n"
                f"  Suspicious pixel ratio : {self.multi_ela_suspicious_ratio:.1%}  {mela_level}\n"
                f"  Mean cross-q variance  : {self.multi_ela_mean_variance:.3f}\n"
                f"  Max cross-q variance   : {self.multi_ela_max_variance:.3f}"
                f"{divergent_info}\n"
                f"  → Pixels/blocks with high cross-quality variance have 'compression memory':\n"
                f"    they were likely derived from a JPEG source (e.g. pasted from another\n"
                f"    document or edited in software that quantised pixel values).\n"
                f"    CPI forgeries (Copy-Paste Inside) may leave this signature at paste\n"
                f"    boundaries even in PNG files."
            )
        else:
            lines.append("[Multi-ELA] Not available.")

        lines.append("")

        # ── Noise Map ──
        if self.noise_anomalous_ratio is not None:
            noise_level = self._interpret_noise()
            lines.append(
                f"[Noise Map]\n"
                f"  Anomalous blocks : {self.noise_anomalous_blocks}/{self.noise_total_blocks} "
                f"({self.noise_anomalous_ratio:.1%})  {noise_level}\n"
                f"  → Anomalous noise blocks may indicate regions with different compression "
                f"history (e.g., pasted from a different source)."
            )
        else:
            lines.append("[Noise Map] Not available.")

        lines.append("")

        # ── Frequency (DCT) ──
        if self.freq_anomalous_blocks is not None:
            freq_level = self._interpret_freq()
            lines.append(
                f"[Frequency Analysis (DCT)]\n"
                f"  Anomalous DCT blocks : {self.freq_anomalous_blocks}  {freq_level}\n"
                f"  Global HF mean ratio : {self.freq_hf_mean:.4f}\n"
                f"  → High anomalous block count suggests sharp artificial edges or "
                f"frequency discontinuities from text insertion."
            )
        else:
            lines.append("[Frequency Analysis] Not available.")

        lines.append("")

        # ── Copy-Move Detection ──
        if self.cm_num_matches is not None:
            cm_level = self._interpret_copy_move()
            lines.append(
                f"[Copy-Move Detection (ORB keypoint matching)]\n"
                f"  Keypoint matches  : {self.cm_num_matches}\n"
                f"  Matched clusters  : {self.cm_num_clusters}\n"
                f"  Detection conf.   : {self.cm_confidence:.2f}  {cm_level}\n"
                f"  → Thresholds calibrated for receipt text (200+ matches, 5+ clusters\n"
                f"    required for high confidence). A high score is genuine evidence\n"
                f"    of CPI forgery (a region was copied and pasted within the document)."
            )
        else:
            lines.append("[Copy-Move Detection] Not available.")

        lines.append("")

        # ── OCR Text ──
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
            lines.append("[OCR Transcription] Not available (no paired .txt file found).")

        lines.append("")

        # ── Arithmetic Verification ──
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

    # ── Interpretation helpers ──

    def _interpret_multi_ela(self) -> str:
        r = self.multi_ela_suspicious_ratio or 0.0
        if r > 0.10:
            return "⚠ HIGH — significant cross-quality variance detected"
        elif r > 0.03:
            return "⚡ MODERATE — some suspicious areas"
        else:
            return "✓ LOW — consistent compression behaviour"

    def _interpret_noise(self) -> str:
        r = self.noise_anomalous_ratio or 0.0
        if r > 0.15:
            return "⚠ HIGH"
        elif r > 0.05:
            return "⚡ MODERATE"
        else:
            return "✓ LOW"

    def _interpret_freq(self) -> str:
        n = self.freq_anomalous_blocks or 0
        if n > 50:
            return "⚠ HIGH"
        elif n > 20:
            return "⚡ MODERATE"
        else:
            return "✓ LOW"

    def _interpret_copy_move(self) -> str:
        c = self.cm_confidence or 0.0
        if c > 0.5:
            return "⚠ HIGH — genuine copy-move strongly indicated"
        elif c > 0.2:
            return "⚡ MODERATE"
        else:
            return "✓ LOW"

    @staticmethod
    def _format_arithmetic_report(report: Dict[str, Any]) -> str:
        """Format the arithmetic verification report for the judge prompt."""
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

        # Item amounts list
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
            lines.append(
                f"  Difference (total - sum): {sign}{discrepancy:.2f}{pct_str}"
            )

            if arithmetic_consistent:
                lines.append(
                    "  → ✓ ARITHMETIC CONSISTENT (within expected tax/rounding tolerance).\n"
                    "    The stated total is plausible given the item amounts."
                )
            else:
                lines.append(
                    f"  → ⚠ ARITHMETIC DISCREPANCY — stated total differs from item sum by more\n"
                    f"    than expected. This is a strong signal of a manipulated total or\n"
                    f"    inserted/deleted line items. Verify each line item carefully."
                )
        else:
            lines.append(
                "  → Could not complete arithmetic cross-check (insufficient OCR data)."
            )

        lines.append(
            "  NOTE: OCR accuracy may be imperfect. Use these numbers as guidance,\n"
            "  not as definitive ground truth — cross-verify with what you see in\n"
            "  the image."
        )

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline class
# ─────────────────────────────────────────────────────────────────────────────

class ForensicPipeline:
    """
    Wrapper around ForensicAnalyzer that produces ForensicContext objects.

    Usage:
        fp = ForensicPipeline(output_dir="outputs/forensic")
        context = fp.analyze("path/to/receipt.png", ocr_txt_path="path/to/receipt.txt")
        prompt_section = context.to_prompt_section()

    The output_dir is used to save intermediate forensic images (Multi-ELA maps,
    noise maps, etc.) which can be inspected or passed as additional images to
    multimodal models.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs/forensic",
        # Multi-ELA params
        mela_qualities: tuple = (70, 85, 95),
        mela_block_size: int = 16,
        mela_variance_threshold: float = 5.0,
        # Noise params
        noise_block_size: int = 32,
        freq_block_size: int = 32,
        # Copy-move params (calibrated to suppress receipt-text false positives)
        orb_features: int = 3000,
        match_threshold: float = 0.55,
        min_match_distance: float = 150.0,
        cluster_eps: float = 30.0,
        min_cluster_size: int = 8,
        save_images: bool = True,
        verbose: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.save_images = save_images
        self.verbose = verbose

        try:
            from forensic_analysis import ForensicAnalyzer
            self._analyzer = ForensicAnalyzer(
                output_dir=self.output_dir,
                mela_qualities=mela_qualities,
                mela_block_size=mela_block_size,
                mela_variance_threshold=mela_variance_threshold,
                noise_block_size=noise_block_size,
                freq_block_size=freq_block_size,
                orb_features=orb_features,
                match_threshold=match_threshold,
                min_match_distance=min_match_distance,
                cluster_eps=cluster_eps,
                min_cluster_size=min_cluster_size,
            )
        except ImportError:
            if verbose:
                print("[forensic] WARNING: forensic_analysis.py not found. "
                      "Forensic context will be empty.")
            self._analyzer = None

    def analyze(
        self,
        image_path: Union[str, Path],
        ocr_txt_path: Optional[Union[str, Path]] = None,
    ) -> ForensicContext:
        """
        Run forensic analysis on a single receipt image.

        Args:
            image_path: Path to the PNG receipt image.
            ocr_txt_path: Path to the paired OCR .txt transcription (optional).
                          When provided, structured text extraction and arithmetic
                          verification are included.

        Returns:
            ForensicContext with all computed signals and interpretations.
        """
        image_path = Path(image_path)
        ctx = ForensicContext(image_path=str(image_path))

        if self._analyzer is None:
            ctx.errors["forensic_analyzer"] = "ForensicAnalyzer not available"
            return ctx

        try:
            report = self._analyzer.full_analysis(
                image_path=image_path,
                ocr_txt_path=ocr_txt_path,
                save=self.save_images,
            )

            # ── Populate Multi-ELA ──
            if report.multi_ela is not None:
                ctx.multi_ela_suspicious_ratio = report.multi_ela.suspicious_ratio
                ctx.multi_ela_mean_variance = report.multi_ela.mean_variance
                ctx.multi_ela_max_variance = report.multi_ela.max_variance
                ctx.multi_ela_divergent_blocks = report.multi_ela.divergent_blocks
                ctx.multi_ela_total_blocks = report.multi_ela.total_blocks

            # ── Populate Noise ──
            if report.noise is not None:
                total = max(1, report.noise.total_blocks)
                ctx.noise_anomalous_blocks = report.noise.anomalous_blocks
                ctx.noise_total_blocks = report.noise.total_blocks
                ctx.noise_anomalous_ratio = report.noise.anomalous_blocks / total

            # ── Populate Frequency ──
            if report.frequency is not None:
                ctx.freq_anomalous_blocks = report.frequency.anomalous_blocks
                ctx.freq_hf_mean = report.frequency.global_hf_mean

            # ── Populate Copy-Move ──
            if report.copy_move is not None:
                ctx.cm_num_matches = report.copy_move.num_matches
                ctx.cm_num_clusters = report.copy_move.num_clusters
                ctx.cm_confidence = report.copy_move.confidence

            # ── Populate OCR ──
            if report.ocr is not None:
                ctx.ocr_quality = report.ocr.quality_score
                ctx.ocr_cleaned_text = report.ocr.cleaned_text
                structured = report.ocr.structured
                ctx.ocr_company = structured.get("company_lines", [])
                ctx.ocr_dates = structured.get("date_candidates", [])
                ctx.ocr_totals = structured.get("total_candidates", [])
                ctx.ocr_items = structured.get("item_candidates", [])
                ctx.ocr_amounts = structured.get("all_amounts", [])

                # ── Arithmetic verification ──
                ctx.ocr_arithmetic_report = self._compute_arithmetic_report(structured)

            # ── Propagate errors ──
            ctx.errors = dict(report.errors)

        except Exception as exc:
            ctx.errors["full_analysis"] = f"{type(exc).__name__}: {exc}"
            if self.verbose:
                print(f"[forensic] ERROR during analysis of {image_path}: {exc}")

        return ctx

    def analyze_batch(
        self,
        receipts: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> Dict[str, ForensicContext]:
        """
        Run forensic analysis on a batch of receipts.

        Args:
            receipts: List of sample dicts (as returned by ReceiptSampler.load()).
                      Each dict must have "id", "image_path", and optionally "ocr_txt_path".
            verbose: Print progress.

        Returns:
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
                if ctx.cm_confidence is not None:
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

    # ── Arithmetic verification helper ────────────────────────────────────────

    @staticmethod
    def _compute_arithmetic_report(structured: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute an arithmetic verification report from structured OCR fields.

        Extracts item prices from item_candidates, identifies the stated total
        from total_candidates, computes the sum, and flags discrepancies.

        Returns a dict with keys:
            item_count, item_amounts, item_sum, stated_total, tax_amount,
            discrepancy, discrepancy_pct, arithmetic_consistent.
        """
        price_re = re.compile(r"\d+\.\d{2}")

        # ── Item amounts: rightmost price on each item line ──
        item_amounts: List[float] = []
        for line in structured.get("item_candidates", []):
            amounts = [float(a) for a in price_re.findall(line)]
            if amounts:
                item_amounts.append(amounts[-1])

        # ── Stated total: prefer lines with "total" / "amount" ──
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

        # ── Tax amount (optional) ──
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
            discrepancy = round(stated_total - item_sum, 2)
            discrepancy_pct = round(abs(discrepancy) / stated_total * 100, 1) if stated_total > 0 else None
            # Allow up to 25% difference (accommodates tax, service charge, rounding)
            arithmetic_consistent = (discrepancy_pct is None or discrepancy_pct < 25.0)

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
