"""
ForensicPipeline: Pre-processing layer that runs ForensicAnalyzer before the LLM judges.

This module bridges the forensic_analysis.py toolkit (image-level signal extraction)
with the multi-judge voting pipeline. It produces a structured "forensic context" block
that is appended to the judge's text prompt, providing amplified forgery signals to
the VLMs before they inspect the image.

=== Why this matters for the Find-It-Again dataset ===
The dominant forgery type (CPI — Copy-Paste Inside, ~78%) leaves subtle pixel-level
and frequency-domain traces that are difficult to spot by visual inspection alone:
  • ELA reveals areas with inconsistent JPEG recompression artifacts.
  • Noise map highlights blocks with anomalous variance (likely pasted regions).
  • DCT/FFT exposes high-frequency anomalies typical of copy-paste boundaries.
  • Copy-move detection specifically targets CPI patterns.
  • OCR extraction provides the structured text for arithmetic cross-checking.

These signals are converted to a textual "FORENSIC PRE-ANALYSIS" section that
is prepended to the judge's existing prompt.

=== Usage ===
    from pipeline.forensic_pipeline import ForensicPipeline
    fp = ForensicPipeline()
    context = fp.analyze(image_path, ocr_txt_path)   # → ForensicContext
    prompt_suffix = context.to_prompt_section()       # → str for appending to prompt
"""

from __future__ import annotations

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

    # ELA metrics
    ela_mean_error: Optional[float] = None
    ela_suspicious_ratio: Optional[float] = None
    ela_max_error: Optional[float] = None

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

        # ── ELA ──
        if self.ela_mean_error is not None:
            ela_level = self._interpret_ela()
            lines.append(
                f"[ELA — Error Level Analysis]\n"
                f"  Suspicious pixel ratio : {self.ela_suspicious_ratio:.1%}  {ela_level}\n"
                f"  Mean error             : {self.ela_mean_error:.2f}\n"
                f"  Max error              : {self.ela_max_error:.2f}\n"
                f"  → High suspicious ratio suggests copy-paste or edited regions. "
                f"CPI (Copy-Paste Inside) is the dominant forgery type in this dataset."
            )
        else:
            lines.append("[ELA] Not available.")

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
                f"  → A high match count with ≥2 clusters is strong evidence of CPI forgery "
                f"(a region was copied and pasted within the same document)."
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
                if len(self.ocr_amounts) > 3:
                    total_sum = sum(self.ocr_amounts)
                    lines.append(f"  Amounts sum   : {total_sum:.2f}  (use to cross-check stated total)")
            if self.ocr_items:
                lines.append(f"  Item lines ({len(self.ocr_items)}): {' | '.join(self.ocr_items[:4])}")
            lines.append(
                "  → Use the extracted amounts to verify arithmetic BEFORE reading values "
                "from the image. Discrepancies between OCR amounts and visual totals are "
                "strong forgery signals."
            )
        else:
            lines.append("[OCR Transcription] Not available (no paired .txt file found).")

        lines.append("")

        # ── Errors ──
        if self.errors:
            lines.append(f"[Analysis Errors] {self.errors}")
            lines.append("")

        lines.append("=== END FORENSIC PRE-ANALYSIS ===")
        return "\n".join(lines)

    # ── Interpretation helpers ──

    def _interpret_ela(self) -> str:
        r = self.ela_suspicious_ratio or 0.0
        if r > 0.15:
            return "⚠ HIGH — significant editing likely"
        elif r > 0.05:
            return "⚡ MODERATE — some suspicious areas"
        else:
            return "✓ LOW — consistent compression"

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
            return "⚠ HIGH — possible copy-move detected"
        elif c > 0.2:
            return "⚡ MODERATE"
        else:
            return "✓ LOW"


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

    The output_dir is used to save intermediate forensic images (ELA maps, noise maps, etc.)
    which can be inspected or passed as additional images to multimodal models.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs/forensic",
        ela_quality: int = 95,
        ela_threshold: float = 25.0,
        ela_scale: float = 10.0,
        noise_block_size: int = 32,
        freq_block_size: int = 32,
        orb_features: int = 3000,
        match_threshold: float = 0.70,
        min_match_distance: float = 50.0,
        cluster_eps: float = 40.0,
        min_cluster_size: int = 3,
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
                ela_quality=ela_quality,
                ela_threshold=ela_threshold,
                ela_scale=ela_scale,
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
                          When provided, structured text extraction is included.

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

            # ── Populate ELA ──
            if report.ela is not None:
                ctx.ela_mean_error = report.ela.mean_error
                ctx.ela_suspicious_ratio = report.ela.suspicious_ratio
                ctx.ela_max_error = report.ela.max_error

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
                if ctx.ela_suspicious_ratio is not None:
                    signals.append(f"ELA={ctx.ela_suspicious_ratio:.0%}")
                if ctx.cm_confidence is not None:
                    signals.append(f"CM={ctx.cm_confidence:.2f}")
                if ctx.errors:
                    signals.append(f"errors={list(ctx.errors.keys())}")
                print(" | ".join(signals) or "ok")

        return results
