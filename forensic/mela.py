"""
Enhanced Multi-Quality Error Level Analysis (MELA) for receipt forgery detection.

This module replaces the simpler ``multi_ela_analysis`` in ``forensic_analysis.py``
with a production-grade pipeline that adds:

* **Multi-scale block analysis** across several block sizes (fused via
  pixel-wise max).
* **Edge-aware suppression** (low k=0.15) so only the strongest natural
  edges are mildly down-weighted; character interiors are fully preserved.
* **Adaptive thresholding over positive values only** (percentile or
  MAD-based) to prevent collapse when a large fraction of the variance map
  is zero (flat/background regions).
* **Dual suspicious-ratio reporting** — total (fraction of all pixels) and
  nonwhite (fraction of ink/content pixels only). Receipt white space
  dilutes the total ratio severely; the nonwhite ratio is the primary
  forensic signal.
* **Connected-component ROI extraction** with composite scoring (mean +
  peak variance) to surface concentrated anomalies over diffuse noise.
* **Morphological consolidation** of the binary mask before CC analysis
  to merge nearby suspicious pixels from the same pasted region.
* **Robust percentile statistics** (p50 .. p99, tail mass).
* **Early-exit optimisation** that skips remaining JPEG quality levels once
  a high suspicious ratio is already detected.

All heavy computation uses NumPy vectorised operations where possible.
Only ``cv2``, ``numpy``, and ``PIL`` are required.

Typical usage
-------------
>>> from forensic.mela import mela_analyze
>>> result = mela_analyze(image_rgb, output_dir=Path("out"))
>>> print(result.suspicious_ratio_nonwhite, result.top_rois)
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from forensic.utils import (
    ROI,
    apply_colormap,
    compute_edge_weight_map,
    compute_nonwhite_mask,
    normalize_to_uint8,
    save_image,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class MELAResult:
    """Enhanced Multi-ELA result with ROIs, percentiles, and multi-scale support."""

    # Core maps
    variance_map: np.ndarray          # Per-pixel cross-quality variance (float64)
    variance_display: np.ndarray      # Amplified grayscale for display (uint8)
    variance_color: np.ndarray        # Color-mapped heatmap (BGR)

    # Global statistics
    mean_variance: float
    max_variance: float
    suspicious_ratio: float           # flagged / total_pixels (diluted by white space)
    suspicious_ratio_nonwhite: float  # flagged / nonwhite_pixels (primary metric)

    # Block-level
    divergent_blocks: int
    total_blocks: int
    blocks_divergent_ratio: float

    # Config
    qualities_used: Tuple[int, ...]
    scales_used: List[int]

    # Enhanced statistics
    percentiles: Dict[str, float]     # {"p50", "p75", "p90", "p95", "p99"}
    tail_mass_pct: float              # % of nonwhite pixels above p95
    threshold_used: float             # adaptive threshold that was applied
    threshold_method: str             # "percentile" or "mad"

    # Connected components / ROIs
    components_count: int
    largest_component_area_pct: float
    top_rois: List[ROI]

    # Saved image paths
    saved_images: Dict[str, str]      # name -> path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compress_and_diff(
    pil_img: Image.Image,
    original_rgb: np.ndarray,
    quality: int,
) -> np.ndarray:
    """Compress *pil_img* as JPEG at *quality*, decompress, return per-pixel
    max-channel absolute difference with *original_rgb* (float32, H x W)."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
    diff = np.abs(original_rgb.astype(np.float32) - recompressed)
    return np.max(diff, axis=2)  # (H, W)


def _block_divergent_counts(
    variance_map: np.ndarray,
    block_size: int,
) -> Tuple[int, int, np.ndarray]:
    """Return (divergent_blocks, total_blocks, block_mean_map) for one scale.

    A block is *divergent* if its mean variance exceeds the global mean
    by more than 2 standard deviations.

    The returned ``block_mean_map`` has shape ``(rows, cols)`` where each
    element is the mean variance within that block.
    """
    h, w = variance_map.shape[:2]
    rows = max(1, h // block_size)
    cols = max(1, w // block_size)

    block_means = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        r0 = r * block_size
        r1 = min(r0 + block_size, h)
        for c in range(cols):
            c0 = c * block_size
            c1 = min(c0 + block_size, w)
            blk = variance_map[r0:r1, c0:c1]
            if blk.size == 0:
                continue
            block_means[r, c] = float(blk.mean())

    bm_mean = float(block_means.mean())
    bm_std = float(block_means.std())
    threshold = bm_mean + 2.0 * bm_std
    divergent = int((block_means > threshold).sum())
    total = rows * cols
    return divergent, total, block_means


def _upsample_block_map(
    block_map: np.ndarray,
    block_size: int,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Up-sample a (rows, cols) block-level map to (target_h, target_w) via
    nearest-neighbour interpolation."""
    return cv2.resize(
        block_map.astype(np.float64),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )


def _extract_rois(
    binary_mask: np.ndarray,
    variance_map: np.ndarray,
    max_rois: int,
    min_component_area_px: int,
    global_max_var: float = 1.0,
) -> Tuple[int, float, List[ROI]]:
    """Run connected-component analysis on *binary_mask* and return
    ``(components_count, largest_component_area_pct, top_rois)``.

    ROI scoring uses a composite of mean and peak variance so that
    concentrated high-variance spots (typical of pasted foreign content)
    rank above diffuse low-variance regions.

    Each ROI carries ``source="mela"``."""
    h, w = binary_mask.shape[:2]
    total_pixels = max(h * w, 1)

    # Ensure mask is uint8
    mask_u8 = binary_mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8,
    )

    global_max_var = max(global_max_var, 1e-6)

    # Label 0 is background
    components: List[dict] = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_component_area_px:
            continue
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        bw = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

        component_mask = (labels == label_idx)
        comp_values = variance_map[component_mask]
        mean_score = float(comp_values.mean())
        peak_score = float(comp_values.max())

        # Composite score: rewards concentrated high-variance spots.
        # A small region with very high peak ranks above a large region
        # with moderate mean. Normalise peak by global max so it's [0, 1].
        composite_score = mean_score * (1.0 + peak_score / global_max_var)

        area_pct = area / total_pixels * 100.0

        components.append({
            "label": label_idx,
            "area": area,
            "area_pct": area_pct,
            "bbox": (x, y, bw, bh),
            "mean_score": mean_score,
            "peak_score": peak_score,
            "composite_score": composite_score,
            "centroid": (float(centroids[label_idx][0]), float(centroids[label_idx][1])),
        })

    components_count = len(components)

    if components_count == 0:
        return 0, 0.0, []

    largest_area_pct = max(c["area_pct"] for c in components)

    # Sort by composite score descending, take top-N
    components.sort(key=lambda c: c["composite_score"], reverse=True)
    top = components[:max_rois]

    rois: List[ROI] = []
    for c in top:
        x, y, bw, bh = c["bbox"]
        cx, cy = c["centroid"]
        h_pos = "left" if cx < w * 0.33 else ("right" if cx > w * 0.67 else "center")
        v_pos = "top" if cy < h * 0.33 else ("bottom" if cy > h * 0.67 else "middle")
        notes = (
            f"{v_pos}-{h_pos} | area={c['area']}px ({c['area_pct']:.1f}%)"
            f" | peak_var={c['peak_score']:.1f}"
        )
        rois.append(
            ROI(
                bbox=(x, y, bw, bh),
                score=c["mean_score"],
                area_pct=c["area_pct"],
                source="mela",
                notes=notes,
            )
        )

    return components_count, largest_area_pct, rois


def _draw_rois(
    image_rgb: np.ndarray,
    rois: List[ROI],
) -> np.ndarray:
    """Draw bounding boxes for each ROI on a BGR copy of *image_rgb*."""
    vis = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for i, roi in enumerate(rois):
        color = (0, 0, 255)  # red in BGR
        thickness = 2
        x, y, bw, bh = roi.bbox
        pt1 = (x, y)
        pt2 = (x + bw, y + bh)
        cv2.rectangle(vis, pt1, pt2, color, thickness)
        label_text = f"ROI-{i+1}  s={roi.score:.2f}"
        label_y = y - 6 if y > 16 else y + 16
        cv2.putText(
            vis, label_text, (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
    return vis


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def mela_analyze(
    image_rgb: np.ndarray,
    *,
    output_dir: Optional[Path] = None,
    prefix: str = "",
    # JPEG qualities
    qualities: Tuple[int, ...] = (60, 70, 80, 85, 90, 95),
    # Multi-scale block sizes
    block_sizes: Tuple[int, ...] = (8, 16, 32),
    # Adaptive threshold
    threshold_method: str = "mad",       # default changed from "percentile" to "mad"
    threshold_percentile: float = 98.0,
    mad_z_threshold: float = 3.5,
    # ROI extraction
    max_rois: int = 5,
    min_component_area_px: int = 50,     # lowered from 100 to catch small forged regions
    # Edge suppression — low k preserves signal in character interiors
    edge_suppression_k: float = 0.15,    # lowered from 0.5; CPI forgeries appear at char boundaries
    # Early exit (skip remaining qualities if HIGH already detected)
    early_exit: bool = True,
    early_exit_ratio: float = 0.10,      # fraction of nonwhite pixels
) -> MELAResult:
    """Run Enhanced Multi-Quality Error Level Analysis on a receipt image.

    Parameters
    ----------
    image_rgb : np.ndarray
        Original receipt image in RGB uint8 (H, W, 3).  Grayscale is
        computed internally — no pre-cropped version is required.
    output_dir : Path, optional
        Directory to save output images.  Nothing is saved when *None*.
    prefix : str
        Filename prefix for saved images.
    qualities : tuple of int
        JPEG quality levels to compress at.
    block_sizes : tuple of int
        Block sizes for multi-scale block analysis.
    threshold_method : str
        ``"mad"`` (default) or ``"percentile"`` for adaptive thresholding.
        Both operate on **positive values only** to prevent collapse when
        many background/flat pixels carry zero variance.
    threshold_percentile : float
        Percentile used when *threshold_method* is ``"percentile"``.
    mad_z_threshold : float
        Number of MADs above median for the ``"mad"`` method.
    max_rois : int
        Maximum number of ROIs to return.
    min_component_area_px : int
        Minimum connected-component area (pixels) to keep as ROI.
    edge_suppression_k : float
        Strength of edge-aware suppression.  Lower k = milder suppression,
        preserving signal near character edges where CPI forgeries appear.
    early_exit : bool
        If *True*, stop adding JPEG quality levels once the running
        suspicious ratio (relative to nonwhite pixels) exceeds *early_exit_ratio*.
    early_exit_ratio : float
        Threshold for the early-exit check (fraction of nonwhite pixels).

    Returns
    -------
    MELAResult
        Populated result dataclass with maps, statistics, ROIs, and
        (optionally) saved image paths.
    """

    # ------------------------------------------------------------------
    # 0. Validate / prepare
    # ------------------------------------------------------------------
    h, w = image_rgb.shape[:2]
    total_pixels = max(h * w, 1)

    if h < 4 or w < 4:
        return _empty_result(qualities, list(block_sizes))

    # Compute grayscale internally — no cropped version needed
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Compute non-white mask early so we can use it for the quick ratio in early-exit
    nonwhite_mask = compute_nonwhite_mask(gray, threshold=240)
    nonwhite_bool = nonwhite_mask.astype(bool)
    nonwhite_count = max(int(nonwhite_bool.sum()), 1)

    pil_img = Image.fromarray(image_rgb)

    # ------------------------------------------------------------------
    # 1. Multi-quality ELA
    # ------------------------------------------------------------------
    ela_maps: List[np.ndarray] = []
    used_qualities: List[int] = []

    for q in qualities:
        emap = _compress_and_diff(pil_img, image_rgb, q)
        ela_maps.append(emap)
        used_qualities.append(q)

        # Early-exit check: compute quick suspicious ratio relative to nonwhite pixels
        if early_exit and len(ela_maps) >= 2:
            stack_tmp = np.stack(ela_maps, axis=2)
            var_tmp = np.var(stack_tmp, axis=2)
            pos_vals_tmp = var_tmp[nonwhite_bool & (var_tmp > 0)]
            if pos_vals_tmp.size >= 10:
                quick_thresh = float(np.percentile(pos_vals_tmp, 95))
                quick_flagged = int(((var_tmp > quick_thresh) & nonwhite_bool).sum())
                running_susp_ratio = quick_flagged / nonwhite_count
                if running_susp_ratio > early_exit_ratio:
                    logger.debug(
                        "MELA early-exit at quality %d (nw_susp_ratio=%.4f > %.4f)",
                        q, running_susp_ratio, early_exit_ratio,
                    )
                    break

    qualities_tuple: Tuple[int, ...] = tuple(used_qualities)

    # ------------------------------------------------------------------
    # 2. Cross-quality variance
    # ------------------------------------------------------------------
    ela_stack = np.stack(ela_maps, axis=2).astype(np.float64)  # (H, W, Q)
    variance_map = np.var(ela_stack, axis=2)                   # (H, W) float64

    # ------------------------------------------------------------------
    # 3. Edge-aware suppression (mild — preserves signal near char edges)
    # ------------------------------------------------------------------
    edge_weight = compute_edge_weight_map(gray, k=edge_suppression_k)
    variance_map = variance_map * edge_weight

    # ------------------------------------------------------------------
    # 4. Non-white mask — zero out white background pixels
    # ------------------------------------------------------------------
    variance_map[~nonwhite_bool] = 0.0

    # Pixels used for statistics (non-white only)
    nw_values = variance_map[nonwhite_bool]
    if nw_values.size == 0:
        return _empty_result(qualities_tuple, list(block_sizes))

    # **Positive values only** for threshold computation.
    # Many background-adjacent or flat nonwhite pixels carry zero variance;
    # including them would drag the MAD/percentile threshold toward zero,
    # causing threshold collapse or near-100% flagging.
    pos_nw_values = nw_values[nw_values > 1e-6]
    if pos_nw_values.size < max(10, nw_values.size // 100):
        # Very few positive values — image is nearly flat; fall back to all nw
        pos_nw_values = nw_values

    # ------------------------------------------------------------------
    # 5. Multi-scale block analysis
    # ------------------------------------------------------------------
    total_divergent = 0
    total_block_count = 0
    fused_score_map = np.zeros((h, w), dtype=np.float64)

    for bs in block_sizes:
        if bs > min(h, w):
            continue
        div, tot, block_mean_map = _block_divergent_counts(variance_map, bs)
        total_divergent += div
        total_block_count += tot

        upsampled = _upsample_block_map(block_mean_map, bs, h, w)
        fused_score_map = np.maximum(fused_score_map, upsampled)

    if total_block_count == 0:
        total_block_count = 1

    blocks_divergent_ratio = total_divergent / total_block_count

    # ------------------------------------------------------------------
    # 6. Adaptive threshold — computed over POSITIVE values only
    # ------------------------------------------------------------------
    if threshold_method == "mad":
        median_val = float(np.median(pos_nw_values))
        mad_val = float(np.median(np.abs(pos_nw_values - median_val)))
        # Scale MAD to σ-equivalent for normal distributions
        threshold_val = median_val + mad_z_threshold * mad_val * 1.4826
        actual_method = "mad"
    else:
        # "percentile" — applied to positive values only
        threshold_val = float(np.percentile(pos_nw_values, threshold_percentile))
        actual_method = "percentile"

    # Guard against degenerate threshold
    threshold_val = max(threshold_val, 1e-12)

    # Binary mask from adaptive threshold (restricted to nonwhite)
    binary_mask = (variance_map > threshold_val).astype(np.uint8) * 255
    binary_mask[~nonwhite_bool] = 0

    # ------------------------------------------------------------------
    # 6b. Morphological consolidation
    # Dilate then erode (closing) to merge nearby suspicious pixels from
    # the same pasted region before connected-component analysis.
    # Kernel size 3×3, 1 iteration — conservative to avoid inflating ROIs.
    # ------------------------------------------------------------------
    if np.any(binary_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Re-apply nonwhite restriction (morphology may expand into white area)
        binary_mask[~nonwhite_bool] = 0

    # ------------------------------------------------------------------
    # 6c. Dual suspicious-ratio reporting
    # ------------------------------------------------------------------
    n_flagged = int(np.count_nonzero(binary_mask))
    suspicious_ratio = n_flagged / total_pixels          # diluted by white space
    suspicious_ratio_nonwhite = n_flagged / nonwhite_count  # primary forensic signal

    # ------------------------------------------------------------------
    # 7. ROI extraction
    # ------------------------------------------------------------------
    global_max_var = float(variance_map.max()) if variance_map.max() > 0 else 1.0
    components_count, largest_component_area_pct, top_rois = _extract_rois(
        binary_mask, variance_map, max_rois, min_component_area_px, global_max_var,
    )

    # ------------------------------------------------------------------
    # 8. Robust statistics (over positive nonwhite values for meaningful percentiles)
    # ------------------------------------------------------------------
    p50 = float(np.percentile(pos_nw_values, 50))
    p75 = float(np.percentile(pos_nw_values, 75))
    p90 = float(np.percentile(pos_nw_values, 90))
    p95 = float(np.percentile(pos_nw_values, 95))
    p99 = float(np.percentile(pos_nw_values, 99))

    percentiles: Dict[str, float] = {
        "p50": p50, "p75": p75, "p90": p90, "p95": p95, "p99": p99,
    }

    # Tail mass: % of nonwhite pixels above p95
    tail_mass_pct = float(np.sum(nw_values > p95) / max(nw_values.size, 1) * 100.0)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    # Amplified grayscale: 16x amplification (up from 8x) for better visibility
    variance_display = np.clip(variance_map * 16.0, 0, 255).astype(np.uint8)
    variance_color = apply_colormap(variance_display)

    # ------------------------------------------------------------------
    # 9. Save images
    # ------------------------------------------------------------------
    saved_images: Dict[str, str] = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        name_variance = f"{prefix}_mela_variance.png" if prefix else "mela_variance.png"
        name_color = f"{prefix}_mela_color.png" if prefix else "mela_color.png"
        name_rois = f"{prefix}_mela_rois.png" if prefix else "mela_rois.png"

        p_var = save_image(variance_display, output_dir, name_variance)
        saved_images["variance"] = str(p_var)

        p_col = save_image(variance_color, output_dir, name_color)
        saved_images["color"] = str(p_col)

        if top_rois:
            roi_vis = _draw_rois(image_rgb, top_rois)
        else:
            roi_vis = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        p_rois = save_image(roi_vis, output_dir, name_rois)
        saved_images["rois"] = str(p_rois)

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    mean_variance = float(variance_map.mean())
    max_variance = float(variance_map.max())

    return MELAResult(
        variance_map=variance_map,
        variance_display=variance_display,
        variance_color=variance_color,
        mean_variance=mean_variance,
        max_variance=max_variance,
        suspicious_ratio=suspicious_ratio,
        suspicious_ratio_nonwhite=suspicious_ratio_nonwhite,
        divergent_blocks=total_divergent,
        total_blocks=total_block_count,
        blocks_divergent_ratio=blocks_divergent_ratio,
        qualities_used=qualities_tuple,
        scales_used=list(block_sizes),
        percentiles=percentiles,
        tail_mass_pct=tail_mass_pct,
        threshold_used=threshold_val,
        threshold_method=actual_method,
        components_count=components_count,
        largest_component_area_pct=largest_component_area_pct,
        top_rois=top_rois,
        saved_images=saved_images,
    )


# ---------------------------------------------------------------------------
# Degenerate / edge-case helper
# ---------------------------------------------------------------------------

def _empty_result(
    qualities_used: Tuple[int, ...],
    scales_used: List[int],
) -> MELAResult:
    """Return a zeroed-out ``MELAResult`` for degenerate inputs."""
    empty_map = np.zeros((1, 1), dtype=np.float64)
    empty_u8 = np.zeros((1, 1), dtype=np.uint8)
    empty_bgr = np.zeros((1, 1, 3), dtype=np.uint8)

    return MELAResult(
        variance_map=empty_map,
        variance_display=empty_u8,
        variance_color=empty_bgr,
        mean_variance=0.0,
        max_variance=0.0,
        suspicious_ratio=0.0,
        suspicious_ratio_nonwhite=0.0,
        divergent_blocks=0,
        total_blocks=0,
        blocks_divergent_ratio=0.0,
        qualities_used=qualities_used,
        scales_used=scales_used,
        percentiles={"p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0},
        tail_mass_pct=0.0,
        threshold_used=0.0,
        threshold_method="n/a",
        components_count=0,
        largest_component_area_pct=0.0,
        top_rois=[],
        saved_images={},
    )
