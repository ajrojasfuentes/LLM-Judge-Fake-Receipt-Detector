"""
Enhanced Multi-Quality Error Level Analysis (MELA) for receipt forgery detection.

This module replaces the simpler ``multi_ela_analysis`` in ``forensic_analysis.py``
with a production-grade pipeline that adds:

* **Multi-scale block analysis** across several block sizes (fused via
  pixel-wise max).
* **Edge-aware suppression** so that natural high-contrast edges (text
  boundaries, table lines) do not dominate the variance map.
* **Adaptive thresholding** (percentile or MAD-based) instead of a fixed
  variance cutoff.
* **Connected-component ROI extraction** with area filtering and scoring.
* **Robust percentile statistics** (p50 .. p99, tail mass).
* **Early-exit optimisation** that skips remaining JPEG quality levels once
  a high suspicious ratio is already detected.

All heavy computation uses NumPy vectorised operations where possible.
Only ``cv2``, ``numpy``, and ``PIL`` are required.

Typical usage
-------------
>>> from forensic.mela import mela_analyze
>>> result = mela_analyze(image_rgb, cropped_gray, output_dir=Path("out"))
>>> print(result.suspicious_ratio, result.top_rois)
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
    suspicious_ratio: float           # fraction above adaptive threshold

    # Block-level
    divergent_blocks: int
    total_blocks: int
    blocks_divergent_ratio: float

    # Config
    qualities_used: Tuple[int, ...]
    scales_used: List[int]

    # Enhanced statistics
    percentiles: Dict[str, float]     # {"p50", "p75", "p90", "p95", "p99"}
    tail_mass_pct: float              # % of pixels above p95
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
) -> Tuple[int, float, List[ROI]]:
    """Run connected-component analysis on *binary_mask* and return
    ``(components_count, largest_component_area_pct, top_rois)``.

    Each ROI carries ``source="mela"``."""
    h, w = binary_mask.shape[:2]
    total_pixels = max(h * w, 1)

    # Ensure mask is uint8
    mask_u8 = binary_mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8,
    )

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

        # Mean score inside the component
        component_mask = (labels == label_idx)
        mean_score = float(variance_map[component_mask].mean())
        area_pct = area / total_pixels * 100.0

        components.append({
            "label": label_idx,
            "area": area,
            "area_pct": area_pct,
            "bbox": (x, y, bw, bh),
            "mean_score": mean_score,
            "centroid": (float(centroids[label_idx][0]), float(centroids[label_idx][1])),
        })

    components_count = len(components)

    if components_count == 0:
        return 0, 0.0, []

    largest_area_pct = max(c["area_pct"] for c in components)

    # Sort by mean_score descending, take top-N
    components.sort(key=lambda c: c["mean_score"], reverse=True)
    top = components[:max_rois]

    rois: List[ROI] = []
    for c in top:
        x, y, bw, bh = c["bbox"]
        # Build position note for the judge
        cx, cy = c["centroid"]
        h_pos = "left" if cx < w * 0.33 else ("right" if cx > w * 0.67 else "center")
        v_pos = "top" if cy < h * 0.33 else ("bottom" if cy > h * 0.67 else "middle")
        notes = f"{v_pos}-{h_pos} | area={c['area']}px ({c['area_pct']:.1f}%)"
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
        # Place label above the box; fall back to inside if at top edge
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
    cropped_gray: np.ndarray,
    *,
    output_dir: Optional[Path] = None,
    prefix: str = "",
    # JPEG qualities
    qualities: Tuple[int, ...] = (60, 70, 80, 85, 90, 95),
    # Multi-scale block sizes
    block_sizes: Tuple[int, ...] = (8, 16, 32),
    # Adaptive threshold
    threshold_method: str = "percentile",  # "percentile" or "mad"
    threshold_percentile: float = 98.0,
    mad_z_threshold: float = 3.5,
    # ROI extraction
    max_rois: int = 5,
    min_component_area_px: int = 100,
    # Edge suppression
    edge_suppression_k: float = 0.5,
    # Early exit (skip remaining qualities if HIGH already detected)
    early_exit: bool = True,
    early_exit_ratio: float = 0.10,
) -> MELAResult:
    """Run Enhanced Multi-Quality Error Level Analysis on a receipt image.

    Parameters
    ----------
    image_rgb : np.ndarray
        Original receipt image in RGB uint8 (H, W, 3).
    cropped_gray : np.ndarray
        Grayscale version of the (possibly cropped) receipt, uint8 (H, W).
        Used for edge-weight computation.
    output_dir : Path, optional
        Directory to save output images.  Nothing is saved when *None*.
    prefix : str
        Filename prefix for saved images.
    qualities : tuple of int
        JPEG quality levels to compress at.
    block_sizes : tuple of int
        Block sizes for multi-scale block analysis.
    threshold_method : str
        ``"percentile"`` or ``"mad"`` for adaptive thresholding.
    threshold_percentile : float
        Percentile used when *threshold_method* is ``"percentile"``
        (applied to non-white pixels only).
    mad_z_threshold : float
        Number of MADs above median for the ``"mad"`` method.
    max_rois : int
        Maximum number of ROIs to return.
    min_component_area_px : int
        Minimum connected-component area (pixels) to keep as ROI.
    edge_suppression_k : float
        Strength of edge-aware suppression.  Higher values suppress more
        aggressively near edges.
    early_exit : bool
        If *True*, stop adding JPEG quality levels once the running
        suspicious ratio exceeds *early_exit_ratio*.
    early_exit_ratio : float
        Threshold for the early-exit check.

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

    # Handle degenerate images (all-white, tiny, etc.)
    if h < 4 or w < 4:
        return _empty_result(qualities, list(block_sizes))

    pil_img = Image.fromarray(image_rgb)

    # ------------------------------------------------------------------
    # 1. Multi-quality ELA
    # ------------------------------------------------------------------
    ela_maps: List[np.ndarray] = []
    used_qualities: List[int] = []
    running_susp_ratio = 0.0

    for q in qualities:
        emap = _compress_and_diff(pil_img, image_rgb, q)
        ela_maps.append(emap)
        used_qualities.append(q)

        # Early-exit check: compute quick suspicious ratio on running stack
        if early_exit and len(ela_maps) >= 2:
            stack_tmp = np.stack(ela_maps, axis=2)
            var_tmp = np.var(stack_tmp, axis=2)
            quick_thresh = np.percentile(var_tmp[var_tmp > 0], 95) if np.any(var_tmp > 0) else 1.0
            running_susp_ratio = float((var_tmp > quick_thresh).sum()) / total_pixels
            if running_susp_ratio > early_exit_ratio:
                logger.debug(
                    "MELA early-exit at quality %d (suspicious_ratio=%.4f > %.4f)",
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
    # 3. Edge-aware suppression
    # ------------------------------------------------------------------
    # Ensure cropped_gray matches spatial dimensions of image_rgb
    if cropped_gray.shape[:2] != (h, w):
        cropped_gray = cv2.resize(cropped_gray, (w, h), interpolation=cv2.INTER_AREA)

    edge_weight = compute_edge_weight_map(cropped_gray, k=edge_suppression_k)
    variance_map = variance_map * edge_weight

    # ------------------------------------------------------------------
    # 4. Non-white mask
    # ------------------------------------------------------------------
    nonwhite_mask = compute_nonwhite_mask(cropped_gray, threshold=240)
    nonwhite_bool = nonwhite_mask.astype(bool)   # uint8 (0/255) → bool for indexing
    variance_map[~nonwhite_bool] = 0.0

    # Pixels used for statistics (non-white only)
    nw_values = variance_map[nonwhite_bool]
    if nw_values.size == 0:
        # Entire image is white / background
        return _empty_result(qualities_tuple, list(block_sizes))

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

    # Ensure we have at least some blocks counted
    if total_block_count == 0:
        total_block_count = 1

    blocks_divergent_ratio = total_divergent / total_block_count

    # ------------------------------------------------------------------
    # 6. Adaptive threshold
    # ------------------------------------------------------------------
    if threshold_method == "mad":
        median_val = float(np.median(nw_values))
        mad_val = float(np.median(np.abs(nw_values - median_val)))
        # Scale MAD to approximate standard deviation for normal distributions
        threshold_val = median_val + mad_z_threshold * mad_val * 1.4826
        actual_method = "mad"
    else:
        # Default: percentile
        threshold_val = float(np.percentile(nw_values, threshold_percentile))
        actual_method = "percentile"

    # Ensure threshold is not degenerate
    threshold_val = max(threshold_val, 1e-12)

    # Binary mask from adaptive threshold
    binary_mask = (variance_map > threshold_val).astype(np.uint8) * 255
    # Restrict to non-white
    binary_mask[~nonwhite_bool] = 0

    suspicious_ratio = float(np.count_nonzero(binary_mask) / total_pixels)

    # ------------------------------------------------------------------
    # 7. ROI extraction
    # ------------------------------------------------------------------
    components_count, largest_component_area_pct, top_rois = _extract_rois(
        binary_mask, variance_map, max_rois, min_component_area_px,
    )

    # ------------------------------------------------------------------
    # 8. Robust statistics
    # ------------------------------------------------------------------
    p50 = float(np.percentile(nw_values, 50))
    p75 = float(np.percentile(nw_values, 75))
    p90 = float(np.percentile(nw_values, 90))
    p95 = float(np.percentile(nw_values, 95))
    p99 = float(np.percentile(nw_values, 99))

    percentiles: Dict[str, float] = {
        "p50": p50,
        "p75": p75,
        "p90": p90,
        "p95": p95,
        "p99": p99,
    }

    # Tail mass: % of non-white pixels above p95
    if nw_values.size > 0:
        tail_mass_pct = float(np.sum(nw_values > p95) / nw_values.size * 100.0)
    else:
        tail_mass_pct = 0.0

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    # Amplified grayscale: scale variance for visibility (8x amplification)
    variance_display = np.clip(variance_map * 8.0, 0, 255).astype(np.uint8)
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

        # ROI visualisation
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
        # Core maps
        variance_map=variance_map,
        variance_display=variance_display,
        variance_color=variance_color,
        # Global statistics
        mean_variance=mean_variance,
        max_variance=max_variance,
        suspicious_ratio=suspicious_ratio,
        # Block-level
        divergent_blocks=total_divergent,
        total_blocks=total_block_count,
        blocks_divergent_ratio=blocks_divergent_ratio,
        # Config
        qualities_used=qualities_tuple,
        scales_used=list(block_sizes),
        # Enhanced statistics
        percentiles=percentiles,
        tail_mass_pct=tail_mass_pct,
        threshold_used=threshold_val,
        threshold_method=actual_method,
        # Connected components / ROIs
        components_count=components_count,
        largest_component_area_pct=largest_component_area_pct,
        top_rois=top_rois,
        # Saved image paths
        saved_images=saved_images,
    )


# ---------------------------------------------------------------------------
# Degenerate / edge-case helper
# ---------------------------------------------------------------------------

def _empty_result(
    qualities_used: Tuple[int, ...],
    scales_used: List[int],
) -> MELAResult:
    """Return a zeroed-out ``MELAResult`` for degenerate inputs (all-white
    images, images too small to analyse, etc.)."""
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
