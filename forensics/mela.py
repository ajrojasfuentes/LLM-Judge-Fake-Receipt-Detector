"""
forensics.mela — Multi-Quality Error Level Analysis (MELA).

MELA extends classical ELA by compressing the image at multiple JPEG
quality levels, computing the pixel-wise reconstruction error at each
level, and fusing the results with multi-scale local variance analysis.

Algorithm overview
------------------
1. For each quality level ``q`` in *qualities* (default 95, 90, …, 70):
   a. Compress the image to JPEG at quality ``q`` and decode it back.
   b. Compute the per-pixel mean absolute difference across RGB channels.
   c. Accumulate the element-wise maximum across all quality levels.
   d. Apply an early-exit heuristic when the hot-pixel ratio exceeds a
      threshold (``early_exit_ratio``).

2. Normalise the accumulated ELA map to ``[0, 1]``.

3. Compute multi-scale local variance over the normalised ELA map using
   box filters at each block size (default 8, 16, 32) and fuse by
   element-wise maximum.

4. Optionally suppress regions near strong natural edges (Sobel
   magnitude) to reduce false positives from legitimate high-contrast
   boundaries.

5. Threshold the fused variance map at the ``thr_percentile`` percentile
   and extract connected components with ``cv2.connectedComponentsWithStats``.

6. Filter small components (``< min_area_px``) and keep the top-K ROIs
   ranked by mean anomaly score.

Outputs saved to *out_dir*:
    ``mela_heat.png``        Grayscale heatmap of the fused variance.
    ``mela_overlay.png``     Red-channel overlay of the heatmap on the
                             original image.
    ``mela_rois.png``        Original image with green bounding boxes.
    ``mela_roi_<id>.png``    Cropped image for each returned ROI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .utils import (
    ROI,
    crop_rgb,
    draw_bboxes,
    ensure_dir,
    normalize_01,
    overlay_heatmap,
    pil_jpeg_bytes,
    pil_load_jpeg_bytes,
    robust_percentiles,
    save_gray_png,
    save_rgb_png,
    to_gray_u8,
)

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _local_var_map(gray_f: np.ndarray, k: int) -> np.ndarray:
    """Compute a fast local-variance map using box filters.

    Variance is estimated as ``E[X^2] - E[X]^2`` with square box kernels
    of side length *k*.

    Parameters
    ----------
    gray_f : np.ndarray
        2-D float array (single-channel image or map).
    k : int
        Box-filter kernel side length in pixels.

    Returns
    -------
    np.ndarray
        Local-variance map with the same shape as *gray_f* (float32).
        Negative variance artefacts are clamped to zero.
    """
    if cv2 is None:
        from scipy.ndimage import uniform_filter  # type: ignore

        m = uniform_filter(gray_f, size=k)
        m2 = uniform_filter(gray_f * gray_f, size=k)
        return np.maximum(0.0, m2 - m * m)

    m = cv2.blur(gray_f, (k, k))
    m2 = cv2.blur(gray_f * gray_f, (k, k))
    return np.maximum(0.0, m2 - m * m)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mela_analyze(
    rgb: np.ndarray,
    out_dir: str,
    qualities: Sequence[int] = (95, 90, 85, 80, 75, 70),
    block_sizes: Sequence[int] = (8, 16, 32),
    edge_suppress: bool = True,
    thr_percentile: float = 99.3,
    min_area_px: int = 120,
    topk_rois: int = 6,
    early_exit_ratio: float = 0.12,
) -> Dict[str, Any]:
    """Run Multi-Quality Error Level Analysis on an RGB image.

    Parameters
    ----------
    rgb : np.ndarray
        Input image of shape ``(H, W, 3)``, dtype ``uint8``.
    out_dir : str
        Directory where output artifact images will be saved.
    qualities : sequence of int
        JPEG quality levels to test (descending order recommended).
    block_sizes : sequence of int
        Box-filter kernel sizes for multi-scale variance fusion.
    edge_suppress : bool
        If ``True``, suppress variance near strong natural edges to
        reduce false positives.
    thr_percentile : float
        Percentile used to threshold the fused variance map (0-100).
    min_area_px : int
        Minimum connected-component area (in pixels) to keep as an ROI.
    topk_rois : int
        Maximum number of ROIs to return, ranked by mean score.
    early_exit_ratio : float
        If the fraction of hot pixels exceeds this ratio at any quality
        level, skip remaining quality levels as a cost-saving heuristic.

    Returns
    -------
    dict
        Keys:

        * ``"summary"`` — Scalar metrics (qualities used, threshold,
          suspicious ratio, peak score, percentiles).
        * ``"rois"`` — List of :class:`ROI` dataclass instances.
        * ``"artifacts"`` — Paths to saved PNG images.
    """
    outp = ensure_dir(out_dir)
    H, W = rgb.shape[:2]

    # ------------------------------------------------------------------
    # Step 1: Multi-quality ELA fusion (pixel-wise max across qualities)
    # ------------------------------------------------------------------
    ela_max = np.zeros((H, W), dtype=np.float32)
    used_q: List[int] = []

    for q in qualities:
        jpeg_b = pil_jpeg_bytes(rgb, quality=int(q))
        rec = pil_load_jpeg_bytes(jpeg_b)
        diff = np.abs(rgb.astype(np.int16) - rec.astype(np.int16)).astype(np.float32)
        # Per-pixel mean across channels (grayscale magnitude)
        d = diff.mean(axis=2)
        ela_max = np.maximum(ela_max, d)
        used_q.append(int(q))

        # Early exit: if hot-pixel fraction is already high, additional
        # lower qualities are unlikely to add new information.
        tmp01 = normalize_01(ela_max)
        if float((tmp01 > 0.65).mean()) > early_exit_ratio:
            break

    ela01 = normalize_01(ela_max)

    # ------------------------------------------------------------------
    # Step 2: Multi-scale local variance fusion
    # ------------------------------------------------------------------
    v_fused = np.zeros_like(ela01, dtype=np.float32)
    for k in block_sizes:
        v = _local_var_map(ela01.astype(np.float32), int(k))
        v_fused = np.maximum(v_fused, v.astype(np.float32))
    v01 = normalize_01(v_fused)

    # ------------------------------------------------------------------
    # Step 3: Edge-aware suppression
    # ------------------------------------------------------------------
    if edge_suppress and cv2 is not None:
        gray = to_gray_u8(rgb)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gmag = np.sqrt(gx * gx + gy * gy)
        g01 = normalize_01(gmag)
        v01 = v01 * (1.0 - 0.55 * g01)

    # ------------------------------------------------------------------
    # Step 4: Adaptive threshold and connected-component ROIs
    # ------------------------------------------------------------------
    thr = float(np.percentile(v01.ravel(), thr_percentile))
    mask = (v01 >= thr).astype(np.uint8)

    rois: List[ROI] = []
    if cv2 is not None:
        num, labels, cc_stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8,
        )
        for i in range(1, num):
            x, y, w, h, area = cc_stats[i].tolist()
            if area < min_area_px:
                continue
            score = float(v01[y : y + h, x : x + w].mean())
            rois.append(ROI(
                roi_id=f"r{i}",
                bbox=(int(x), int(y), int(w), int(h)),
                score=score,
                signals=["mela_hotspot"],
            ))
        rois.sort(key=lambda r: r.score, reverse=True)
        rois = rois[:topk_rois]

    suspicious_ratio = float(mask.mean())
    stats_pct = robust_percentiles(v01, ps=(50, 75, 90, 95, 97, 99))
    peak_score = float(np.max(v01))

    # ------------------------------------------------------------------
    # Step 5: Save visualisation artifacts
    # ------------------------------------------------------------------
    heat_path = save_gray_png(v01, outp / "mela_heat.png")
    overlay = overlay_heatmap(rgb, v01, alpha=0.50)
    overlay_path = save_rgb_png(overlay, outp / "mela_overlay.png")

    bbox_img = draw_bboxes(rgb, [r.bbox for r in rois], thickness=2) if rois else rgb
    rois_path = save_rgb_png(bbox_img, outp / "mela_rois.png")

    # Save individual ROI crops
    roi_paths: List[str] = []
    for r in rois:
        crop = crop_rgb(rgb, r.bbox, pad=6)
        cp = outp / f"mela_roi_{r.roi_id}.png"
        r.crop_path = save_rgb_png(crop, cp)
        roi_paths.append(r.crop_path)

    return {
        "summary": {
            "used_qualities": used_q,
            "block_sizes": list(map(int, block_sizes)),
            "thr_percentile": float(thr_percentile),
            "threshold_value": float(thr),
            "suspicious_ratio": suspicious_ratio,
            "peak_score": peak_score,
            "percentiles": stats_pct,
        },
        "rois": rois,
        "artifacts": {
            "mela_heat": heat_path,
            "mela_overlay": overlay_path,
            "mela_rois": rois_path,
            "mela_roi_crops": roi_paths,
        },
    }
