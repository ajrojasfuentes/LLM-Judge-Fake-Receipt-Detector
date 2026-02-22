from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import ROI, crop_rgb, draw_bboxes, ensure_dir, normalize_01, overlay_heatmap, pil_jpeg_bytes, pil_load_jpeg_bytes, robust_percentiles, save_gray_png, save_rgb_png, to_gray_u8

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def _local_var_map(gray_f: np.ndarray, k: int) -> np.ndarray:
    """Fast local variance using box filters (requires cv2)."""
    if cv2 is None:
        # fallback: naive (slow) – but keep safe
        from scipy.ndimage import uniform_filter  # type: ignore
        m = uniform_filter(gray_f, size=k)
        m2 = uniform_filter(gray_f * gray_f, size=k)
        return np.maximum(0.0, m2 - m * m)

    m = cv2.blur(gray_f, (k, k))
    m2 = cv2.blur(gray_f * gray_f, (k, k))
    return np.maximum(0.0, m2 - m * m)


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
    """
    Multi-quality ELA + multi-scale variance fusion + connected-component ROIs.
    Saves:
      - mela_heat.png (0..1)
      - mela_overlay.png
      - mela_rois.png (bboxes drawn)
      - mela_roi_<id>.png crops
    """
    outp = ensure_dir(out_dir)
    H, W = rgb.shape[:2]

    # Multi-quality ELA fusion: pixel-wise max across qualities
    ela_max = np.zeros((H, W), dtype=np.float32)
    used_q = []

    for q in qualities:
        jpeg_b = pil_jpeg_bytes(rgb, quality=int(q))
        rec = pil_load_jpeg_bytes(jpeg_b)
        diff = np.abs(rgb.astype(np.int16) - rec.astype(np.int16)).astype(np.float32)
        # magnitude + grayscale
        d = diff.mean(axis=2)
        ela_max = np.maximum(ela_max, d)
        used_q.append(int(q))

        # early exit (cheap heuristic)
        tmp01 = normalize_01(ela_max)
        if float((tmp01 > 0.65).mean()) > early_exit_ratio:
            break

    ela01 = normalize_01(ela_max)

    # Multi-scale variance fusion (variance of ELA map)
    v_fused = np.zeros_like(ela01, dtype=np.float32)
    for k in block_sizes:
        v = _local_var_map(ela01.astype(np.float32), int(k))
        v_fused = np.maximum(v_fused, v.astype(np.float32))
    v01 = normalize_01(v_fused)

    # edge-aware suppression (reduce strong natural edges)
    if edge_suppress and cv2 is not None:
        gray = to_gray_u8(rgb)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gmag = np.sqrt(gx * gx + gy * gy)
        g01 = normalize_01(gmag)
        # suppress where edge is strong
        v01 = v01 * (1.0 - 0.55 * g01)

    # adaptive threshold
    thr = float(np.percentile(v01.ravel(), thr_percentile))
    mask = (v01 >= thr).astype(np.uint8)

    # connected components -> ROIs
    rois: List[ROI] = []
    if cv2 is not None:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num):
            x, y, w, h, area = stats[i].tolist()
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

    # Save artifacts
    heat_path = save_gray_png(v01, outp / "mela_heat.png")
    overlay = overlay_heatmap(rgb, v01, alpha=0.50)
    overlay_path = save_rgb_png(overlay, outp / "mela_overlay.png")

    bbox_img = draw_bboxes(rgb, [r.bbox for r in rois], thickness=2) if rois else rgb
    rois_path = save_rgb_png(bbox_img, outp / "mela_rois.png")

    # ROI crops
    roi_paths = []
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