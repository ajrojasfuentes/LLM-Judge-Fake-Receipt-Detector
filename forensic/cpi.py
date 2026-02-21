"""
Dense Block Copy-Paste Inside (CPI) detection for receipt forgery detection.

Detects intra-document copy-paste by:
  1. ROI gating  — combines anomaly ROIs from MELA, noise, and frequency analyses
     with an OCR-semantic bonus for total/tax zones.
  2. Dense block extraction — 16×16 blocks at stride 8 within gated ROIs.
  3. DCT low-frequency descriptor — 4×4 DCT coefficients (16 dims), L2-normalised.
  4. Lexicographic sort + neighbour-window search — finds similar block pairs
     without brute-force O(N²) comparison.
  5. Shift clustering — DBSCAN on (dx, dy) space; each cluster is one CPI hypothesis.
  6. NCC verification — validates each hypothesis by computing normalised
     cross-correlation on full-size patches; counts inliers (NCC > threshold).
  7. Anti-text heuristics — down-weights clusters whose cloned region is a single
     thin line (typical of repeated font patterns) unless it overlaps a total zone.
  8. Composite confidence score + output artefacts (mask, overlay, debug JSON).

Public API
----------
>>> from forensic.cpi import cpi_analyze, CPIResult, CPIPair
>>> result = cpi_analyze(
...     cropped_gray,
...     mela_rois=mela_result.top_rois,
...     noise_rois=noise_result.top_rois,
...     freq_rois=freq_result.top_rois,
...     mela_suspicious_ratio=mela_result.suspicious_ratio,
...     noise_anomalous_ratio=noise_result.anomalous_ratio,
...     freq_anomalous_ratio=freq_result.anomalous_ratio,
...     ocr_text=ocr_text,
...     output_dir=Path("out"),
... )
>>> print(result.level, result.confidence, result.best_shift)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .utils import ROI, compute_iou, merge_rois, rank_rois, save_image, fallback_bottom_roi

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CPIPair:
    """One verified copy-paste hypothesis (a single shift cluster)."""
    dest_bbox: Tuple[int, int, int, int]     # (x, y, w, h) in cropped image coords
    src_bbox: Tuple[int, int, int, int]
    shift: Tuple[int, int]                   # (dx, dy) best representative shift
    inlier_ratio: float                      # fraction of candidate pairs with NCC > threshold
    mean_ncc: float
    verified_pairs: int
    clone_area_pct: float                    # % of image area affected
    overlaps_total_zone: bool                # dest_bbox overlaps bottom-30% of image
    overlaps_tax_zone: bool
    cluster_size: int                        # number of raw candidate pairs in cluster
    penalised: bool = False                  # True if anti-text heuristic applied


@dataclass
class CPIResult:
    """Dense block CPI detection result."""

    # Top-level summary
    confidence: float           # 0–1 composite confidence
    level: str                  # "LOW" | "MOD" | "HIGH"
    best_shift: Optional[Tuple[int, int]]   # (dx, dy) of top hypothesis
    inlier_ratio: float
    clone_area_pct: float
    num_hypotheses: int         # number of verified shift clusters
    verified_pairs: int         # total inlier pairs across all hypotheses

    # Per-hypothesis details (up to 3)
    top_pairs: List[CPIPair]

    # Gating info (which ROIs were searched)
    gating_rois: List[ROI]

    # Visualisation arrays (None if no evidence found)
    mask: Optional[np.ndarray]      # H×W uint8: 128=src, 255=dest
    overlay: Optional[np.ndarray]   # BGR colour overlay on cropped image

    # Saved image paths
    saved_images: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable summary for logging / JSON output."""
        return {
            "confidence": round(self.confidence, 4),
            "level": self.level,
            "best_shift": list(self.best_shift) if self.best_shift else None,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "clone_area_pct": round(self.clone_area_pct, 4),
            "num_hypotheses": self.num_hypotheses,
            "verified_pairs": self.verified_pairs,
            "top_pairs": [
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
                for p in self.top_pairs
            ],
            "gating_rois": [
                {
                    "bbox": list(r.bbox),
                    "score": round(r.score, 4),
                    "source": r.source,
                    "notes": r.notes,
                }
                for r in self.gating_rois
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ncc(patch1: np.ndarray, patch2: np.ndarray) -> float:
    """Normalised cross-correlation of two equal-sized patches (float in [-1, 1])."""
    p1 = patch1.astype(np.float64)
    p2 = patch2.astype(np.float64)
    p1 -= p1.mean()
    p2 -= p2.mean()
    n1 = np.sqrt(np.sum(p1 ** 2))
    n2 = np.sqrt(np.sum(p2 ** 2))
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.sum(p1 * p2) / (n1 * n2))


def _dct_descriptor(
    block: np.ndarray,
    coeff_size: int = 4,
) -> np.ndarray:
    """Return an L2-normalised DCT low-frequency descriptor for a square block.

    Parameters
    ----------
    block : np.ndarray
        Float32 square block (block_size × block_size).
    coeff_size : int
        Take the top-left *coeff_size × coeff_size* DCT coefficients
        (excluding DC at [0,0] to reduce illumination sensitivity).

    Returns
    -------
    np.ndarray
        1-D float32 descriptor of length ``coeff_size**2 - 1``.
    """
    dct = cv2.dct(block.astype(np.float32))
    # Take top-left coeff_size×coeff_size, skip DC (0,0)
    coeffs = dct[:coeff_size, :coeff_size].ravel()[1:]   # exclude DC
    norm = np.linalg.norm(coeffs)
    if norm < 1e-8:
        return np.zeros_like(coeffs)
    return (coeffs / norm).astype(np.float32)


def _extract_blocks(
    gray: np.ndarray,
    roi_bbox: Tuple[int, int, int, int],
    block_size: int,
    stride: int,
    coeff_size: int,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Extract block descriptors and their top-left pixel coordinates within *roi_bbox*.

    Returns
    -------
    descriptors : np.ndarray  shape (N, D)
    positions   : list of (x_global, y_global) top-left pixel coords
    """
    x0, y0, bw, bh = roi_bbox
    h, w = gray.shape[:2]

    descriptors: List[np.ndarray] = []
    positions: List[Tuple[int, int]] = []

    y = y0
    while y + block_size <= min(y0 + bh, h):
        x = x0
        while x + block_size <= min(x0 + bw, w):
            block = gray[y : y + block_size, x : x + block_size].astype(np.float32)
            desc = _dct_descriptor(block, coeff_size)
            descriptors.append(desc)
            positions.append((x, y))
            x += stride
        y += stride

    if not descriptors:
        return np.zeros((0, coeff_size ** 2 - 1), dtype=np.float32), []

    return np.stack(descriptors, axis=0), positions


def _find_similar_pairs(
    descriptors: np.ndarray,
    positions: List[Tuple[int, int]],
    neighbor_window: int,
    min_shift_distance: int,
) -> List[Tuple[int, int, int, int]]:
    """Find similar block pairs using lexicographic sort + sliding window.

    Returns
    -------
    List of (idx_a, idx_b, dx, dy) for candidate pairs, where
    (dx, dy) = positions[idx_b] - positions[idx_a].
    """
    if len(descriptors) < 2:
        return []

    # Quantise descriptors to 2 decimal places for stable sorting
    quant = np.round(descriptors, 2)

    # Lexicographic sort
    # Convert to structured-array-like by converting rows to tuple keys
    sort_indices = np.lexsort(quant[:, ::-1].T)  # sort on first dim first

    pairs: List[Tuple[int, int, int, int]] = []
    n = len(sort_indices)
    W = min(neighbor_window, n - 1)

    for rank_i in range(n):
        idx_a = sort_indices[rank_i]
        xa, ya = positions[idx_a]

        for rank_j in range(rank_i + 1, min(rank_i + 1 + W, n)):
            idx_b = sort_indices[rank_j]
            xb, yb = positions[idx_b]

            dx = xb - xa
            dy = yb - ya
            shift_dist = (dx ** 2 + dy ** 2) ** 0.5

            if shift_dist < min_shift_distance:
                continue  # too close / same-block

            # Quick descriptor similarity check (dot product on L2-normalised)
            sim = float(np.dot(descriptors[idx_a], descriptors[idx_b]))
            if sim < 0.90:           # loose threshold; NCC will verify
                continue

            pairs.append((idx_a, idx_b, dx, dy))

    return pairs


def _dbscan_shifts(
    pairs: List[Tuple[int, int, int, int]],
    eps: float,
    min_samples: int,
) -> Dict[int, List[int]]:
    """Cluster candidate pairs by (dx, dy) using DBSCAN.

    Returns
    -------
    Dict mapping cluster_id (>=0) to list of pair indices.
    Noise points (cluster_id == -1) are excluded.
    """
    if not pairs:
        return {}

    shifts = np.array([[p[2], p[3]] for p in pairs], dtype=np.float32)

    # Try sklearn DBSCAN; fall back to histogram-peak clustering
    try:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(shifts)
    except ImportError:
        labels = _histogram_cluster(shifts, eps=eps, min_samples=min_samples)

    clusters: Dict[int, List[int]] = {}
    for pair_idx, label in enumerate(labels):
        if label < 0:
            continue
        clusters.setdefault(int(label), []).append(pair_idx)

    return clusters


def _histogram_cluster(
    shifts: np.ndarray,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """Fallback clustering: round shifts to eps grid, group by cell, label by size."""
    labels = np.full(len(shifts), -1, dtype=int)
    if len(shifts) == 0:
        return labels

    # Quantise to grid of size eps
    grid = np.round(shifts / eps).astype(int)
    cell_to_indices: Dict[Tuple[int, int], List[int]] = {}
    for i, (gx, gy) in enumerate(grid):
        key = (int(gx), int(gy))
        cell_to_indices.setdefault(key, []).append(i)

    cluster_id = 0
    for key, indices in cell_to_indices.items():
        if len(indices) >= min_samples:
            for i in indices:
                labels[i] = cluster_id
            cluster_id += 1

    return labels


def _verify_cluster(
    cluster_pair_indices: List[int],
    pairs: List[Tuple[int, int, int, int]],
    positions: List[Tuple[int, int]],
    gray: np.ndarray,
    patch_size: int,
    ncc_threshold: float,
    max_sample: int,
) -> Tuple[float, float, int, Tuple[int, int]]:
    """NCC-verify a cluster of candidate pairs.

    Returns
    -------
    inlier_ratio, mean_ncc, verified_pairs, representative_shift (dx, dy)
    """
    h, w = gray.shape[:2]
    half = patch_size // 2

    # Sample up to max_sample pairs
    sample_indices = cluster_pair_indices[:max_sample]

    ncc_scores: List[float] = []
    shift_list: List[Tuple[int, int]] = []

    for pair_idx in sample_indices:
        idx_a, idx_b, dx, dy = pairs[pair_idx]
        xa, ya = positions[idx_a]
        xb, yb = positions[idx_b]

        # Centre the patch on the block centre
        cxa, cya = xa + half, ya + half
        cxb, cyb = xb + half, yb + half

        # Bounds check
        if (cya - half < 0 or cya + half > h or cxa - half < 0 or cxa + half > w or
                cyb - half < 0 or cyb + half > h or cxb - half < 0 or cxb + half > w):
            continue

        patch_a = gray[cya - half : cya + half, cxa - half : cxa + half]
        patch_b = gray[cyb - half : cyb + half, cxb - half : cxb + half]

        score = _ncc(patch_a, patch_b)
        ncc_scores.append(score)
        shift_list.append((dx, dy))

    if not ncc_scores:
        return 0.0, 0.0, 0, (0, 0)

    ncc_arr = np.array(ncc_scores)
    inliers = int((ncc_arr >= ncc_threshold).sum())
    inlier_ratio = inliers / len(ncc_arr)
    mean_ncc = float(ncc_arr.mean())
    verified_pairs = inliers

    # Representative shift = median over all shift samples
    if shift_list:
        dxs = [s[0] for s in shift_list]
        dys = [s[1] for s in shift_list]
        rep_shift = (int(np.median(dxs)), int(np.median(dys)))
    else:
        rep_shift = (0, 0)

    return inlier_ratio, mean_ncc, verified_pairs, rep_shift


def _build_region_bbox(
    inlier_pair_indices: List[int],
    pairs: List[Tuple[int, int, int, int]],
    positions: List[Tuple[int, int]],
    block_size: int,
    image_h: int,
    image_w: int,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Build src_bbox and dest_bbox from inlier pair block positions."""
    src_xs, src_ys = [], []
    dst_xs, dst_ys = [], []

    for pair_idx in inlier_pair_indices:
        idx_a, idx_b, dx, dy = pairs[pair_idx]
        xa, ya = positions[idx_a]
        xb, yb = positions[idx_b]

        # Treat the pair with larger coordinates as "destination"
        if (abs(dx) + abs(dy)) > 0:
            src_xs.append(xa); src_ys.append(ya)
            dst_xs.append(xb); dst_ys.append(yb)
        else:
            src_xs.append(xa); src_ys.append(ya)
            dst_xs.append(xb); dst_ys.append(yb)

    def _bbox_from_coords(xs, ys):
        if not xs:
            return (0, 0, 1, 1)
        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2 = min(image_w, max(xs) + block_size)
        y2 = min(image_h, max(ys) + block_size)
        return (x1, y1, x2 - x1, y2 - y1)

    src_bbox = _bbox_from_coords(src_xs, src_ys)
    dst_bbox = _bbox_from_coords(dst_xs, dst_ys)
    return src_bbox, dst_bbox


def _is_near_bottom(
    bbox: Tuple[int, int, int, int],
    image_h: int,
    fraction: float = 0.30,
) -> bool:
    """Return True if the bounding box overlaps the bottom fraction of the image."""
    x, y, bw, bh = bbox
    return (y + bh) > image_h * (1.0 - fraction)


def _is_thin_line(
    bbox: Tuple[int, int, int, int],
    block_size: int,
) -> bool:
    """Return True if the bounding box is a single thin line of blocks."""
    _, _, bw, bh = bbox
    return bh < 2 * block_size or bw < 2 * block_size


def _ocr_has_keywords(
    ocr_text: Optional[str],
    keywords: List[str],
) -> bool:
    if not ocr_text:
        return False
    text_lower = ocr_text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _semantic_overlap_bonus(
    bbox: Tuple[int, int, int, int],
    image_h: int,
    ocr_text: Optional[str],
) -> float:
    """Compute OCR-semantic bonus for a ROI: total/tax zone proximity."""
    bonus = 0.0
    if _is_near_bottom(bbox, image_h, fraction=0.30):
        bonus += 0.06
    if _ocr_has_keywords(ocr_text, ["total", "grand total", "amount due", "amount"]):
        if _is_near_bottom(bbox, image_h, fraction=0.30):
            bonus += 0.04
    if _ocr_has_keywords(ocr_text, ["tax", "gst", "vat"]):
        if _is_near_bottom(bbox, image_h, fraction=0.35):
            bonus += 0.02
    return min(bonus, 0.10)


# ─────────────────────────────────────────────────────────────────────────────
# ROI Gating
# ─────────────────────────────────────────────────────────────────────────────

def _build_gating_rois(
    image_h: int,
    image_w: int,
    mela_rois: Optional[List[ROI]],
    noise_rois: Optional[List[ROI]],
    freq_rois: Optional[List[ROI]],
    mela_suspicious_ratio: float,
    noise_anomalous_ratio: float,
    freq_anomalous_ratio: float,
    ocr_text: Optional[str],
    max_k: int,
    iou_threshold: float,
    use_bottom_fallback: bool,
) -> List[ROI]:
    """Merge anomaly ROIs from all forensic tools into a ranked gating list.

    Score formula (per ROI):
        0.45 * mela_score_norm + 0.25 * noise_score_norm
        + 0.20 * freq_score_norm + 0.10 * ocr_semantic_bonus
    """
    total_area = max(image_h * image_w, 1)

    # Normalise per-source scores to [0, 1] for weighting
    def _norm_score(rois: Optional[List[ROI]], global_ratio: float) -> List[Tuple[ROI, float]]:
        if not rois:
            return []
        max_s = max((r.score for r in rois), default=1.0)
        if max_s < 1e-8:
            max_s = 1.0
        out = []
        for r in rois:
            ns = r.score / max_s * (0.5 + 0.5 * global_ratio)
            out.append((r, min(ns, 1.0)))
        return out

    scored_mela = _norm_score(mela_rois, mela_suspicious_ratio)
    scored_noise = _norm_score(noise_rois, noise_anomalous_ratio)
    scored_freq = _norm_score(freq_rois, freq_anomalous_ratio)

    combined: List[ROI] = []

    for (roi, ns) in scored_mela:
        ocr_bonus = _semantic_overlap_bonus(roi.bbox, image_h, ocr_text)
        final_score = 0.45 * ns + 0.10 * ocr_bonus
        combined.append(ROI(
            bbox=roi.bbox,
            score=final_score,
            area_pct=roi.area_pct,
            source="mela",
            notes=roi.notes,
        ))

    for (roi, ns) in scored_noise:
        ocr_bonus = _semantic_overlap_bonus(roi.bbox, image_h, ocr_text)
        final_score = 0.25 * ns + 0.10 * ocr_bonus
        combined.append(ROI(
            bbox=roi.bbox,
            score=final_score,
            area_pct=roi.area_pct,
            source="noise",
            notes=roi.notes,
        ))

    for (roi, ns) in scored_freq:
        ocr_bonus = _semantic_overlap_bonus(roi.bbox, image_h, ocr_text)
        final_score = 0.20 * ns + 0.10 * ocr_bonus
        combined.append(ROI(
            bbox=roi.bbox,
            score=final_score,
            area_pct=roi.area_pct,
            source="freq",
            notes=roi.notes,
        ))

    # Merge overlapping ROIs, keep top-K
    merged = merge_rois(combined, iou_threshold=iou_threshold)
    gating = rank_rois(merged, max_k=max_k)

    # Fallback: always include bottom-25% band if not already represented
    if use_bottom_fallback:
        bottom_roi = fallback_bottom_roi(image_h, image_w, fraction=0.25)
        # Only add if not already substantially covered
        covered = any(
            compute_iou(r.bbox, bottom_roi.bbox) > 0.3
            for r in gating
        )
        if not covered:
            gating.append(bottom_roi)

    return gating


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_overlay(
    gray: np.ndarray,
    pairs: List[CPIPair],
) -> np.ndarray:
    """Build a BGR colour overlay showing src (blue) and dest (red) regions."""
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = bgr.copy()

    for pair in pairs:
        dx, dy, dw, dh = pair.dest_bbox
        sx, sy, sw, sh = pair.src_bbox
        # Destination: red tint
        overlay[dy : dy + dh, dx : dx + dw] = cv2.addWeighted(
            overlay[dy : dy + dh, dx : dx + dw], 0.55,
            np.full((dh, dw, 3), (0, 0, 200), dtype=np.uint8), 0.45, 0,
        )
        # Source: blue tint
        overlay[sy : sy + sh, sx : sx + sw] = cv2.addWeighted(
            overlay[sy : sy + sh, sx : sx + sw], 0.55,
            np.full((sh, sw, 3), (200, 0, 0), dtype=np.uint8), 0.45, 0,
        )
        # Draw rectangles
        cv2.rectangle(overlay, (dx, dy), (dx + dw, dy + dh), (0, 0, 255), 2)
        cv2.rectangle(overlay, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
        cv2.putText(overlay, "DEST", (dx, max(dy - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, "SRC", (sx, max(sy - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    return overlay


def _build_mask(
    gray: np.ndarray,
    pairs: List[CPIPair],
) -> np.ndarray:
    """Build a grayscale mask: 255=destination, 128=source, 0=background."""
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    for pair in pairs:
        dx, dy, dw, dh = pair.dest_bbox
        sx, sy, sw, sh = pair.src_bbox
        mask[sy : sy + sh, sx : sx + sw] = 128
        mask[dy : dy + dh, dx : dx + dw] = 255
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Empty result helper
# ─────────────────────────────────────────────────────────────────────────────

def _empty_result(gating_rois: Optional[List[ROI]] = None) -> CPIResult:
    return CPIResult(
        confidence=0.0,
        level="LOW",
        best_shift=None,
        inlier_ratio=0.0,
        clone_area_pct=0.0,
        num_hypotheses=0,
        verified_pairs=0,
        top_pairs=[],
        gating_rois=gating_rois or [],
        mask=None,
        overlay=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def cpi_analyze(
    cropped_gray: np.ndarray,
    *,
    # ROIs and global anomaly ratios from other tools
    mela_rois: Optional[List[ROI]] = None,
    noise_rois: Optional[List[ROI]] = None,
    freq_rois: Optional[List[ROI]] = None,
    mela_suspicious_ratio: float = 0.0,
    noise_anomalous_ratio: float = 0.0,
    freq_anomalous_ratio: float = 0.0,
    ocr_text: Optional[str] = None,
    # Output
    output_dir: Optional[Path] = None,
    prefix: str = "",
    # Block extraction
    block_size: int = 16,
    stride: int = 8,
    dct_coeff_size: int = 4,
    # Similarity search
    neighbor_window: int = 20,
    min_shift_distance: int = 15,
    # Shift clustering
    dbscan_eps: float = 5.0,
    dbscan_min_samples: int = 20,
    # NCC verification
    ncc_threshold: float = 0.85,
    ncc_sample_pairs: int = 50,
    patch_size: int = 24,
    max_hypotheses: int = 3,
    # Gating
    max_gating_rois: int = 8,
    gating_iou_threshold: float = 0.3,
    use_bottom_fallback: bool = True,
) -> CPIResult:
    """Run dense block copy-paste inside (CPI) detection.

    Parameters
    ----------
    cropped_gray : np.ndarray
        Grayscale receipt image (uint8, single-channel), ideally already
        cropped by ``crop_receipt``.
    mela_rois, noise_rois, freq_rois : list of ROI, optional
        Anomaly ROIs from the corresponding analysis modules.
    mela_suspicious_ratio, noise_anomalous_ratio, freq_anomalous_ratio : float
        Global anomaly ratios used to weight the per-source ROI scores.
    ocr_text : str, optional
        Raw OCR text of the receipt (used for semantic zone detection).
    output_dir : Path, optional
        Directory to save output images and debug JSON.
    prefix : str
        Filename prefix for saved outputs.
    block_size : int
        Block side length in pixels (default 16).
    stride : int
        Extraction stride (default 8, giving 50 % overlap).
    dct_coeff_size : int
        Number of DCT rows/columns to take as descriptor (default 4 → 15 dims).
    neighbor_window : int
        Neighbour window for sorted-descriptor search (default 20).
    min_shift_distance : int
        Minimum pixel shift to consider a pair (excludes trivial self-matches).
    dbscan_eps : float
        DBSCAN epsilon in shift-space pixels (default 5).
    dbscan_min_samples : int
        DBSCAN minimum cluster size (default 20).
    ncc_threshold : float
        NCC threshold for counting a pair as an inlier (default 0.85).
    ncc_sample_pairs : int
        Max pairs to NCC-verify per hypothesis (default 50).
    patch_size : int
        Patch radius for NCC computation (default 24).
    max_hypotheses : int
        Maximum number of shift hypotheses to verify (default 3).
    max_gating_rois : int
        Maximum number of gating ROIs to search (default 8).
    gating_iou_threshold : float
        IoU threshold for merging overlapping gating ROIs.
    use_bottom_fallback : bool
        Always include a bottom-quarter fallback ROI in the gating set.

    Returns
    -------
    CPIResult
    """
    h, w = cropped_gray.shape[:2]
    total_pixels = max(h * w, 1)

    if h < block_size * 4 or w < block_size * 4:
        return _empty_result()

    # ------------------------------------------------------------------
    # 0. Normalise image (CLAHE + light median denoise)
    # ------------------------------------------------------------------
    gray = cropped_gray.copy()
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    # Light CLAHE (clip_limit=1.5) for equalization without artefacts
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Light median denoising
    gray_proc = cv2.medianBlur(gray_eq, 3)

    # ------------------------------------------------------------------
    # 1. ROI gating
    # ------------------------------------------------------------------
    gating_rois = _build_gating_rois(
        image_h=h,
        image_w=w,
        mela_rois=mela_rois,
        noise_rois=noise_rois,
        freq_rois=freq_rois,
        mela_suspicious_ratio=mela_suspicious_ratio,
        noise_anomalous_ratio=noise_anomalous_ratio,
        freq_anomalous_ratio=freq_anomalous_ratio,
        ocr_text=ocr_text,
        max_k=max_gating_rois,
        iou_threshold=gating_iou_threshold,
        use_bottom_fallback=use_bottom_fallback,
    )

    if not gating_rois:
        return _empty_result()

    # ------------------------------------------------------------------
    # 2. Extract blocks from all gating ROIs (deduplicated by position)
    # ------------------------------------------------------------------
    all_descriptors: List[np.ndarray] = []
    all_positions: List[Tuple[int, int]] = []
    seen_positions: set = set()

    for roi in gating_rois:
        descs, positions = _extract_blocks(
            gray_proc, roi.bbox, block_size, stride, dct_coeff_size,
        )
        for desc, pos in zip(descs, positions):
            if pos not in seen_positions:
                seen_positions.add(pos)
                all_descriptors.append(desc)
                all_positions.append(pos)

    if len(all_descriptors) < 2:
        return _empty_result(gating_rois)

    descriptors = np.stack(all_descriptors, axis=0)

    # ------------------------------------------------------------------
    # 3. Find similar block pairs
    # ------------------------------------------------------------------
    candidate_pairs = _find_similar_pairs(
        descriptors, all_positions, neighbor_window, min_shift_distance,
    )

    if not candidate_pairs:
        return _empty_result(gating_rois)

    # ------------------------------------------------------------------
    # 4. Cluster by (dx, dy)
    # ------------------------------------------------------------------
    clusters = _dbscan_shifts(candidate_pairs, eps=dbscan_eps, min_samples=dbscan_min_samples)

    if not clusters:
        return _empty_result(gating_rois)

    # Sort clusters by size (largest first), take top-N
    sorted_clusters = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
    top_clusters = sorted_clusters[:max_hypotheses]

    # ------------------------------------------------------------------
    # 5. NCC verification per cluster
    # ------------------------------------------------------------------
    total_candidate_pairs = len(candidate_pairs)
    verified_hypotheses: List[CPIPair] = []
    best_inlier_ratio = 0.0
    best_shift: Optional[Tuple[int, int]] = None
    total_verified = 0

    for cluster_id, pair_indices in top_clusters:
        inlier_ratio, mean_ncc, verified_pairs, rep_shift = _verify_cluster(
            cluster_pair_indices=pair_indices,
            pairs=candidate_pairs,
            positions=all_positions,
            gray=gray,
            patch_size=patch_size,
            ncc_threshold=ncc_threshold,
            max_sample=ncc_sample_pairs,
        )

        if inlier_ratio < 0.10 or verified_pairs < 3:
            continue  # not enough evidence

        # Build source/dest bboxes from inlier pairs
        inlier_pair_indices = [
            pair_indices[k] for k in range(min(ncc_sample_pairs, len(pair_indices)))
        ]
        src_bbox, dest_bbox = _build_region_bbox(
            inlier_pair_indices, candidate_pairs, all_positions,
            block_size, h, w,
        )

        # Compute clone area
        _, _, dw, dh = dest_bbox
        clone_area_pct = float(dw * dh / total_pixels * 100.0)

        # Semantic zone checks
        overlaps_total = _is_near_bottom(dest_bbox, h, fraction=0.30)
        overlaps_tax = _is_near_bottom(dest_bbox, h, fraction=0.35)

        # Anti-text heuristic
        penalised = False
        if _is_thin_line(dest_bbox, block_size) and not (overlaps_total or overlaps_tax):
            penalised = True
            inlier_ratio *= 0.5   # down-weight

        cluster_strength = len(pair_indices) / max(total_candidate_pairs, 1)

        # Semantic bonus
        ocr_bonus = (0.10 if (overlaps_total and _ocr_has_keywords(
            ocr_text, ["total", "grand total", "amount due", "amount"])) else 0.0)
        ocr_bonus += (0.05 if (overlaps_tax and _ocr_has_keywords(
            ocr_text, ["tax", "gst", "vat"])) else 0.0)

        # Per-pair confidence
        clone_area_norm = min(clone_area_pct / 5.0, 1.0)  # normalise, cap at 1
        pair_confidence = (
            0.40 * inlier_ratio
            + 0.25 * clone_area_norm
            + 0.20 * cluster_strength
            + 0.15 * min(ocr_bonus / 0.15, 1.0)
        )

        hypothesis = CPIPair(
            dest_bbox=dest_bbox,
            src_bbox=src_bbox,
            shift=rep_shift,
            inlier_ratio=inlier_ratio,
            mean_ncc=mean_ncc,
            verified_pairs=verified_pairs,
            clone_area_pct=clone_area_pct,
            overlaps_total_zone=overlaps_total,
            overlaps_tax_zone=overlaps_tax,
            cluster_size=len(pair_indices),
            penalised=penalised,
        )
        verified_hypotheses.append(hypothesis)
        total_verified += verified_pairs

        if inlier_ratio > best_inlier_ratio:
            best_inlier_ratio = inlier_ratio
            best_shift = rep_shift

        # Early exit: strong evidence at total zone
        if pair_confidence >= 0.65 and (overlaps_total or overlaps_tax):
            break

    if not verified_hypotheses:
        return _empty_result(gating_rois)

    # ------------------------------------------------------------------
    # 6. Composite confidence
    # ------------------------------------------------------------------
    top_pair = verified_hypotheses[0]
    cluster_strength = top_pair.cluster_size / max(total_candidate_pairs, 1)
    clone_area_norm = min(top_pair.clone_area_pct / 5.0, 1.0)
    ocr_bonus = 0.0
    if top_pair.overlaps_total_zone and _ocr_has_keywords(
            ocr_text, ["total", "grand total", "amount due"]):
        ocr_bonus = 0.10
    elif top_pair.overlaps_tax_zone and _ocr_has_keywords(ocr_text, ["tax", "gst", "vat"]):
        ocr_bonus = 0.05

    confidence = (
        0.40 * top_pair.inlier_ratio
        + 0.25 * clone_area_norm
        + 0.20 * cluster_strength
        + 0.15 * min(ocr_bonus / 0.15, 1.0)
    )
    confidence = min(float(confidence), 1.0)

    # Override to HIGH if strong individual signals
    if top_pair.inlier_ratio >= 0.80 and top_pair.clone_area_pct >= 0.6:
        confidence = max(confidence, 0.65)

    if confidence >= 0.65:
        level = "HIGH"
    elif confidence >= 0.40:
        level = "MOD"
    else:
        level = "LOW"

    # ------------------------------------------------------------------
    # 7. Build visualisations
    # ------------------------------------------------------------------
    overlay = _build_overlay(gray, verified_hypotheses[:3])
    mask = _build_mask(gray, verified_hypotheses[:3])

    # ------------------------------------------------------------------
    # 8. Save images and debug JSON
    # ------------------------------------------------------------------
    saved_images: Dict[str, str] = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        name_overlay = f"{prefix}_cpi_overlay.png" if prefix else "cpi_overlay.png"
        name_mask = f"{prefix}_cpi_mask.png" if prefix else "cpi_mask.png"
        name_debug = f"{prefix}_cpi_debug.json" if prefix else "cpi_debug.json"

        p_ov = save_image(overlay, output_dir, name_overlay)
        saved_images["overlay"] = str(p_ov)

        p_mk = save_image(mask, output_dir, name_mask)
        saved_images["mask"] = str(p_mk)

        # Debug JSON
        result_tmp = CPIResult(
            confidence=confidence,
            level=level,
            best_shift=best_shift,
            inlier_ratio=top_pair.inlier_ratio,
            clone_area_pct=top_pair.clone_area_pct,
            num_hypotheses=len(verified_hypotheses),
            verified_pairs=total_verified,
            top_pairs=verified_hypotheses[:3],
            gating_rois=gating_rois,
            mask=None,
            overlay=None,
        )
        debug_path = output_dir / name_debug
        try:
            with open(debug_path, "w") as f:
                json.dump(result_tmp.to_dict(), f, indent=2)
            saved_images["debug_json"] = str(debug_path)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 9. Assemble final result
    # ------------------------------------------------------------------
    return CPIResult(
        confidence=confidence,
        level=level,
        best_shift=best_shift,
        inlier_ratio=top_pair.inlier_ratio,
        clone_area_pct=top_pair.clone_area_pct,
        num_hypotheses=len(verified_hypotheses),
        verified_pairs=total_verified,
        top_pairs=verified_hypotheses[:3],
        gating_rois=gating_rois,
        mask=mask,
        overlay=overlay,
        saved_images=saved_images,
    )
