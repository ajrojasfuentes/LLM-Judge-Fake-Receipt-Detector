"""
forensics_analysis.copy_move — Copy-Move forgery detection via ORB
keypoint matching and translation-vector clustering.

Copy-move forgery duplicates a region of the image and pastes it
elsewhere (often to hide or replicate content).  This module detects
such forgeries by:

1. Extracting ORB (Oriented FAST and Rotated BRIEF) keypoints from the
   grayscale image.
2. Matching keypoints against *themselves* using a brute-force Hamming
   distance matcher with Lowe's ratio test.
3. Clustering the resulting translation vectors (``dst - src``) into
   discrete spatial bins.
4. Selecting the top clusters that exceed a minimum match count and
   deriving bounding boxes for source/destination region pairs.

Outputs saved to *out_dir*:
    ``copymove_rois.png``         Original image with green bounding boxes
                                  around source and destination regions.
    ``copymove_pair_<k>.png``     Side-by-side crops of each detected
                                  source/destination pair.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .utils import crop_rgb, draw_bboxes, ensure_dir, save_rgb_png, to_gray_u8

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cluster_translations(
    dxy: np.ndarray, bin_px: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster match translation vectors via coarse spatial binning.

    Each translation vector ``(dx, dy)`` is discretised into a grid
    cell of side length *bin_px*.  Cells are counted and returned in
    descending order of frequency.

    Parameters
    ----------
    dxy : np.ndarray
        Array of shape ``(N, 2)`` — translation vectors.
    bin_px : int
        Bin side length in pixels.

    Returns
    -------
    bins : np.ndarray
        Unique bin coordinates sorted by descending count, shape
        ``(K, 2)``.
    counts : np.ndarray
        Corresponding match counts, shape ``(K,)``.
    """
    b = np.floor(dxy / float(bin_px)).astype(np.int32)
    uniq, inv, counts = np.unique(b, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)
    return uniq[order], counts[order]


def _bbox_from_pts(
    pts: np.ndarray, img_shape: Tuple[int, ...], pad: int = 12,
) -> Tuple[int, int, int, int]:
    """Compute a bounding box ``(x, y, w, h)`` around a set of points.

    Parameters
    ----------
    pts : np.ndarray
        Array of shape ``(N, 2)`` — ``(x, y)`` coordinates.
    img_shape : tuple of int
        ``(H, W, ...)`` shape of the source image, used for clamping.
    pad : int
        Extra padding in pixels on every side.

    Returns
    -------
    tuple of int
        ``(x, y, w, h)`` bounding box clamped to image bounds.
    """
    x0 = max(0, int(np.min(pts[:, 0]) - pad))
    y0 = max(0, int(np.min(pts[:, 1]) - pad))
    x1 = min(img_shape[1] - 1, int(np.max(pts[:, 0]) + pad))
    y1 = min(img_shape[0] - 1, int(np.max(pts[:, 1]) + pad))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def copy_move_detect(
    rgb: np.ndarray,
    out_dir: str,
    max_pairs: int = 2,
    min_matches_cluster: int = 18,
) -> Dict[str, Any]:
    """Detect copy-move forgery within a single image.

    Parameters
    ----------
    rgb : np.ndarray
        Input image of shape ``(H, W, 3)``, dtype ``uint8``.
    out_dir : str
        Directory where output artifact images will be saved.
    max_pairs : int
        Maximum number of source/destination pairs to return.
    min_matches_cluster : int
        Minimum number of keypoint matches required in a translation
        cluster to be considered a valid copy-move pair.

    Returns
    -------
    dict
        Keys:

        * ``"summary"`` — Scalar metrics (pairs found, good matches,
          keypoints count, or reason string if no pairs detected).
        * ``"pairs"`` — List of dicts, each with ``"src_bbox"``,
          ``"dst_bbox"``, ``"translation_bin"``, and
          ``"matches_in_cluster"``.
        * ``"artifacts"`` — Paths to saved PNG images.
    """
    outp = ensure_dir(out_dir)

    if cv2 is None:
        return {
            "summary": {"error": "cv2 not available"},
            "pairs": [],
            "artifacts": {},
        }

    gray = to_gray_u8(rgb)

    # ------------------------------------------------------------------
    # Step 1: ORB keypoint detection
    # ------------------------------------------------------------------
    orb = cv2.ORB_create(
        nfeatures=3500, scaleFactor=1.2, nlevels=8, fastThreshold=12,
    )
    kps, des = orb.detectAndCompute(gray, None)
    if des is None or len(kps) < 80:
        return {
            "summary": {"pairs_found": 0, "reason": "few_features"},
            "pairs": [],
            "artifacts": {},
        }

    # ------------------------------------------------------------------
    # Step 2: Self-matching with Lowe's ratio test
    # ------------------------------------------------------------------
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des, des, k=2)

    good: List[cv2.DMatch] = []
    for m, n in knn:
        # Skip self-matches (same keypoint index)
        if m.queryIdx == m.trainIdx:
            continue
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 40:
        return {
            "summary": {
                "pairs_found": 0,
                "reason": "few_good_matches",
                "good_matches": len(good),
            },
            "pairs": [],
            "artifacts": {},
        }

    # ------------------------------------------------------------------
    # Step 3: Translation-vector clustering
    # ------------------------------------------------------------------
    pts_q = np.float32([kps[m.queryIdx].pt for m in good])
    pts_t = np.float32([kps[m.trainIdx].pt for m in good])
    dxy = pts_t - pts_q  # shape (N, 2)

    bin_px = 14
    bins, counts = _cluster_translations(dxy, bin_px=bin_px)
    pairs: List[Dict[str, Any]] = []
    bboxes: List[Tuple[int, int, int, int]] = []

    used = 0
    for bi, cnt in zip(bins, counts):
        if used >= max_pairs:
            break
        if int(cnt) < min_matches_cluster:
            break

        # Select matches belonging to this bin
        mask = (np.floor(dxy / float(bin_px)).astype(np.int32) == bi).all(axis=1)
        idx = np.where(mask)[0]
        if idx.size < min_matches_cluster:
            continue

        src = pts_q[idx]
        dst = pts_t[idx]

        src_bb = _bbox_from_pts(src, rgb.shape)
        dst_bb = _bbox_from_pts(dst, rgb.shape)

        pairs.append({
            "translation_bin": [int(bi[0]), int(bi[1])],
            "matches_in_cluster": int(idx.size),
            "src_bbox": src_bb,
            "dst_bbox": dst_bb,
        })
        bboxes.extend([src_bb, dst_bb])
        used += 1

    if not pairs:
        return {
            "summary": {"pairs_found": 0, "reason": "no_cluster"},
            "pairs": [],
            "artifacts": {},
        }

    # ------------------------------------------------------------------
    # Step 4: Save visualisation artifacts
    # ------------------------------------------------------------------
    bbox_img = draw_bboxes(rgb, bboxes, thickness=2)
    rois_path = save_rgb_png(bbox_img, outp / "copymove_rois.png")

    pair_paths: List[str] = []
    for k, pr in enumerate(pairs, start=1):
        src_crop = crop_rgb(rgb, pr["src_bbox"], pad=6)
        dst_crop = crop_rgb(rgb, pr["dst_bbox"], pad=6)
        # Side-by-side canvas
        h = max(src_crop.shape[0], dst_crop.shape[0])
        w = src_crop.shape[1] + dst_crop.shape[1]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[: src_crop.shape[0], : src_crop.shape[1]] = src_crop
        canvas[: dst_crop.shape[0], src_crop.shape[1] : src_crop.shape[1] + dst_crop.shape[1]] = dst_crop
        pp = save_rgb_png(canvas, outp / f"copymove_pair_{k}.png")
        pair_paths.append(pp)

    return {
        "summary": {
            "pairs_found": int(len(pairs)),
            "good_matches": int(len(good)),
            "keypoints": int(len(kps)),
        },
        "pairs": pairs,
        "artifacts": {
            "copymove_rois": rois_path,
            "copymove_pair_grids": pair_paths,
        },
    }
