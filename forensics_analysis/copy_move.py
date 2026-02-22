from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils import ROI, crop_rgb, draw_bboxes, ensure_dir, save_rgb_png, to_gray_u8

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def _cluster_translations(dxy: np.ndarray, bin_px: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster match translations via coarse binning in (dx,dy).
    Returns (bin_ids, counts) sorted desc.
    """
    # bin
    b = np.floor(dxy / float(bin_px)).astype(np.int32)
    # unique bins
    uniq, inv, counts = np.unique(b, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)
    return uniq[order], counts[order]


def copy_move_detect(
    rgb: np.ndarray,
    out_dir: str,
    max_pairs: int = 2,
    min_matches_cluster: int = 18,
) -> Dict[str, Any]:
    """
    Copy-move via ORB matching within same image:
      - ORB features
      - BFMatcher knn
      - cluster by translation vectors
      - return top cluster pairs as ROI pairs (src/dst)
    Saves:
      - copymove_rois.png (bboxes)
      - copymove_pair_<k>.png (side-by-side crops)
    """
    outp = ensure_dir(out_dir)

    if cv2 is None:
        return {"summary": {"error": "cv2 not available"}, "pairs": [], "artifacts": {}}

    gray = to_gray_u8(rgb)

    orb = cv2.ORB_create(nfeatures=3500, scaleFactor=1.2, nlevels=8, fastThreshold=12)
    kps, des = orb.detectAndCompute(gray, None)
    if des is None or len(kps) < 80:
        return {"summary": {"pairs_found": 0, "reason": "few_features"}, "pairs": [], "artifacts": {}}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des, des, k=2)

    good = []
    for m, n in knn:
        if m.queryIdx == m.trainIdx:
            continue
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 40:
        return {"summary": {"pairs_found": 0, "reason": "few_good_matches", "good_matches": len(good)}, "pairs": [], "artifacts": {}}

    # translation vectors
    pts_q = np.float32([kps[m.queryIdx].pt for m in good])
    pts_t = np.float32([kps[m.trainIdx].pt for m in good])
    dxy = pts_t - pts_q  # (N,2)

    bins, counts = _cluster_translations(dxy, bin_px=14)
    pairs = []
    bboxes = []

    used = 0
    for bi, cnt in zip(bins, counts):
        if used >= max_pairs:
            break
        if int(cnt) < min_matches_cluster:
            break

        # select matches in this bin
        mask = (np.floor(dxy / 14.0).astype(np.int32) == bi).all(axis=1)
        idx = np.where(mask)[0]
        if idx.size < min_matches_cluster:
            continue

        src = pts_q[idx]
        dst = pts_t[idx]

        # bbox around points (pad)
        def bbox_from_pts(pts: np.ndarray, pad: int = 12) -> Tuple[int, int, int, int]:
            x0 = max(0, int(np.min(pts[:, 0]) - pad))
            y0 = max(0, int(np.min(pts[:, 1]) - pad))
            x1 = min(rgb.shape[1] - 1, int(np.max(pts[:, 0]) + pad))
            y1 = min(rgb.shape[0] - 1, int(np.max(pts[:, 1]) + pad))
            return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

        src_bb = bbox_from_pts(src)
        dst_bb = bbox_from_pts(dst)

        pairs.append({
            "translation_bin": [int(bi[0]), int(bi[1])],
            "matches_in_cluster": int(idx.size),
            "src_bbox": src_bb,
            "dst_bbox": dst_bb,
        })
        bboxes.extend([src_bb, dst_bb])
        used += 1

    if not pairs:
        return {"summary": {"pairs_found": 0, "reason": "no_cluster"}, "pairs": [], "artifacts": {}}

    # Save bbox image
    bbox_img = draw_bboxes(rgb, bboxes, thickness=2)
    rois_path = save_rgb_png(bbox_img, outp / "copymove_rois.png")

    # Save pair crops
    pair_paths = []
    for k, pr in enumerate(pairs, start=1):
        src = crop_rgb(rgb, pr["src_bbox"], pad=6)
        dst = crop_rgb(rgb, pr["dst_bbox"], pad=6)
        # side-by-side
        h = max(src.shape[0], dst.shape[0])
        w = src.shape[1] + dst.shape[1]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[: src.shape[0], : src.shape[1]] = src
        canvas[: dst.shape[0], src.shape[1] : src.shape[1] + dst.shape[1]] = dst
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