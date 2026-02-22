from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .utils import to_gray_u8, tile_view

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def quality_metrics(rgb: np.ndarray) -> Dict[str, Any]:
    gray = to_gray_u8(rgb)
    g = gray.astype(np.float32)

    brightness_mean = float(np.mean(g))
    brightness_std = float(np.std(g))

    contrast_std = brightness_std  # simple proxy (std of grayscale)

    blur_laplacian_var = None
    if cv2 is not None:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        blur_laplacian_var = float(lap.var())

    return {
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "contrast_std": float(contrast_std),
        "blur_laplacian_var": blur_laplacian_var,
    }


def blur_tile_stats(rgb: np.ndarray, tile: int = 64) -> Dict[str, Any]:
    """Compute blur (variance of Laplacian) per-tile and return std/min/max + tile_mean."""
    gray = to_gray_u8(rgb)
    if cv2 is None:
        return {"error": "cv2 not available", "blur_tile_mean": None, "blur_tile_std": None, "blur_tile_min": None, "blur_tile_max": None}

    tiles, nh, nw = tile_view(gray, tile, tile)  # (nh, nw, th, tw)
    vals = []
    for i in range(nh):
        for j in range(nw):
            t = tiles[i, j]
            lap = cv2.Laplacian(t, cv2.CV_64F)
            vals.append(lap.var())
    v = np.array(vals, dtype=np.float32)
    if v.size == 0:
        return {"blur_tile_mean": None, "blur_tile_std": None, "blur_tile_min": None, "blur_tile_max": None}

    return {
        "blur_tile_mean": float(v.mean()),
        "blur_tile_std": float(v.std()),
        "blur_tile_min": float(v.min()),
        "blur_tile_max": float(v.max()),
        "tile": int(tile),
        "tiles_hw": [int(nh), int(nw)],
    }