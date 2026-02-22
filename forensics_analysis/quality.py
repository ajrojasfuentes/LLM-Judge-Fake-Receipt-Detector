"""
forensics_analysis.quality — Global and per-tile image quality metrics.

Provides two functions:

* :func:`quality_metrics` — Fast global descriptors (brightness mean/std,
  contrast proxy, and Laplacian blur variance).
* :func:`blur_tile_stats` — Per-tile Laplacian variance statistics that
  can reveal locally blurred regions (e.g. an artificially smoothed
  pasted area inside an otherwise sharp document).

These metrics do not produce output files; they return scalar
dictionaries consumed by the evidence pack.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .utils import tile_view, to_gray_u8

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def quality_metrics(rgb: np.ndarray) -> Dict[str, Any]:
    """Compute global image quality descriptors.

    Parameters
    ----------
    rgb : np.ndarray
        Input image of shape ``(H, W, 3)``, dtype ``uint8``.

    Returns
    -------
    dict
        * ``"brightness_mean"`` — Mean pixel intensity (0-255).
        * ``"brightness_std"`` — Standard deviation of pixel intensity.
        * ``"contrast_std"`` — Simple contrast proxy (same as
          ``brightness_std``).
        * ``"blur_laplacian_var"`` — Variance of the Laplacian
          (higher = sharper).  ``None`` if OpenCV is unavailable.
    """
    gray = to_gray_u8(rgb)
    g = gray.astype(np.float32)

    brightness_mean = float(np.mean(g))
    brightness_std = float(np.std(g))

    # Simple contrast proxy: standard deviation of grayscale intensities
    contrast_std = brightness_std

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
    """Compute per-tile Laplacian blur variance and return statistics.

    Large differences between ``blur_tile_min`` and ``blur_tile_max``
    can indicate that parts of the image were blurred (e.g. to conceal
    an edit) while other parts remain sharp.

    Parameters
    ----------
    rgb : np.ndarray
        Input image of shape ``(H, W, 3)``, dtype ``uint8``.
    tile : int
        Side length of the square non-overlapping tiles (pixels).

    Returns
    -------
    dict
        * ``"blur_tile_mean"`` — Mean Laplacian variance across tiles.
        * ``"blur_tile_std"`` — Standard deviation across tiles.
        * ``"blur_tile_min"`` / ``"blur_tile_max"`` — Extremes.
        * ``"tile"`` — Tile size used.
        * ``"tiles_hw"`` — ``[n_rows, n_cols]`` tile grid dimensions.
        * ``"error"`` — Present only if OpenCV is unavailable.
    """
    gray = to_gray_u8(rgb)
    if cv2 is None:
        return {
            "error": "cv2 not available",
            "blur_tile_mean": None,
            "blur_tile_std": None,
            "blur_tile_min": None,
            "blur_tile_max": None,
        }

    tiles, nh, nw = tile_view(gray, tile, tile)
    vals = []
    for i in range(nh):
        for j in range(nw):
            t = tiles[i, j]
            lap = cv2.Laplacian(t, cv2.CV_64F)
            vals.append(lap.var())
    v = np.array(vals, dtype=np.float32)
    if v.size == 0:
        return {
            "blur_tile_mean": None,
            "blur_tile_std": None,
            "blur_tile_min": None,
            "blur_tile_max": None,
        }

    return {
        "blur_tile_mean": float(v.mean()),
        "blur_tile_std": float(v.std()),
        "blur_tile_min": float(v.min()),
        "blur_tile_max": float(v.max()),
        "tile": int(tile),
        "tiles_hw": [int(nh), int(nw)],
    }
