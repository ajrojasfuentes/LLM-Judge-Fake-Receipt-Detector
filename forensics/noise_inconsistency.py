"""
forensics_analysis.noise_inconsistency — Residual-noise-based per-tile
inconsistency analysis.

Forged images often contain regions with different noise characteristics
because pasted content originates from a different source (camera,
compression level, or synthetic generator).  This module estimates
per-tile noise levels and flags statistical outliers.

Algorithm
---------
1. Convert the image to grayscale float.
2. Compute the high-frequency residual by subtracting a Gaussian-blurred
   version of the image.
3. Divide the absolute residual into non-overlapping tiles of size
   *tile* x *tile*.
4. Compute the mean absolute residual per tile.
5. Z-score each tile's mean against the global tile distribution.
6. Up-scale the Z-score grid back to the original resolution and
   normalise the positive side to ``[0, 1]`` to produce a heatmap.

Outputs saved to *out_dir*:
    ``noise_heat.png``       Grayscale heatmap of tile-level noise
                             Z-scores (positive side only).
    ``noise_overlay.png``    Red-channel overlay of the heatmap on the
                             original image.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .utils import (
    ensure_dir,
    normalize_01,
    overlay_heatmap,
    save_gray_png,
    save_rgb_png,
    tile_view,
    to_gray_u8,
)

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def noise_inconsistency(
    rgb: np.ndarray,
    out_dir: str,
    tile: int = 64,
    sigma: float = 1.2,
) -> Dict[str, Any]:
    """Detect per-tile noise inconsistencies in an RGB image.

    Parameters
    ----------
    rgb : np.ndarray
        Input image of shape ``(H, W, 3)``, dtype ``uint8``.
    out_dir : str
        Directory where output artifact images will be saved.
    tile : int
        Side length of the square non-overlapping tiles (pixels).
    sigma : float
        Standard deviation of the Gaussian blur used to separate the
        low-frequency image component from the high-frequency noise
        residual.

    Returns
    -------
    dict
        Keys:

        * ``"summary"`` — Scalar metrics (tile size, sigma, maximum
          tile Z-score, mean and std of absolute residual per tile).
        * ``"artifacts"`` — Paths to saved heatmap and overlay PNGs.
    """
    outp = ensure_dir(out_dir)
    gray = to_gray_u8(rgb).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 1: Compute high-frequency residual
    # ------------------------------------------------------------------
    if cv2 is not None:
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    else:
        from scipy.ndimage import uniform_filter  # type: ignore

        blur = uniform_filter(gray, size=5)

    resid = gray - blur
    absr = np.abs(resid)

    # ------------------------------------------------------------------
    # Step 2: Per-tile mean absolute residual
    # ------------------------------------------------------------------
    tiles, nh, nw = tile_view(absr, tile, tile)
    if tiles.size == 0:
        return {
            "summary": {"error": "image too small for tiling"},
            "artifacts": {},
        }

    tmean = tiles.reshape(nh, nw, -1).mean(axis=2)
    mu = float(tmean.mean())
    sd = float(tmean.std() + 1e-6)
    z = (tmean - mu) / sd

    # ------------------------------------------------------------------
    # Step 3: Up-scale Z-scores and build heatmap
    # ------------------------------------------------------------------
    heat = np.kron(z, np.ones((tile, tile), dtype=np.float32))
    heat = heat[: gray.shape[0], : gray.shape[1]]
    # Keep only the positive (above-average) side for visualisation
    heat01 = normalize_01(np.maximum(0.0, heat))

    heat_path = save_gray_png(heat01, outp / "noise_heat.png")
    overlay = overlay_heatmap(rgb, heat01, alpha=0.45)
    overlay_path = save_rgb_png(overlay, outp / "noise_overlay.png")

    return {
        "summary": {
            "tile": int(tile),
            "sigma": float(sigma),
            "max_tile_z": float(np.max(z)),
            "mean_abs_resid": float(absr.mean()),
            "std_tile_mean_abs_resid": float(tmean.std()),
        },
        "artifacts": {
            "noise_heat": heat_path,
            "noise_overlay": overlay_path,
        },
    }
