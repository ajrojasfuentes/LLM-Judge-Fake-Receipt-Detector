from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .utils import ensure_dir, normalize_01, overlay_heatmap, save_gray_png, save_rgb_png, tile_view, to_gray_u8

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
    """
    Residual-based inconsistency:
      residual = gray - gaussian_blur(gray)
      compute per-tile mean(abs(residual)) and z-score vs global.
    Saves:
      - noise_heat.png
      - noise_overlay.png
    """
    outp = ensure_dir(out_dir)
    gray = to_gray_u8(rgb).astype(np.float32)

    if cv2 is not None:
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    else:
        # fallback simple box blur
        from scipy.ndimage import uniform_filter  # type: ignore
        blur = uniform_filter(gray, size=5)

    resid = gray - blur
    absr = np.abs(resid)

    tiles, nh, nw = tile_view(absr, tile, tile)  # (nh,nw,th,tw)
    if tiles.size == 0:
        return {"summary": {"error": "image too small for tiling"}, "artifacts": {}}

    # per tile score
    tmean = tiles.reshape(nh, nw, -1).mean(axis=2)
    mu = float(tmean.mean())
    sd = float(tmean.std() + 1e-6)
    z = (tmean - mu) / sd

    # upscale z to full size heatmap
    heat = np.kron(z, np.ones((tile, tile), dtype=np.float32))
    heat = heat[: gray.shape[0], : gray.shape[1]]
    heat01 = normalize_01(np.maximum(0.0, heat))  # show only high side

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