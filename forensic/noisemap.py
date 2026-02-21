"""
Block-based noise variance analysis for receipt forgery detection.

Computes a local noise variance map by subtracting a Gaussian-denoised
baseline from the original grayscale image and measuring per-block variance
of the residual.  Regions that were pasted from a different source, or
manipulated with image-editing tools, typically exhibit a noise profile
that differs from the surrounding authentic content.

Enhanced over the original ``forensic_analysis.NoiseMapResult`` with:
  - ROI extraction from connected groups of anomalous blocks.
  - Richer statistics (percentiles, anomalous ratio).
  - Configurable minimum group size for ROI filtering.
  - Optional image saving with ROI overlay visualisation.

Usage
-----
    from forensic.noisemap import noise_analyze
    result = noise_analyze(cropped_gray, output_dir=Path("out"), prefix="rcpt")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from forensic.utils import ROI, apply_colormap, normalize_to_uint8, save_image


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class NoiseResult:
    """Enhanced noise analysis result."""

    # Maps
    noise_map: np.ndarray       # Grayscale noise variance map resized to image (uint8)
    noise_color: np.ndarray     # Color-mapped heatmap (BGR)

    # Block-level data
    block_variances: np.ndarray  # Raw variance per block (float64)

    # Statistics
    global_mean_var: float
    global_std_var: float
    anomalous_blocks: int
    total_blocks: int
    block_size_used: int
    anomalous_ratio: float       # anomalous_blocks / total_blocks

    # Enhanced statistics
    percentiles: Dict[str, float]  # p50, p75, p90, p95 of block variances

    # ROIs from anomalous regions
    top_rois: List[ROI]

    # Saved image paths
    saved_images: Dict[str, str]


# ---------------------------------------------------------------------------
# Helpers (internal)
# ---------------------------------------------------------------------------

def _connected_components_block(binary: np.ndarray) -> List[List[tuple]]:
    """Return connected components of *True* cells in a 2-D boolean array.

    Each component is a list of ``(row, col)`` block indices.  Uses simple
    BFS with 4-connectivity.
    """
    rows, cols = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    components: List[List[tuple]] = []

    for r in range(rows):
        for c in range(cols):
            if binary[r, c] and not visited[r, c]:
                # BFS
                component: List[tuple] = []
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    component.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and binary[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                components.append(component)

    return components


def _draw_rois(
    gray_image: np.ndarray,
    rois: List[ROI],
    color: tuple = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw ROI bounding boxes on a BGR copy of *gray_image*."""
    if len(gray_image.shape) == 2:
        canvas = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = gray_image.copy()

    for roi in rois:
        x, y, bw, bh = roi.bbox
        x1, y1, x2, y2 = x, y, x + bw, y + bh
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        # Small label with score
        label = f"{roi.score:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        # Background rectangle for text readability
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def noise_analyze(
    cropped_gray: np.ndarray,
    *,
    output_dir: Optional[Path] = None,
    prefix: str = "",
    block_size: int = 32,
    max_rois: int = 5,
    min_anomalous_group_blocks: int = 2,
) -> NoiseResult:
    """Perform enhanced block-based noise variance analysis.

    Parameters
    ----------
    cropped_gray:
        Single-channel (H, W) grayscale image, uint8 or float.
    output_dir:
        If provided, save diagnostic images to this directory.
    prefix:
        Filename prefix for saved images.
    block_size:
        Side length of each analysis block in pixels.
    max_rois:
        Maximum number of ROIs to return (sorted by mean variance).
    min_anomalous_group_blocks:
        Minimum number of connected anomalous blocks required to form an ROI.

    Returns
    -------
    NoiseResult
        Dataclass with maps, statistics, ROIs, and saved image paths.
    """

    h, w = cropped_gray.shape[:2]

    # ------------------------------------------------------------------
    # 1. Convert to float32
    # ------------------------------------------------------------------
    gray_f = cropped_gray.astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Compute denoised version (Gaussian blur)
    # ------------------------------------------------------------------
    ksize = max(3, (block_size // 4) | 1)  # ensure odd
    denoised = cv2.GaussianBlur(gray_f, (ksize, ksize), 0)

    # ------------------------------------------------------------------
    # 3. Noise residual
    # ------------------------------------------------------------------
    residual = gray_f - denoised

    # ------------------------------------------------------------------
    # 4. Divide into blocks and compute variance per block
    # ------------------------------------------------------------------
    rows = h // block_size
    cols = w // block_size

    block_variances = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            block = residual[
                r * block_size : (r + 1) * block_size,
                c * block_size : (c + 1) * block_size,
            ]
            block_variances[r, c] = float(np.var(block))

    total_blocks = rows * cols

    # ------------------------------------------------------------------
    # 5 & 6. Statistics
    # ------------------------------------------------------------------
    global_mean_var = float(block_variances.mean()) if total_blocks > 0 else 0.0
    global_std_var = float(block_variances.std()) if total_blocks > 0 else 0.0

    # ------------------------------------------------------------------
    # 7. Anomalous blocks (variance > mean + 2*sigma)
    # ------------------------------------------------------------------
    threshold = global_mean_var + 2.0 * global_std_var
    anomalous_mask = block_variances > threshold  # boolean (rows, cols)
    anomalous_blocks = int(anomalous_mask.sum())
    anomalous_ratio = anomalous_blocks / max(1, total_blocks)

    # ------------------------------------------------------------------
    # 8. Percentiles
    # ------------------------------------------------------------------
    flat_vars = block_variances.ravel()
    if flat_vars.size > 0:
        p50, p75, p90, p95 = np.percentile(flat_vars, [50, 75, 90, 95])
    else:
        p50 = p75 = p90 = p95 = 0.0
    percentiles: Dict[str, float] = {
        "p50": float(p50),
        "p75": float(p75),
        "p90": float(p90),
        "p95": float(p95),
    }

    # ------------------------------------------------------------------
    # 9. ROI extraction from anomalous blocks
    # ------------------------------------------------------------------
    components = _connected_components_block(anomalous_mask)

    roi_candidates: List[ROI] = []
    for comp in components:
        if len(comp) < min_anomalous_group_blocks:
            continue

        # Bounding box in block coordinates
        block_rows = [rc[0] for rc in comp]
        block_cols = [rc[1] for rc in comp]
        br_min, br_max = min(block_rows), max(block_rows)
        bc_min, bc_max = min(block_cols), max(block_cols)

        # Convert to pixel coordinates
        x1 = bc_min * block_size
        y1 = br_min * block_size
        x2 = (bc_max + 1) * block_size
        y2 = (br_max + 1) * block_size

        # Clamp to image dimensions
        x2 = min(x2, w)
        y2 = min(y2, h)

        # Score: mean variance of blocks in this group
        group_vars = [block_variances[r, c] for r, c in comp]
        score = float(np.mean(group_vars))

        bw = x2 - x1
        bh = y2 - y1
        area_pct = (bw * bh) / max(h * w, 1) * 100.0
        roi_candidates.append(
            ROI(
                bbox=(x1, y1, bw, bh),
                score=score,
                area_pct=area_pct,
                source="noise",
                notes=f"blocks={len(comp)}",
            )
        )

    # Sort by score descending, take top max_rois
    roi_candidates.sort(key=lambda roi: roi.score, reverse=True)
    top_rois = roi_candidates[:max_rois]

    # ------------------------------------------------------------------
    # 10. Normalize block variances to uint8, resize to image dimensions
    # ------------------------------------------------------------------
    noise_gray_blocks = normalize_to_uint8(block_variances)
    noise_map = cv2.resize(
        noise_gray_blocks,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )

    # ------------------------------------------------------------------
    # 11. Apply colormap
    # ------------------------------------------------------------------
    noise_color = apply_colormap(noise_map)

    # ------------------------------------------------------------------
    # 12. Save images if output_dir is given
    # ------------------------------------------------------------------
    saved_images: Dict[str, str] = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Grayscale noise map
        gray_name = f"{prefix}_noise_gray.png" if prefix else "noise_gray.png"
        gray_path = save_image(noise_map, output_dir, gray_name)
        saved_images["noise_gray"] = str(gray_path)

        # Color-mapped noise heatmap
        color_name = f"{prefix}_noise_color.png" if prefix else "noise_color.png"
        color_path = save_image(noise_color, output_dir, color_name)
        saved_images["noise_color"] = str(color_path)

        # ROI overlay on original grayscale
        rois_name = f"{prefix}_noise_rois.png" if prefix else "noise_rois.png"
        # Ensure we have a proper uint8 image for the overlay
        if cropped_gray.dtype != np.uint8:
            overlay_base = normalize_to_uint8(cropped_gray)
        else:
            overlay_base = cropped_gray
        rois_vis = _draw_rois(overlay_base, top_rois)
        rois_path = save_image(rois_vis, output_dir, rois_name)
        saved_images["noise_rois"] = str(rois_path)

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    return NoiseResult(
        noise_map=noise_map,
        noise_color=noise_color,
        block_variances=block_variances,
        global_mean_var=global_mean_var,
        global_std_var=global_std_var,
        anomalous_blocks=anomalous_blocks,
        total_blocks=total_blocks,
        block_size_used=block_size,
        anomalous_ratio=anomalous_ratio,
        percentiles=percentiles,
        top_rois=top_rois,
        saved_images=saved_images,
    )
