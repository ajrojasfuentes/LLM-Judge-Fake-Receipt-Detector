"""
Block-wise DCT + global FFT frequency analysis for receipt forgery detection.

Extracted and improved from the monolithic ``forensic_analysis.py``.

Algorithm
---------
**DCT (per-block)**
    1. Convert the cropped grayscale receipt to float32.
    2. Divide into ``block_size x block_size`` non-overlapping blocks.
    3. For each block, apply ``cv2.dct`` and compute the ratio of
       high-frequency energy (coefficients where ``i + j > block_size // 4``)
       to total energy.
    4. Blocks whose HF ratio deviates by more than 2 sigma from the
       global mean are flagged as anomalous.

**FFT (global)**
    1. Pad to optimal DFT size, compute 2-D DFT via ``cv2.dft``.
    2. Shift zero-frequency to center, compute log-magnitude spectrum.
    3. Normalise to uint8.

Saved images (when *output_dir* is provided):
    ``{prefix}_dct_gray.png``   -- grayscale DCT energy map
    ``{prefix}_dct_color.png``  -- colour-mapped DCT heatmap
    ``{prefix}_fft_gray.png``   -- grayscale FFT magnitude spectrum
    ``{prefix}_fft_color.png``  -- colour-mapped FFT spectrum
    ``{prefix}_freq_rois.png``  -- top anomalous ROIs drawn on DCT map

Dependencies: cv2, numpy, forensic.utils
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .utils import ROI, apply_colormap, normalize_to_uint8, save_image


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class FrequencyResult:
    """Frequency analysis result (DCT + FFT)."""

    # DCT maps
    dct_map: np.ndarray                # Grayscale DCT energy map resized (uint8)
    dct_color: np.ndarray              # Color-mapped DCT heatmap (BGR)

    # FFT maps
    fft_magnitude: np.ndarray          # Grayscale FFT magnitude spectrum (uint8)
    fft_color: np.ndarray              # Color-mapped FFT spectrum (BGR)

    # Block-level data
    high_freq_ratio_map: np.ndarray    # Per-block HF energy ratio (float64)

    # Statistics
    global_hf_mean: float
    global_hf_std: float
    anomalous_blocks: int
    total_blocks: int
    block_size_used: int
    anomalous_ratio: float             # anomalous_blocks / total_blocks

    # Enhanced
    percentiles: Dict[str, float]      # p50, p75, p90, p95 of HF ratios

    # ROIs from anomalous frequency blocks
    top_rois: List[ROI]

    # Saved image paths
    saved_images: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_hf_mask(block_size: int) -> np.ndarray:
    """Precompute the boolean high-frequency mask for a given block size.

    A coefficient at position ``(i, j)`` is considered high-frequency when
    ``i + j > block_size // 4``.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(block_size, block_size)``.
    """
    hf_mask = np.zeros((block_size, block_size), dtype=bool)
    hf_cutoff = block_size // 4
    for i in range(block_size):
        for j in range(block_size):
            if i + j > hf_cutoff:
                hf_mask[i, j] = True
    return hf_mask


def _extract_rois_from_block_mask(
    anomaly_mask: np.ndarray,
    block_size: int,
    image_h: int,
    image_w: int,
    max_rois: int,
    min_group_blocks: int,
) -> List[ROI]:
    """Extract ROIs from a binary anomalous-block map using connected components.

    Parameters
    ----------
    anomaly_mask : np.ndarray
        2-D boolean / uint8 array where ``True`` (or 255) marks an anomalous
        block.  Shape is ``(block_rows, block_cols)``.
    block_size : int
        Pixel size of each block (used to convert block coords to pixel coords).
    image_h, image_w : int
        Original image dimensions (for clamping).
    max_rois : int
        Maximum number of ROIs to return.
    min_group_blocks : int
        Minimum number of connected anomalous blocks to form a valid ROI.

    Returns
    -------
    List[ROI]
        Up to *max_rois* ROIs sorted by area (largest first).
    """
    # Ensure uint8 for connectedComponents
    mask_u8 = (anomaly_mask.astype(np.uint8)) * 255

    num_labels, labels = cv2.connectedComponents(mask_u8, connectivity=8)

    rois: List[ROI] = []
    for label_id in range(1, num_labels):
        ys, xs = np.where(labels == label_id)
        if len(ys) < min_group_blocks:
            continue

        # Convert block coordinates to pixel coordinates
        y_min = int(ys.min()) * block_size
        x_min = int(xs.min()) * block_size
        y_max = int(ys.max() + 1) * block_size
        x_max = int(xs.max() + 1) * block_size

        # Clamp to image bounds
        y_min = max(0, y_min)
        x_min = max(0, x_min)
        y_max = min(image_h, y_max)
        x_max = min(image_w, x_max)

        area = (y_max - y_min) * (x_max - x_min)
        if area <= 0:
            continue

        bw = x_max - x_min
        bh = y_max - y_min
        area_pct = (bw * bh) / max(image_h * image_w, 1) * 100.0
        rois.append(ROI(
            bbox=(x_min, y_min, bw, bh),
            score=float(len(ys)),
            area_pct=area_pct,
            source="freq",
            notes=f"freq_anomaly_{label_id} blocks={len(ys)}",
        ))

    # Sort by area descending, take top N (use bbox tuple: (x, y, w, h))
    rois.sort(key=lambda r: r.bbox[2] * r.bbox[3], reverse=True)
    return rois[:max_rois]


def _draw_rois(
    base_img: np.ndarray,
    rois: List[ROI],
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw ROI rectangles and labels onto a BGR image.

    Parameters
    ----------
    base_img : np.ndarray
        BGR image to draw on (will be copied).
    rois : List[ROI]
        ROIs to draw.
    color : tuple
        BGR colour for the rectangle.
    thickness : int
        Line thickness.

    Returns
    -------
    np.ndarray
        Annotated BGR image.
    """
    vis = base_img.copy()
    for roi in rois:
        x, y, bw, bh = roi.bbox
        pt1 = (x, y)
        pt2 = (x + bw, y + bh)
        cv2.rectangle(vis, pt1, pt2, color, thickness)
        # Label above the rectangle
        label_text = roi.notes.split()[0] if roi.notes else f"freq_roi"
        label_y = max(y - 6, 12)
        cv2.putText(
            vis,
            label_text,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def frequency_analyze(
    cropped_gray: np.ndarray,
    *,
    output_dir: Optional[Path] = None,
    prefix: str = "",
    block_size: int = 32,
    max_rois: int = 5,
    min_anomalous_group_blocks: int = 2,
) -> FrequencyResult:
    """Run block-wise DCT and global FFT frequency analysis.

    Parameters
    ----------
    cropped_gray : np.ndarray
        Grayscale receipt image (uint8, single-channel).
    output_dir : Path or None
        If provided, diagnostic images are saved here.
    prefix : str
        Filename prefix for saved images.
    block_size : int
        Side length (px) of non-overlapping blocks for DCT analysis.
    max_rois : int
        Maximum number of anomalous ROIs to return.
    min_anomalous_group_blocks : int
        Minimum connected anomalous blocks to form a valid ROI.

    Returns
    -------
    FrequencyResult
        Comprehensive frequency analysis output.
    """
    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------
    gray = cropped_gray.astype(np.float32)
    h, w = gray.shape
    rows = h // block_size
    cols = w // block_size
    total_blocks = rows * cols

    # ------------------------------------------------------------------
    # DCT per block
    # ------------------------------------------------------------------
    hf_ratio_map = np.zeros((rows, cols), dtype=np.float64)
    dct_energy_map = np.zeros((rows, cols), dtype=np.float64)

    # Precompute HF mask once
    hf_mask = _build_hf_mask(block_size)

    for r in range(rows):
        r_start = r * block_size
        r_end = r_start + block_size
        for c in range(cols):
            c_start = c * block_size
            c_end = c_start + block_size

            block = gray[r_start:r_end, c_start:c_end]
            dct_block = cv2.dct(block)

            total_energy = float(np.sum(dct_block ** 2))
            if total_energy < 1e-8:
                hf_ratio_map[r, c] = 0.0
                dct_energy_map[r, c] = 0.0
                continue

            hf_energy = float(np.sum(dct_block[hf_mask] ** 2))
            hf_ratio_map[r, c] = hf_energy / total_energy
            dct_energy_map[r, c] = total_energy

    # ------------------------------------------------------------------
    # DCT statistics
    # ------------------------------------------------------------------
    if total_blocks == 0:
        hf_mean = 0.0
        hf_std = 0.0
        anomalous_blocks = 0
        anomalous_ratio = 0.0
    else:
        hf_mean = float(hf_ratio_map.mean())
        hf_std = float(hf_ratio_map.std())

        upper_thresh = hf_mean + 2.0 * hf_std
        lower_thresh = max(0.0, hf_mean - 2.0 * hf_std)

        anomaly_mask = (hf_ratio_map > upper_thresh) | (hf_ratio_map < lower_thresh)
        anomalous_blocks = int(anomaly_mask.sum())
        anomalous_ratio = anomalous_blocks / max(1, total_blocks)

    # Percentiles of HF ratios (flattened)
    flat_ratios = hf_ratio_map.ravel()
    if flat_ratios.size > 0:
        percentiles: Dict[str, float] = {
            "p50": float(np.percentile(flat_ratios, 50)),
            "p75": float(np.percentile(flat_ratios, 75)),
            "p90": float(np.percentile(flat_ratios, 90)),
            "p95": float(np.percentile(flat_ratios, 95)),
        }
    else:
        percentiles = {"p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0}

    # ------------------------------------------------------------------
    # ROI extraction (connected components of anomalous block map)
    # ------------------------------------------------------------------
    top_rois = _extract_rois_from_block_mask(
        anomaly_mask=anomaly_mask,
        block_size=block_size,
        image_h=h,
        image_w=w,
        max_rois=max_rois,
        min_group_blocks=min_anomalous_group_blocks,
    )

    # ------------------------------------------------------------------
    # DCT visualisation
    # ------------------------------------------------------------------
    dct_gray = normalize_to_uint8(hf_ratio_map)
    # Resize to the original image dimensions so ROI coordinates align correctly
    dct_resized = cv2.resize(
        dct_gray,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )
    dct_color = apply_colormap(dct_resized)

    # ------------------------------------------------------------------
    # FFT global
    # ------------------------------------------------------------------
    dft_rows = cv2.getOptimalDFTSize(h)
    dft_cols = cv2.getOptimalDFTSize(w)
    padded = np.zeros((dft_rows, dft_cols), dtype=np.float32)
    padded[:h, :w] = gray

    dft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft, axes=(0, 1))

    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude = np.log1p(magnitude)

    fft_gray = normalize_to_uint8(magnitude)
    # Crop back to original size for consistency
    fft_gray = fft_gray[:h, :w]
    fft_color = apply_colormap(fft_gray)

    # ------------------------------------------------------------------
    # Save images
    # ------------------------------------------------------------------
    saved_images: Dict[str, str] = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        name_dct_gray = f"{prefix}_dct_gray.png" if prefix else "dct_gray.png"
        name_dct_color = f"{prefix}_dct_color.png" if prefix else "dct_color.png"
        name_fft_gray = f"{prefix}_fft_gray.png" if prefix else "fft_gray.png"
        name_fft_color = f"{prefix}_fft_color.png" if prefix else "fft_color.png"
        name_freq_rois = f"{prefix}_freq_rois.png" if prefix else "freq_rois.png"

        saved_images["dct_gray"] = str(save_image(dct_resized, output_dir, name_dct_gray))
        saved_images["dct_color"] = str(save_image(dct_color, output_dir, name_dct_color))
        saved_images["fft_gray"] = str(save_image(fft_gray, output_dir, name_fft_gray))
        saved_images["fft_color"] = str(save_image(fft_color, output_dir, name_fft_color))

        # ROI overlay on the colour-mapped DCT
        if top_rois:
            roi_vis = _draw_rois(dct_color, top_rois)
        else:
            roi_vis = dct_color.copy()
        saved_images["freq_rois"] = str(save_image(roi_vis, output_dir, name_freq_rois))

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    return FrequencyResult(
        dct_map=dct_resized,
        dct_color=dct_color,
        fft_magnitude=fft_gray,
        fft_color=fft_color,
        high_freq_ratio_map=hf_ratio_map,
        global_hf_mean=hf_mean,
        global_hf_std=hf_std,
        anomalous_blocks=anomalous_blocks,
        total_blocks=total_blocks,
        block_size_used=block_size,
        anomalous_ratio=anomalous_ratio,
        percentiles=percentiles,
        top_rois=top_rois,
        saved_images=saved_images,
    )
