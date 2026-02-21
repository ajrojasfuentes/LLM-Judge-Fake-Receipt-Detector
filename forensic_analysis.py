"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  FORENSIC ANALYSIS MODULE — Find it again! Receipt Forgery Detection       ║
║                                                                            ║
║  Five forensic extraction pipelines designed to feed a VLM (Qwen2.5-VL)   ║
║  with amplified forgery signals:                                           ║
║                                                                            ║
║    1. Error Level Analysis (ELA)                                           ║
║    2. Local Noise Map                                                      ║
║    3. Frequency Analysis (DCT/FFT per block)                               ║
║    4. Copy-Move Detection (ORB keypoint matching)                          ║
║    5. OCR Text Extraction & Cleaning                                       ║
║                                                                            ║
║  Usage:                                                                    ║
║    from forensic_analysis import ForensicAnalyzer                          ║
║    fa = ForensicAnalyzer(output_dir="path/to/output")                      ║
║    results = fa.full_analysis("receipt.png", ocr_txt_path="receipt.txt")   ║
║                                                                            ║
║  Each method can also be called independently.                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import io
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES — structured results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MultiELAResult:
    """
    Result container for Multi-Quality ELA (replaces single-quality ELA).

    Instead of compressing once and comparing, this compresses at several
    JPEG quality levels and measures the *variance* of the error across
    those levels, pixel by pixel.  A pristine PNG region compresses
    smoothly across qualities; a region with prior JPEG history (e.g. pasted
    from a differently-encoded source) shows higher cross-quality variance.
    """
    variance_map: np.ndarray        # Per-pixel cross-quality variance heatmap (uint8)
    variance_color: np.ndarray      # Color-mapped variance heatmap (BGR)
    mean_variance: float            # Global mean of cross-quality variance
    max_variance: float             # Maximum per-pixel variance found
    suspicious_ratio: float         # Fraction of pixels with variance > threshold
    divergent_blocks: int           # Blocks with variance > 2σ from mean
    total_blocks: int               # Total blocks analysed
    qualities_used: Tuple[int, ...] # JPEG quality levels used, e.g. (70, 85, 95)
    variance_threshold: float       # Threshold used for suspicious_ratio


@dataclass
class NoiseMapResult:
    """Result container for Local Noise Analysis."""
    noise_map: np.ndarray           # Grayscale noise variance map (0-255 normalized)
    noise_color: np.ndarray         # Color-mapped noise heatmap (BGR)
    block_variances: np.ndarray     # Raw variance per block (float)
    global_mean_var: float          # Mean of block variances
    global_std_var: float           # Std of block variances
    anomalous_blocks: int           # Blocks with variance > 2σ from mean
    total_blocks: int               # Total blocks analyzed
    block_size_used: int            # Block size (pixels)


@dataclass
class FrequencyResult:
    """Result container for Frequency Analysis (DCT/FFT)."""
    dct_map: np.ndarray             # Grayscale DCT energy map (0-255 normalized)
    dct_color: np.ndarray           # Color-mapped DCT heatmap (BGR)
    fft_magnitude: np.ndarray       # Grayscale FFT magnitude spectrum (0-255)
    fft_color: np.ndarray           # Color-mapped FFT spectrum (BGR)
    high_freq_ratio_map: np.ndarray # Per-block high-freq energy ratio (float)
    global_hf_mean: float           # Mean high-frequency ratio
    global_hf_std: float            # Std of high-frequency ratio
    anomalous_blocks: int           # Blocks with abnormal freq profile
    block_size_used: int


@dataclass
class CopyMoveResult:
    """Result container for Copy-Move Detection."""
    visualization: np.ndarray       # Image with matched regions highlighted (BGR)
    match_mask: np.ndarray          # Binary mask of matched regions (0/255)
    match_heatmap: np.ndarray       # Color heatmap of match density (BGR)
    num_matches: int                # Total keypoint matches found
    num_clusters: int               # Distinct matched region clusters
    matched_regions: List[Dict]     # List of {center, radius, num_points} per cluster
    confidence: float               # 0-1 confidence that copy-move was detected


@dataclass
class OCRResult:
    """Result container for OCR Text Extraction & Cleaning."""
    raw_text: str                   # Original text from .txt file
    cleaned_text: str               # Cleaned/normalized text
    structured: Dict[str, Any]      # Extracted structured fields
    line_count: int                 # Number of non-empty lines
    char_count: int                 # Total character count (cleaned)
    quality_score: float            # 0-1 estimated OCR quality


@dataclass
class ForensicReport:
    """Complete forensic analysis report for a single image."""
    image_path: str
    multi_ela: Optional[MultiELAResult] = None
    noise: Optional[NoiseMapResult] = None
    frequency: Optional[FrequencyResult] = None
    copy_move: Optional[CopyMoveResult] = None
    ocr: Optional[OCRResult] = None
    errors: Dict[str, str] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYZER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ForensicAnalyzer:
    """
    Forensic analysis toolkit for receipt forgery detection.

    Generates visual evidence maps and structured data designed to be fed
    to a Vision-Language Model (VLM) like Qwen2.5-VL as complementary
    inputs alongside the original receipt image.

    Parameters
    ----------
    output_dir : str or Path
        Directory where generated images/reports will be saved.
    mela_qualities : tuple of int
        JPEG quality levels for Multi-ELA cross-quality variance (default: (70, 85, 95)).
    mela_block_size : int
        Block size in pixels for Multi-ELA block-level analysis (default: 16).
    mela_variance_threshold : float
        Per-pixel variance threshold to flag a pixel as suspicious (default: 5.0).
    noise_block_size : int
        Block size in pixels for noise analysis (default: 32).
    freq_block_size : int
        Block size for DCT/FFT analysis (default: 32).
    orb_features : int
        Max features for ORB detector in copy-move (default: 3000).
    match_threshold : float
        Distance ratio threshold for Lowe's ratio test (default: 0.55, stricter
        than the common 0.70 to reduce false positives from repetitive text).
    min_match_distance : float
        Minimum pixel distance between matched keypoints to count as copy-move.
        Set to 150.0 to filter out character-level texture repetition (default: 150.0).
    cluster_eps : float
        Epsilon for spatial clustering of matched keypoints (default: 30.0).
    min_cluster_size : int
        Minimum points in a cluster to be considered a valid copy-move region
        (default: 8 — higher than common 3 to suppress text-pattern noise).
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "./forensic_output",
        *,
        # Multi-ELA params
        mela_qualities: Tuple[int, ...] = (70, 85, 95),
        mela_block_size: int = 16,
        mela_variance_threshold: float = 5.0,
        # Noise params
        noise_block_size: int = 32,
        # Frequency params
        freq_block_size: int = 32,
        # Copy-move params (calibrated for receipt text images)
        orb_features: int = 3000,
        match_threshold: float = 0.55,
        min_match_distance: float = 150.0,
        cluster_eps: float = 30.0,
        min_cluster_size: int = 8,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Multi-ELA
        self.mela_qualities = mela_qualities
        self.mela_block_size = mela_block_size
        self.mela_variance_threshold = mela_variance_threshold

        # Noise
        self.noise_block_size = noise_block_size

        # Frequency
        self.freq_block_size = freq_block_size

        # Copy-move
        self.orb_features = orb_features
        self.match_threshold = match_threshold
        self.min_match_distance = min_match_distance
        self.cluster_eps = cluster_eps
        self.min_cluster_size = min_cluster_size

    # ── PRIVATE HELPERS ──────────────────────────────────────────────────

    @staticmethod
    def _load_image_rgb(path: Union[str, Path]) -> np.ndarray:
        """Load image and convert to RGB (3-channel), handling RGBA/LA/L."""
        img = Image.open(path)
        if img.mode in ("RGBA", "LA"):
            # Composite onto white background to remove alpha
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "LA":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode == "L":
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)

    @staticmethod
    def _load_grayscale(path: Union[str, Path]) -> np.ndarray:
        """Load image as grayscale uint8."""
        img = Image.open(path)
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "LA":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1])
            img = background
        return np.array(img.convert("L"))

    @staticmethod
    def _apply_colormap(gray: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Apply a colormap to a grayscale image. Returns BGR."""
        if gray.dtype != np.uint8:
            g = gray.astype(np.float64)
            g = ((g - g.min()) / (g.max() - g.min() + 1e-8) * 255).astype(np.uint8)
        else:
            g = gray
        return cv2.applyColorMap(g, colormap)

    @staticmethod
    def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
        """Normalize float array to 0-255 uint8."""
        a = arr.astype(np.float64)
        mn, mx = a.min(), a.max()
        if mx - mn < 1e-8:
            return np.zeros_like(a, dtype=np.uint8)
        return ((a - mn) / (mx - mn) * 255).astype(np.uint8)

    def _save_image(self, img: np.ndarray, name: str, subfolder: str = "") -> Path:
        """Save a numpy image (BGR or grayscale) to output_dir/subfolder/name."""
        out = self.output_dir / subfolder if subfolder else self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        path = out / name
        cv2.imwrite(str(path), img)
        return path

    # ── 1. MULTI-QUALITY ELA (Multi-ELA) ─────────────────────────────────

    def multi_ela_analysis(
        self,
        image_path: Union[str, Path],
        qualities: Tuple[int, ...] = (70, 85, 95),
        block_size: int = 16,
        variance_threshold: float = 5.0,
        save: bool = True,
        prefix: str = "",
    ) -> MultiELAResult:
        """
        Perform Multi-Quality Error Level Analysis on a receipt image.

        Unlike single-quality ELA, this method compresses the image at
        several JPEG quality levels and measures the cross-quality *variance*
        of the recompression error, pixel by pixel.

        How it works:
        1. Load the original image (PNG or JPEG — format-agnostic).
        2. Re-compress in-memory at each quality level in `qualities`.
        3. Compute the per-pixel max-channel absolute difference from the
           original for each quality level → one error map per quality.
        4. Stack all error maps and compute per-pixel variance across them.
        5. High variance → region responds differently to different quality
           levels → "compression memory" (prior JPEG encoding history).
        6. Low variance → pristine region (consistent compression behaviour).

        Why this works on PNG files:
        For an unedited scan (PNG, lossless), every region compresses
        smoothly and consistently across quality levels — variance is
        uniformly low.  A region pasted from a JPEG-derived source, or
        manipulated in software that quantised pixel values, will exhibit
        a *different* error-vs-quality curve, producing detectable
        cross-quality variance.

        Parameters
        ----------
        image_path : path to the receipt image (PNG, JPEG, etc.).
        qualities : JPEG quality levels to compare (default: (70, 85, 95)).
        block_size : block size in pixels for block-level anomaly counting.
        variance_threshold : per-pixel variance value above which a pixel
                             is flagged as "suspicious".
        save : whether to save output images to disk.
        prefix : filename prefix for saved images.

        Returns
        -------
        MultiELAResult with variance heatmap and statistics.
        """
        image_path = Path(image_path)
        if not prefix:
            prefix = image_path.stem

        # ── Load image ──
        rgb = self._load_image_rgb(image_path)
        pil_img = Image.fromarray(rgb)
        h, w = rgb.shape[:2]

        # ── Compress at each quality; compute per-pixel max-channel error ──
        ela_maps: List[np.ndarray] = []
        for q in qualities:
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            recomp = np.array(Image.open(buf).convert("RGB"))
            diff = np.abs(rgb.astype(np.float32) - recomp.astype(np.float32))
            ela_maps.append(np.max(diff, axis=2))   # (H, W)

        # ── Per-pixel variance across quality levels ──
        ela_stack = np.stack(ela_maps, axis=2)              # (H, W, Q)
        variance_map = np.var(ela_stack, axis=2).astype(np.float64)  # (H, W)

        # ── Global statistics ──
        mean_var = float(variance_map.mean())
        max_var = float(variance_map.max())
        suspicious_ratio = float((variance_map > variance_threshold).mean())

        # ── Block-level anomaly analysis ──
        rows = h // block_size
        cols = w // block_size
        block_means = np.zeros((rows, cols), dtype=np.float64)
        for r in range(rows):
            for c in range(cols):
                blk = variance_map[
                    r * block_size:(r + 1) * block_size,
                    c * block_size:(c + 1) * block_size,
                ]
                block_means[r, c] = float(blk.mean())

        bv_mean = float(block_means.mean())
        bv_std = float(block_means.std())
        divergent = int((block_means > bv_mean + 2.0 * bv_std).sum())
        total_blocks = rows * cols

        # ── Visualisation ──
        var_amplified = np.clip(variance_map * 8.0, 0, 255).astype(np.uint8)
        var_color = self._apply_colormap(var_amplified)

        result = MultiELAResult(
            variance_map=var_amplified,
            variance_color=var_color,
            mean_variance=mean_var,
            max_variance=max_var,
            suspicious_ratio=suspicious_ratio,
            divergent_blocks=divergent,
            total_blocks=total_blocks,
            qualities_used=tuple(qualities),
            variance_threshold=variance_threshold,
        )

        if save:
            self._save_image(var_amplified, f"{prefix}_mela_variance.png", "multi_ela")
            self._save_image(var_color, f"{prefix}_mela_color.png", "multi_ela")

        return result

    # ── 2. LOCAL NOISE MAP ───────────────────────────────────────────────

    def noise_map_analysis(
        self,
        image_path: Union[str, Path],
        save: bool = True,
        prefix: str = "",
    ) -> NoiseMapResult:
        """
        Compute a local noise variance map.

        How it works:
        1. Convert image to grayscale.
        2. Apply a denoising filter (Gaussian blur) to get the "clean" version.
        3. Compute the residual (original - denoised) = noise component.
        4. Divide the residual into blocks and compute variance per block.
        5. Blocks with anomalous variance indicate potential tampering.

        Copy-paste from a different source or tool-edited regions typically
        have a different noise profile than the surrounding authentic content.

        Parameters
        ----------
        image_path : path to the PNG image.
        save : whether to save output images to disk.
        prefix : filename prefix for saved images.

        Returns
        -------
        NoiseMapResult with noise heatmap and block-level statistics.
        """
        image_path = Path(image_path)
        if not prefix:
            prefix = image_path.stem

        gray = self._load_grayscale(image_path).astype(np.float32)
        bs = self.noise_block_size

        # Denoise: Gaussian blur with kernel proportional to block size
        # Using a moderate sigma for the denoising baseline
        ksize = max(3, (bs // 4) | 1)  # Ensure odd
        denoised = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # Noise residual
        residual = gray - denoised

        # Block-wise variance
        h, w = gray.shape
        rows = h // bs
        cols = w // bs

        block_vars = np.zeros((rows, cols), dtype=np.float64)
        for r in range(rows):
            for c in range(cols):
                block = residual[r*bs:(r+1)*bs, c*bs:(c+1)*bs]
                block_vars[r, c] = float(np.var(block))

        # Statistics
        mean_var = float(block_vars.mean())
        std_var = float(block_vars.std())

        # Anomalous = variance deviates more than 2σ from mean
        threshold = mean_var + 2.0 * std_var
        anomalous = int((block_vars > threshold).sum())

        # Normalize to image for visualization
        noise_gray = self._normalize_to_uint8(block_vars)
        # Resize back to original (approximate) dimensions for overlay
        noise_resized = cv2.resize(noise_gray, (cols * bs, rows * bs),
                                   interpolation=cv2.INTER_NEAREST)
        noise_color = self._apply_colormap(noise_resized)

        result = NoiseMapResult(
            noise_map=noise_resized,
            noise_color=noise_color,
            block_variances=block_vars,
            global_mean_var=mean_var,
            global_std_var=std_var,
            anomalous_blocks=anomalous,
            total_blocks=rows * cols,
            block_size_used=bs,
        )

        if save:
            self._save_image(noise_resized, f"{prefix}_noise_gray.png", "noise")
            self._save_image(noise_color, f"{prefix}_noise_color.png", "noise")

        return result

    # ── 3. FREQUENCY ANALYSIS (DCT + FFT) ───────────────────────────────

    def frequency_analysis(
        self,
        image_path: Union[str, Path],
        save: bool = True,
        prefix: str = "",
    ) -> FrequencyResult:
        """
        Perform block-wise DCT and global FFT frequency analysis.

        How it works:

        DCT (per-block):
        1. Divide grayscale image into NxN blocks.
        2. Apply 2D DCT to each block.
        3. Compute ratio of high-frequency energy to total energy per block.
        4. Regions with anomalous high-frequency content may indicate editing
           (e.g., sharp artificial edges from text insertion, or smoothed areas
           from clone/stamp tools).

        FFT (global):
        1. Compute 2D FFT of the entire grayscale image.
        2. Shift zero-frequency to center and compute log-magnitude spectrum.
        3. Periodic patterns from copy-paste or regular artifacts appear as
           distinct peaks in the spectrum.

        Parameters
        ----------
        image_path : path to the PNG image.
        save : whether to save output images to disk.
        prefix : filename prefix for saved images.

        Returns
        -------
        FrequencyResult with DCT energy map, FFT spectrum, and statistics.
        """
        image_path = Path(image_path)
        if not prefix:
            prefix = image_path.stem

        gray = self._load_grayscale(image_path).astype(np.float32)
        bs = self.freq_block_size
        h, w = gray.shape
        rows = h // bs
        cols = w // bs

        # ── DCT per block ──
        hf_ratio_map = np.zeros((rows, cols), dtype=np.float64)
        dct_energy_map = np.zeros((rows, cols), dtype=np.float64)

        # Define "high frequency" as the bottom-right triangle of DCT coefficients
        # We use a simple threshold: anything beyond bs//4 from (0,0) is "high freq"
        hf_cutoff = bs // 4

        for r in range(rows):
            for c in range(cols):
                block = gray[r*bs:(r+1)*bs, c*bs:(c+1)*bs]
                dct_block = cv2.dct(block)

                total_energy = float(np.sum(dct_block ** 2))
                if total_energy < 1e-8:
                    hf_ratio_map[r, c] = 0.0
                    dct_energy_map[r, c] = 0.0
                    continue

                # High-frequency: coefficients where (i + j) > hf_cutoff
                mask = np.zeros((bs, bs), dtype=bool)
                for i in range(bs):
                    for j in range(bs):
                        if i + j > hf_cutoff:
                            mask[i, j] = True

                hf_energy = float(np.sum(dct_block[mask] ** 2))
                hf_ratio_map[r, c] = hf_energy / total_energy
                dct_energy_map[r, c] = total_energy

        # DCT statistics
        hf_mean = float(hf_ratio_map.mean())
        hf_std = float(hf_ratio_map.std())
        hf_threshold = hf_mean + 2.0 * hf_std
        anomalous = int(
            ((hf_ratio_map > hf_threshold) | (hf_ratio_map < max(0, hf_mean - 2.0 * hf_std))).sum()
        )

        # Normalize DCT map for visualization
        dct_gray = self._normalize_to_uint8(hf_ratio_map)
        dct_resized = cv2.resize(dct_gray, (cols * bs, rows * bs),
                                 interpolation=cv2.INTER_NEAREST)
        dct_color = self._apply_colormap(dct_resized)

        # ── FFT global ──
        # Pad to optimal DFT size for speed
        dft_rows = cv2.getOptimalDFTSize(h)
        dft_cols = cv2.getOptimalDFTSize(w)
        padded = np.zeros((dft_rows, dft_cols), dtype=np.float32)
        padded[:h, :w] = gray

        # Compute DFT
        dft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft, axes=(0, 1))

        # Magnitude spectrum (log scale)
        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        magnitude = np.log1p(magnitude)

        fft_gray = self._normalize_to_uint8(magnitude)
        # Crop back to original size for consistency
        fft_gray = fft_gray[:h, :w]
        fft_color = self._apply_colormap(fft_gray)

        result = FrequencyResult(
            dct_map=dct_resized,
            dct_color=dct_color,
            fft_magnitude=fft_gray,
            fft_color=fft_color,
            high_freq_ratio_map=hf_ratio_map,
            global_hf_mean=hf_mean,
            global_hf_std=hf_std,
            anomalous_blocks=anomalous,
            block_size_used=bs,
        )

        if save:
            self._save_image(dct_resized, f"{prefix}_dct_gray.png", "frequency")
            self._save_image(dct_color, f"{prefix}_dct_color.png", "frequency")
            self._save_image(fft_gray, f"{prefix}_fft_gray.png", "frequency")
            self._save_image(fft_color, f"{prefix}_fft_color.png", "frequency")

        return result

    # ── 4. COPY-MOVE DETECTION ───────────────────────────────────────────

    def copy_move_detection(
        self,
        image_path: Union[str, Path],
        save: bool = True,
        prefix: str = "",
    ) -> CopyMoveResult:
        """
        Detect copy-move (clone) forgery using ORB keypoint matching.

        How it works:
        1. Detect keypoints and compute descriptors using ORB.
        2. Match descriptors against themselves using BFMatcher.
        3. Filter matches:
           a. Remove self-matches (same keypoint location).
           b. Apply Lowe's ratio test.
           c. Require minimum spatial distance between matched points
              (nearby matches are likely texture repetition, not cloning).
        4. Cluster matched keypoints spatially.
        5. Generate visualization and match density heatmap.

        This is particularly relevant for CPI (copy-paste inside) forgeries,
        which represent ~77.6% of all modifications in this dataset.

        Parameters
        ----------
        image_path : path to the PNG image.
        save : whether to save output images to disk.
        prefix : filename prefix for saved images.

        Returns
        -------
        CopyMoveResult with visualization, mask, and cluster information.
        """
        image_path = Path(image_path)
        if not prefix:
            prefix = image_path.stem

        gray = self._load_grayscale(image_path)
        rgb = self._load_image_rgb(image_path)
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # For OpenCV drawing
        h, w = gray.shape

        # ── Detect keypoints ──
        orb = cv2.ORB_create(nfeatures=self.orb_features)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        match_mask = np.zeros((h, w), dtype=np.uint8)
        matched_regions: List[Dict] = []
        num_matches = 0
        num_clusters = 0
        confidence = 0.0

        if descriptors is not None and len(descriptors) > 1:
            # ── Self-matching ──
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            # k=3: for each descriptor, get top-3 matches (first is itself)
            raw_matches = bf.knnMatch(descriptors, descriptors, k=3)

            good_pairs = []
            for match_group in raw_matches:
                # Skip self-match (distance ~= 0), take next two for ratio test
                candidates = [m for m in match_group if m.distance > 0]
                if len(candidates) < 2:
                    continue

                m1, m2 = candidates[0], candidates[1]

                # Lowe's ratio test
                if m1.distance > self.match_threshold * m2.distance:
                    continue

                # Spatial distance filter
                pt1 = np.array(keypoints[m1.queryIdx].pt)
                pt2 = np.array(keypoints[m1.trainIdx].pt)
                dist = np.linalg.norm(pt1 - pt2)

                if dist < self.min_match_distance:
                    continue

                good_pairs.append((m1.queryIdx, m1.trainIdx, pt1, pt2))

            num_matches = len(good_pairs)

            if num_matches > 0:
                # ── Simple spatial clustering (grid-based) ──
                all_pts = []
                for _, _, pt1, pt2 in good_pairs:
                    all_pts.append(pt1)
                    all_pts.append(pt2)
                all_pts = np.array(all_pts)

                # Grid-based clustering
                clusters = self._cluster_points(all_pts, self.cluster_eps,
                                                self.min_cluster_size)
                num_clusters = len(clusters)

                for cluster in clusters:
                    center = cluster.mean(axis=0).astype(int)
                    radius = int(np.max(np.linalg.norm(cluster - center, axis=1))) + 10
                    matched_regions.append({
                        "center": (int(center[0]), int(center[1])),
                        "radius": radius,
                        "num_points": len(cluster),
                    })

                    # Draw on visualization
                    cv2.circle(vis, (int(center[0]), int(center[1])),
                               radius, (0, 0, 255), 2)

                    # Fill mask
                    for pt in cluster.astype(int):
                        cv2.circle(match_mask, (pt[0], pt[1]), 8, 255, -1)

                # Draw match lines (thin, semi-transparent effect via limited drawing)
                for idx, (_, _, pt1, pt2) in enumerate(good_pairs):
                    if idx > 200:  # Cap lines for readability
                        break
                    p1 = tuple(pt1.astype(int))
                    p2 = tuple(pt2.astype(int))
                    cv2.line(vis, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.circle(vis, p1, 3, (255, 0, 0), -1)
                    cv2.circle(vis, p2, 3, (255, 0, 0), -1)

                # Confidence heuristic — calibrated for text-heavy receipt images.
                # Receipt text produces many incidental ORB matches (same characters,
                # same font patterns).  Thresholds of 200 matches / 5 clusters ensure
                # that only genuine large-scale copy-move regions score high.
                confidence = min(1.0, num_matches / 200.0) * min(1.0, num_clusters / 5.0)

        # Generate heatmap from mask
        match_blurred = cv2.GaussianBlur(match_mask, (31, 31), 0)
        match_heatmap = self._apply_colormap(match_blurred)

        result = CopyMoveResult(
            visualization=vis,
            match_mask=match_mask,
            match_heatmap=match_heatmap,
            num_matches=num_matches,
            num_clusters=num_clusters,
            matched_regions=matched_regions,
            confidence=confidence,
        )

        if save:
            self._save_image(vis, f"{prefix}_copymove_vis.png", "copymove")
            self._save_image(match_mask, f"{prefix}_copymove_mask.png", "copymove")
            self._save_image(match_heatmap, f"{prefix}_copymove_heat.png", "copymove")

        return result

    @staticmethod
    def _cluster_points(
        points: np.ndarray,
        eps: float,
        min_size: int,
    ) -> List[np.ndarray]:
        """
        Simple agglomerative clustering without sklearn dependency.

        Uses a union-find approach: two points are in the same cluster
        if their Euclidean distance is <= eps.
        """
        n = len(points)
        if n == 0:
            return []

        # Union-Find
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Build clusters — O(n²) but n is bounded by orb_features
        # Optimization: use spatial hashing for large n
        if n > 5000:
            # Grid-based spatial hash for efficiency
            grid: Dict[Tuple[int, int], List[int]] = {}
            cell_size = eps
            for i, pt in enumerate(points):
                gx, gy = int(pt[0] / cell_size), int(pt[1] / cell_size)
                grid.setdefault((gx, gy), []).append(i)

            for (gx, gy), indices in grid.items():
                # Check neighboring cells
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        neighbor = grid.get((gx + dx, gy + dy), [])
                        for i in indices:
                            for j in neighbor:
                                if i < j:
                                    d = np.linalg.norm(points[i] - points[j])
                                    if d <= eps:
                                        union(i, j)
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(points[i] - points[j])
                    if d <= eps:
                        union(i, j)

        # Group by root
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        # Filter by min_size
        clusters = []
        for indices in groups.values():
            if len(indices) >= min_size:
                clusters.append(points[indices])

        return clusters

    # ── 5. OCR TEXT EXTRACTION & CLEANING ────────────────────────────────

    def ocr_extraction(
        self,
        txt_path: Union[str, Path],
    ) -> OCRResult:
        """
        Load, clean, and structure OCR text from the dataset's .txt files.

        The .txt files in this dataset come from SROIE-style annotations and
        tend to be noisy: inconsistent formatting, OCR errors, mixed delimiters,
        and sometimes garbled characters.

        This method applies multi-stage cleaning:
        1. Encoding normalization (handle common codec issues).
        2. Line-level deduplication and empty-line removal.
        3. Character-level cleaning (fix common OCR substitutions).
        4. Structural extraction (attempt to find company, date, totals, items).
        5. Quality scoring based on parseable content ratio.

        The structured output is designed to be included in VLM prompts for
        semantic cross-verification (e.g., "do the prices add up?").

        Parameters
        ----------
        txt_path : path to the OCR .txt file.

        Returns
        -------
        OCRResult with raw text, cleaned text, structured fields, and quality score.
        """
        txt_path = Path(txt_path)

        # ── Read raw text (try multiple encodings) ──
        raw_text = ""
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                raw_text = txt_path.read_text(encoding=encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if not raw_text.strip():
            return OCRResult(
                raw_text="", cleaned_text="", structured={},
                line_count=0, char_count=0, quality_score=0.0,
            )

        # ── Stage 1: Basic normalization ──
        text = raw_text
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Remove null bytes and control characters (except newline, tab)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # ── Stage 2: Line-level processing ──
        lines = text.split("\n")
        cleaned_lines = []
        seen = set()
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Remove lines that are just separators
            if re.match(r"^[-=_*#.]{3,}$", stripped):
                continue
            # Deduplicate exact lines
            if stripped in seen:
                continue
            seen.add(stripped)
            cleaned_lines.append(stripped)

        # ── Stage 3: Character-level OCR corrections ──
        corrected_lines = []
        for line in cleaned_lines:
            corrected = self._fix_ocr_common(line)
            corrected_lines.append(corrected)

        cleaned_text = "\n".join(corrected_lines)

        # ── Stage 4: Structural extraction ──
        structured = self._extract_receipt_structure(corrected_lines)

        # ── Stage 5: Quality scoring ──
        quality_score = self._ocr_quality_score(corrected_lines, structured)

        return OCRResult(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            structured=structured,
            line_count=len(corrected_lines),
            char_count=len(cleaned_text),
            quality_score=quality_score,
        )

    @staticmethod
    def _fix_ocr_common(text: str) -> str:
        """
        Fix common OCR misrecognition patterns in receipt text.

        These patterns are derived from typical receipt OCR errors
        (thermal paper, low resolution, font artifacts).
        """
        result = text

        # Common character substitutions in receipt OCR
        # O/0 confusion in numeric contexts
        result = re.sub(r"(?<=\d)O(?=\d)", "0", result)
        result = re.sub(r"(?<=\d)o(?=\d)", "0", result)

        # l/1/I confusion in numeric contexts
        result = re.sub(r"(?<=\d)l(?=\d)", "1", result)
        result = re.sub(r"(?<=\d)I(?=\d)", "1", result)

        # S/5 confusion in numeric contexts
        result = re.sub(r"(?<=\d)S(?=\d)", "5", result)

        # B/8 confusion in numeric contexts
        result = re.sub(r"(?<=\d)B(?=\d)", "8", result)

        # Fix decimal separators: "1,50" or "1.50" are both valid
        # but "1 .50" or "1, 50" with spaces are OCR artifacts
        result = re.sub(r"(\d)\s*([.,])\s*(\d{2})\b", r"\1.\3", result)

        # Remove stray single characters that are likely noise
        # (but keep single-letter words like "a", "I", common abbreviations,
        #  and preserve decimal separators between digits)
        # Only remove standalone single chars surrounded by spaces (not between digits)
        result = re.sub(r"(?<!\d)\b[^aAiI0-9.,:;/\-]\b(?!\d)", " ", result)

        # Normalize multiple spaces
        result = re.sub(r"  +", " ", result).strip()

        return result

    @staticmethod
    def _extract_receipt_structure(lines: List[str]) -> Dict[str, Any]:
        """
        Attempt to extract structured information from OCR lines.

        This is best-effort extraction designed to provide context to the VLM,
        not to be a perfect parser. The VLM can use these hints to cross-verify
        what it sees in the image.

        Extracts:
        - company_lines: first few lines (typically store name/address)
        - date_candidates: lines matching date-like patterns
        - total_candidates: lines containing total/amount patterns
        - item_candidates: lines with price-like patterns (product lines)
        - all numeric values found (for arithmetic verification)
        """
        structured: Dict[str, Any] = {
            "company_lines": [],
            "date_candidates": [],
            "total_candidates": [],
            "item_candidates": [],
            "all_amounts": [],
        }

        # Date patterns (various formats common in receipts)
        date_patterns = [
            r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}",     # DD/MM/YYYY, MM-DD-YY, etc
            r"\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}",        # YYYY-MM-DD
            r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{2,4}",
        ]
        date_re = re.compile("|".join(date_patterns), re.IGNORECASE)

        # Total/payment patterns
        total_keywords = re.compile(
            r"\b(total|subtotal|sub\s*total|amount|paid|cash|change|balance|"
            r"tax|gst|vat|nett|net|grand\s*total|round|rounding|tender|"
            r"visa|master|credit|debit)\b",
            re.IGNORECASE,
        )

        # Price/amount pattern: digits with decimal point
        price_re = re.compile(r"\d+\.\d{2}")

        # Company lines: typically the first 3-5 lines before any items
        company_zone = True

        for i, line in enumerate(lines):
            # Extract all monetary amounts
            amounts = price_re.findall(line)
            for a in amounts:
                try:
                    structured["all_amounts"].append(float(a))
                except ValueError:
                    pass

            # Date detection
            if date_re.search(line):
                structured["date_candidates"].append(line)

            # Total detection
            if total_keywords.search(line):
                structured["total_candidates"].append(line)
                company_zone = False
                continue

            # Item lines: contain a price but not a total keyword
            if amounts and not total_keywords.search(line):
                structured["item_candidates"].append(line)
                company_zone = False
                continue

            # Company zone (top of receipt)
            if company_zone and i < 6:
                structured["company_lines"].append(line)

        return structured

    @staticmethod
    def _ocr_quality_score(lines: List[str], structured: Dict) -> float:
        """
        Estimate OCR quality on a 0-1 scale.

        Heuristic factors:
        - Ratio of alphanumeric characters to total characters
        - Whether structural elements were found (dates, totals, items)
        - Average line length (very short or very long = suspicious)
        - Ratio of recognizable words
        """
        if not lines:
            return 0.0

        all_text = " ".join(lines)
        total_chars = len(all_text)
        if total_chars == 0:
            return 0.0

        # Factor 1: Alphanumeric ratio (higher = cleaner)
        alnum = sum(1 for c in all_text if c.isalnum() or c.isspace())
        alnum_ratio = alnum / total_chars

        # Factor 2: Structural elements found
        struct_score = 0.0
        if structured.get("date_candidates"):
            struct_score += 0.25
        if structured.get("total_candidates"):
            struct_score += 0.35
        if structured.get("item_candidates"):
            struct_score += 0.25
        if structured.get("company_lines"):
            struct_score += 0.15

        # Factor 3: Average line length (ideal: 10-60 chars)
        avg_len = total_chars / len(lines)
        len_score = 1.0
        if avg_len < 5:
            len_score = 0.3
        elif avg_len > 100:
            len_score = 0.5

        # Combine
        quality = (0.35 * alnum_ratio + 0.40 * struct_score + 0.25 * len_score)
        return round(min(1.0, max(0.0, quality)), 3)

    # ── FULL ANALYSIS PIPELINE ───────────────────────────────────────────

    def full_analysis(
        self,
        image_path: Union[str, Path],
        ocr_txt_path: Optional[Union[str, Path]] = None,
        save: bool = True,
        prefix: str = "",
    ) -> ForensicReport:
        """
        Run the complete forensic analysis pipeline on a single receipt.

        Executes all five analyses and returns a unified ForensicReport.
        Each analysis runs independently — if one fails, the others
        still complete.

        Parameters
        ----------
        image_path : path to the receipt PNG image.
        ocr_txt_path : path to the paired OCR .txt file (optional).
        save : whether to save output images.
        prefix : filename prefix for outputs. Defaults to image stem.

        Returns
        -------
        ForensicReport containing results from all analyses.
        """
        image_path = Path(image_path)
        if not prefix:
            prefix = image_path.stem

        report = ForensicReport(image_path=str(image_path))

        # 1. Multi-ELA
        try:
            report.multi_ela = self.multi_ela_analysis(
                image_path,
                qualities=self.mela_qualities,
                block_size=self.mela_block_size,
                variance_threshold=self.mela_variance_threshold,
                save=save,
                prefix=prefix,
            )
        except Exception as e:
            report.errors["multi_ela"] = f"{type(e).__name__}: {e}"

        # 2. Noise Map
        try:
            report.noise = self.noise_map_analysis(image_path, save=save, prefix=prefix)
        except Exception as e:
            report.errors["noise"] = f"{type(e).__name__}: {e}"

        # 3. Frequency Analysis
        try:
            report.frequency = self.frequency_analysis(image_path, save=save, prefix=prefix)
        except Exception as e:
            report.errors["frequency"] = f"{type(e).__name__}: {e}"

        # 4. Copy-Move Detection
        try:
            report.copy_move = self.copy_move_detection(image_path, save=save, prefix=prefix)
        except Exception as e:
            report.errors["copy_move"] = f"{type(e).__name__}: {e}"

        # 5. OCR
        if ocr_txt_path is not None:
            try:
                report.ocr = self.ocr_extraction(ocr_txt_path)
            except Exception as e:
                report.errors["ocr"] = f"{type(e).__name__}: {e}"

        return report

    # ── BATCH PROCESSING ─────────────────────────────────────────────────

    def batch_analysis(
        self,
        image_dir: Union[str, Path],
        txt_dir: Optional[Union[str, Path]] = None,
        image_list: Optional[List[str]] = None,
        save: bool = True,
        verbose: bool = True,
    ) -> List[ForensicReport]:
        """
        Run forensic analysis on a batch of receipt images.

        Parameters
        ----------
        image_dir : directory containing PNG images.
        txt_dir : directory containing paired .txt OCR files.
                  If None, OCR analysis is skipped.
        image_list : specific list of image filenames to process.
                     If None, processes all .png files in image_dir.
        save : whether to save output images.
        verbose : print progress.

        Returns
        -------
        List of ForensicReport, one per image.
        """
        image_dir = Path(image_dir)
        if txt_dir is not None:
            txt_dir = Path(txt_dir)

        if image_list is None:
            image_list = sorted([f.name for f in image_dir.glob("*.png")])

        reports = []
        total = len(image_list)

        for idx, filename in enumerate(image_list):
            if verbose and (idx % 25 == 0 or idx == total - 1):
                print(f"  [{idx + 1}/{total}] Processing {filename}...")

            img_path = image_dir / filename
            txt_path = None
            if txt_dir is not None:
                candidate = txt_dir / (Path(filename).stem + ".txt")
                if candidate.exists():
                    txt_path = candidate

            report = self.full_analysis(img_path, ocr_txt_path=txt_path,
                                        save=save, prefix=Path(filename).stem)
            reports.append(report)

        if verbose:
            errors = sum(1 for r in reports if r.errors)
            print(f"  Done: {total} images processed, {errors} with partial errors.")

        return reports

    # ── REPORT TO DICT (for DataFrame integration) ───────────────────────

    @staticmethod
    def report_to_dict(report: ForensicReport) -> Dict[str, Any]:
        """
        Flatten a ForensicReport into a dictionary suitable for a DataFrame row.

        Extracts scalar metrics from each analysis. Image arrays are NOT
        included (they should be referenced by file path).
        """
        d: Dict[str, Any] = {"image_path": report.image_path}

        # Multi-ELA metrics
        if report.multi_ela:
            d["mela_mean_variance"] = report.multi_ela.mean_variance
            d["mela_max_variance"] = report.multi_ela.max_variance
            d["mela_suspicious_ratio"] = report.multi_ela.suspicious_ratio
            d["mela_divergent_blocks"] = report.multi_ela.divergent_blocks
            d["mela_total_blocks"] = report.multi_ela.total_blocks
        else:
            d["mela_mean_variance"] = None
            d["mela_max_variance"] = None
            d["mela_suspicious_ratio"] = None
            d["mela_divergent_blocks"] = None
            d["mela_total_blocks"] = None

        # Noise metrics
        if report.noise:
            d["noise_mean_var"] = report.noise.global_mean_var
            d["noise_std_var"] = report.noise.global_std_var
            d["noise_anomalous_blocks"] = report.noise.anomalous_blocks
            d["noise_total_blocks"] = report.noise.total_blocks
            d["noise_anomalous_ratio"] = (
                report.noise.anomalous_blocks / max(1, report.noise.total_blocks)
            )
        else:
            d["noise_mean_var"] = None
            d["noise_std_var"] = None
            d["noise_anomalous_blocks"] = None
            d["noise_total_blocks"] = None
            d["noise_anomalous_ratio"] = None

        # Frequency metrics
        if report.frequency:
            d["freq_hf_mean"] = report.frequency.global_hf_mean
            d["freq_hf_std"] = report.frequency.global_hf_std
            d["freq_anomalous_blocks"] = report.frequency.anomalous_blocks
        else:
            d["freq_hf_mean"] = None
            d["freq_hf_std"] = None
            d["freq_anomalous_blocks"] = None

        # Copy-move metrics
        if report.copy_move:
            d["cm_num_matches"] = report.copy_move.num_matches
            d["cm_num_clusters"] = report.copy_move.num_clusters
            d["cm_confidence"] = report.copy_move.confidence
        else:
            d["cm_num_matches"] = None
            d["cm_num_clusters"] = None
            d["cm_confidence"] = None

        # OCR metrics
        if report.ocr:
            d["ocr_line_count"] = report.ocr.line_count
            d["ocr_char_count"] = report.ocr.char_count
            d["ocr_quality_score"] = report.ocr.quality_score
            d["ocr_has_date"] = bool(report.ocr.structured.get("date_candidates"))
            d["ocr_has_total"] = bool(report.ocr.structured.get("total_candidates"))
            d["ocr_num_items"] = len(report.ocr.structured.get("item_candidates", []))
            d["ocr_num_amounts"] = len(report.ocr.structured.get("all_amounts", []))
        else:
            d["ocr_line_count"] = None
            d["ocr_char_count"] = None
            d["ocr_quality_score"] = None
            d["ocr_has_date"] = None
            d["ocr_has_total"] = None
            d["ocr_num_items"] = None
            d["ocr_num_amounts"] = None

        # Error flags
        d["analysis_errors"] = str(report.errors) if report.errors else None

        return d

    @staticmethod
    def reports_to_dataframe(reports: List[ForensicReport]) -> "pd.DataFrame":
        """Convert a list of ForensicReports to a pandas DataFrame."""
        import pandas as pd
        rows = [ForensicAnalyzer.report_to_dict(r) for r in reports]
        return pd.DataFrame(rows)
