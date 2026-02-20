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
class ELAResult:
    """Result container for Error Level Analysis."""
    ela_image: np.ndarray           # Grayscale ELA heatmap (0-255)
    ela_color: np.ndarray           # Color-mapped ELA (BGR, for visualization)
    mean_error: float               # Global mean error level
    std_error: float                # Global std of error levels
    max_error: float                # Maximum error level found
    suspicious_ratio: float         # Fraction of pixels above threshold
    quality_used: int               # JPEG quality used for recompression
    threshold_used: float           # Threshold for "suspicious" pixels


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
    ela: Optional[ELAResult] = None
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
    ela_quality : int
        JPEG recompression quality for ELA (default: 95).
    ela_threshold : float
        Pixel error threshold to flag as "suspicious" in ELA (default: 25.0).
    ela_scale : float
        Amplification factor for ELA visualization (default: 10.0).
    noise_block_size : int
        Block size in pixels for noise analysis (default: 32).
    freq_block_size : int
        Block size for DCT/FFT analysis (default: 32).
    orb_features : int
        Max features for ORB detector in copy-move (default: 3000).
    match_threshold : float
        Distance ratio threshold for keypoint matching (default: 0.70).
    min_match_distance : float
        Minimum pixel distance between matched keypoints to count as
        copy-move (filters self-matches) (default: 50.0).
    cluster_eps : float
        DBSCAN-like clustering epsilon for grouping matches (default: 40.0).
    min_cluster_size : int
        Minimum points in a cluster to be considered valid (default: 3).
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "./forensic_output",
        *,
        # ELA params
        ela_quality: int = 95,
        ela_threshold: float = 25.0,
        ela_scale: float = 10.0,
        # Noise params
        noise_block_size: int = 32,
        # Frequency params
        freq_block_size: int = 32,
        # Copy-move params
        orb_features: int = 3000,
        match_threshold: float = 0.70,
        min_match_distance: float = 50.0,
        cluster_eps: float = 40.0,
        min_cluster_size: int = 3,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ELA
        self.ela_quality = ela_quality
        self.ela_threshold = ela_threshold
        self.ela_scale = ela_scale

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

    # ── 1. ERROR LEVEL ANALYSIS (ELA) ────────────────────────────────────

    def error_level_analysis(
        self,
        image_path: Union[str, Path],
        save: bool = True,
        prefix: str = "",
    ) -> ELAResult:
        """
        Perform Error Level Analysis on a PNG image.

        How it works:
        1. Load the original PNG image.
        2. Re-compress it as JPEG at a fixed quality level (in-memory).
        3. Compute the absolute difference between original and recompressed.
        4. Amplify and normalize the difference to create a heatmap.

        Regions that were edited (especially pasted from differently-compressed
        sources) will show different error levels compared to the rest of the
        image, appearing as bright spots in the ELA map.

        Parameters
        ----------
        image_path : path to the PNG image.
        save : whether to save output images to disk.
        prefix : filename prefix for saved images.

        Returns
        -------
        ELAResult with the ELA heatmap and statistics.
        """
        image_path = Path(image_path)
        if not prefix:
            prefix = image_path.stem

        # Load as RGB
        rgb = self._load_image_rgb(image_path)

        # Re-compress via JPEG in memory
        pil_img = Image.fromarray(rgb)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=self.ela_quality)
        buffer.seek(0)
        recompressed = np.array(Image.open(buffer))

        # Compute absolute difference per channel, then take max across channels
        diff = np.abs(rgb.astype(np.float32) - recompressed.astype(np.float32))
        # Max across channels gives the strongest signal per pixel
        ela_gray = np.max(diff, axis=2)

        # Amplify for visibility
        ela_amplified = np.clip(ela_gray * self.ela_scale, 0, 255).astype(np.uint8)

        # Statistics
        mean_err = float(ela_gray.mean())
        std_err = float(ela_gray.std())
        max_err = float(ela_gray.max())
        suspicious = float((ela_gray > self.ela_threshold).mean())

        # Color-mapped version
        ela_color = self._apply_colormap(ela_amplified)

        result = ELAResult(
            ela_image=ela_amplified,
            ela_color=ela_color,
            mean_error=mean_err,
            std_error=std_err,
            max_error=max_err,
            suspicious_ratio=suspicious,
            quality_used=self.ela_quality,
            threshold_used=self.ela_threshold,
        )

        if save:
            self._save_image(ela_amplified, f"{prefix}_ela_gray.png", "ela")
            self._save_image(ela_color, f"{prefix}_ela_color.png", "ela")

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
        which represent ~56% of all modifications in this dataset.

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

                # Confidence heuristic
                # Based on number of matches and clusters
                confidence = min(1.0, num_matches / 30.0) * min(1.0, num_clusters / 2.0)

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

        # 1. ELA
        try:
            report.ela = self.error_level_analysis(image_path, save=save, prefix=prefix)
        except Exception as e:
            report.errors["ela"] = f"{type(e).__name__}: {e}"

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

        # ELA metrics
        if report.ela:
            d["ela_mean_error"] = report.ela.mean_error
            d["ela_std_error"] = report.ela.std_error
            d["ela_max_error"] = report.ela.max_error
            d["ela_suspicious_ratio"] = report.ela.suspicious_ratio
        else:
            d["ela_mean_error"] = None
            d["ela_std_error"] = None
            d["ela_max_error"] = None
            d["ela_suspicious_ratio"] = None

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
