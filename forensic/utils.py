"""
Shared utilities for the forensic analysis toolkit.

Provides:
- Image I/O helpers (load_image_rgb, load_grayscale, apply_colormap, normalize_to_uint8, save_image)
- Receipt pre-crop / deskew pipeline (crop_receipt)
- Edge-weight and non-white mask computation (used by MELA and others)
- ROI data class and helper functions for merging overlapping ROIs
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class CropResult:
    """Result of receipt crop / deskew."""
    cropped_gray: np.ndarray          # Cropped grayscale image
    cropped_rgb: np.ndarray           # Cropped RGB image
    crop_method: str                  # "contour_warp" | "nonwhite_bbox" | "none"
    receipt_area_ratio: float         # area of receipt / area of original
    crop_bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h) in original coords
    debug_image: Optional[np.ndarray] # Debug visualization (BGR) with contour/bbox drawn
    saved_images: Dict[str, str] = field(default_factory=dict)  # name -> path


@dataclass
class ROI:
    """A Region of Interest identified by a forensic tool."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in cropped image coords
    score: float                      # relevance score [0, 1]
    area_pct: float                   # area as % of total image area
    source: str                       # "mela" | "noise" | "freq" | "cpi" | "ocr" | "fallback"
    notes: str = ""


# ── Image I/O helpers ────────────────────────────────────────────────

def load_image_rgb(path: Union[str, Path]) -> np.ndarray:
    """
    Load any image (handling RGBA/LA/L/P/CMYK/etc.) as RGB uint8.

    Alpha channels are composited onto a white background so that
    transparent regions become white rather than black.

    Parameters
    ----------
    path : str or Path
        Path to the image file.

    Returns
    -------
    np.ndarray
        HxWx3 uint8 array in RGB order.
    """
    path = Path(path)
    img = Image.open(path)

    if img.mode in ("RGBA", "LA"):
        # Composite onto white background to remove alpha
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "LA":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode == "P":
        # Palette images may have transparency
        if "transparency" in img.info:
            img = img.convert("RGBA")
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        else:
            img = img.convert("RGB")
    elif img.mode != "RGB":
        # Covers L, CMYK, I, F, etc.
        img = img.convert("RGB")

    return np.array(img, dtype=np.uint8)


def load_grayscale(path: Union[str, Path]) -> np.ndarray:
    """
    Load any image as grayscale uint8.

    Alpha channels are composited onto a white background before
    conversion so that transparent regions become white.

    Parameters
    ----------
    path : str or Path
        Path to the image file.

    Returns
    -------
    np.ndarray
        HxW uint8 array.
    """
    path = Path(path)
    img = Image.open(path)

    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "LA":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode == "P":
        if "transparency" in img.info:
            img = img.convert("RGBA")
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        else:
            img = img.convert("RGB")
    elif img.mode not in ("L", "RGB"):
        img = img.convert("RGB")

    return np.array(img.convert("L"), dtype=np.uint8)


def apply_colormap(
    gray: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Apply a colormap to a grayscale image.

    If the input is not uint8, it is normalized to [0, 255] first.

    Parameters
    ----------
    gray : np.ndarray
        2D array (grayscale).
    colormap : int
        OpenCV colormap constant (default: cv2.COLORMAP_JET).

    Returns
    -------
    np.ndarray
        HxWx3 uint8 array in BGR order.
    """
    if gray.size == 0:
        return np.zeros((*gray.shape, 3), dtype=np.uint8)

    if gray.dtype != np.uint8:
        g = gray.astype(np.float64)
        mn, mx = g.min(), g.max()
        if mx - mn < 1e-8:
            g = np.zeros_like(g, dtype=np.uint8)
        else:
            g = ((g - mn) / (mx - mn) * 255.0).astype(np.uint8)
    else:
        g = gray

    return cv2.applyColorMap(g, colormap)


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Normalize a float array to [0, 255] uint8.

    If the array has zero range (constant), returns an all-zero uint8 array.

    Parameters
    ----------
    arr : np.ndarray
        Input array of any numeric dtype.

    Returns
    -------
    np.ndarray
        Array of same shape, dtype uint8, values in [0, 255].
    """
    if arr.size == 0:
        return arr.astype(np.uint8)

    a = arr.astype(np.float64)
    mn, mx = a.min(), a.max()
    if mx - mn < 1e-8:
        return np.zeros_like(a, dtype=np.uint8)
    return ((a - mn) / (mx - mn) * 255.0).astype(np.uint8)


def save_image(
    img: np.ndarray,
    output_dir: Union[str, Path],
    name: str,
    subfolder: str = "",
) -> Path:
    """
    Save a numpy image (BGR or grayscale) to disk via OpenCV.

    Parameters
    ----------
    img : np.ndarray
        Image array (HxW for grayscale, HxWx3 for BGR).
    output_dir : str or Path
        Base output directory.
    name : str
        Filename (e.g. "crop_debug.png").
    subfolder : str
        Optional subfolder under output_dir.

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    out = Path(output_dir)
    if subfolder:
        out = out / subfolder
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    cv2.imwrite(str(path), img)
    return path


# ── Receipt crop / deskew pipeline ───────────────────────────────────

def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four points as [top-left, top-right, bottom-right, bottom-left].

    Uses the sum and difference of coordinates to identify corners
    unambiguously regardless of input ordering.

    Parameters
    ----------
    pts : np.ndarray
        (4, 2) array of (x, y) coordinates.

    Returns
    -------
    np.ndarray
        (4, 2) array of ordered points.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)       # x + y
    d = np.diff(pts, axis=1)  # y - x

    rect[0] = pts[np.argmin(s)]   # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right has largest sum
    rect[1] = pts[np.argmin(d)]   # top-right has smallest difference
    rect[3] = pts[np.argmax(d)]   # bottom-left has largest difference

    return rect


def _four_point_warp(
    image: np.ndarray,
    pts: np.ndarray,
) -> np.ndarray:
    """
    Apply a perspective warp to extract the region defined by four points.

    Parameters
    ----------
    image : np.ndarray
        Source image (any channel count).
    pts : np.ndarray
        (4, 2) array of corner points in source image coordinates.

    Returns
    -------
    np.ndarray
        Warped (deskewed) image.
    """
    rect = _order_points(pts)
    tl, tr, br, bl = rect

    # Compute output width as maximum of top and bottom edge lengths
    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    max_w = max(int(width_top), int(width_bot), 1)

    # Compute output height as maximum of left and right edge lengths
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_h = max(int(height_left), int(height_right), 1)

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_w, max_h))
    return warped


def crop_receipt(
    image_rgb: np.ndarray,
    output_dir: Optional[Path] = None,
    prefix: str = "",
) -> CropResult:
    """
    Crop and deskew a receipt from a (possibly larger) scan / photograph.

    Algorithm
    ---------
    1. Convert to grayscale.
    2. Gaussian blur (5, 5) to reduce noise.
    3. Otsu binarization.
    4. Find contours, sort by area descending.
    5. Try to find a quadrilateral contour (4-point polygon approximation)
       whose area covers > 20 % of the image.
    6. If found: perspective-warp to straighten the receipt.
       crop_method = "contour_warp".
    7. If not found: fall back to the bounding box of all non-white pixels
       (luminance < 240).
       - If non-white area > 10 % of image: crop to that bbox.
         crop_method = "nonwhite_bbox".
       - Otherwise: no crop.  crop_method = "none".
    8. Save debug image and cropped image when *output_dir* is provided.
    9. Return CropResult.

    Parameters
    ----------
    image_rgb : np.ndarray
        Input image as HxWx3 uint8 RGB array.
    output_dir : Path or None
        If given, debug and cropped images are saved here.
    prefix : str
        Filename prefix for saved images.

    Returns
    -------
    CropResult
    """
    # Handle degenerate / empty images
    if image_rgb is None or image_rgb.size == 0:
        empty = np.zeros((1, 1), dtype=np.uint8)
        empty_rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        return CropResult(
            cropped_gray=empty,
            cropped_rgb=empty_rgb,
            crop_method="none",
            receipt_area_ratio=0.0,
            crop_bbox=None,
            debug_image=None,
        )

    # Ensure 3-channel
    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)

    h, w = image_rgb.shape[:2]
    total_area = h * w

    # Step 1: grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Step 2: Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Otsu binarization
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert so that the receipt (darker than background) is white in the mask.
    # For receipt-on-white-background scans the receipt text is dark, but the
    # receipt paper itself may be brighter than a dark scanner background.
    # We try both polarities and pick the one that yields better contours.
    thresh_inv = cv2.bitwise_not(thresh)

    # Debug image: draw on a BGR copy of the original
    debug_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Step 4: find contours on both polarities, merge, sort by area
    contours_normal, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    contours_inv, _ = cv2.findContours(
        thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    all_contours = list(contours_normal) + list(contours_inv)
    all_contours.sort(key=cv2.contourArea, reverse=True)

    # Step 5: Try to find a quadrilateral covering > 20% of image
    quad_contour = None
    quad_pts = None
    for cnt in all_contours:
        area = cv2.contourArea(cnt)
        if area < 0.20 * total_area:
            # Contours are sorted by area; once below threshold, stop.
            break

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            quad_contour = cnt
            quad_pts = approx.reshape(4, 2).astype(np.float32)
            break

    # Step 6: Perspective warp if quadrilateral found
    if quad_pts is not None:
        warped_rgb = _four_point_warp(image_rgb, quad_pts)
        warped_gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY)
        receipt_area = cv2.contourArea(quad_pts.astype(np.float32))
        ratio = receipt_area / total_area

        # Compute bounding box of quad in original coords
        x_min = int(quad_pts[:, 0].min())
        y_min = int(quad_pts[:, 1].min())
        x_max = int(quad_pts[:, 0].max())
        y_max = int(quad_pts[:, 1].max())
        crop_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        # Draw contour on debug image
        cv2.drawContours(debug_bgr, [quad_contour], -1, (0, 255, 0), 3)
        for pt in quad_pts.astype(int):
            cv2.circle(debug_bgr, tuple(pt), 8, (0, 0, 255), -1)

        result = CropResult(
            cropped_gray=warped_gray,
            cropped_rgb=warped_rgb,
            crop_method="contour_warp",
            receipt_area_ratio=float(ratio),
            crop_bbox=crop_bbox,
            debug_image=debug_bgr,
        )

        if output_dir is not None:
            output_dir = Path(output_dir)
            p1 = save_image(debug_bgr, output_dir, f"{prefix}crop_debug.png", "crop")
            p2 = save_image(
                cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR),
                output_dir,
                f"{prefix}cropped.png",
                "crop",
            )
            result.saved_images["debug"] = str(p1)
            result.saved_images["cropped"] = str(p2)

        return result

    # Step 7: Fallback — non-white bounding box
    nonwhite_mask = (gray < 240).astype(np.uint8) * 255
    nonwhite_pixels = int(np.count_nonzero(nonwhite_mask))
    nonwhite_ratio = nonwhite_pixels / total_area

    if nonwhite_ratio > 0.10:
        # Find bounding box of non-white region
        coords = cv2.findNonZero(nonwhite_mask)
        if coords is not None:
            x, y, bw, bh = cv2.boundingRect(coords)

            # Clamp to image bounds (safety)
            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            if bw > 0 and bh > 0:
                cropped_rgb = image_rgb[y : y + bh, x : x + bw].copy()
                cropped_gray = gray[y : y + bh, x : x + bw].copy()
                ratio = (bw * bh) / total_area
                crop_bbox = (x, y, bw, bh)

                # Draw bbox on debug image
                cv2.rectangle(debug_bgr, (x, y), (x + bw, y + bh), (255, 0, 0), 3)

                result = CropResult(
                    cropped_gray=cropped_gray,
                    cropped_rgb=cropped_rgb,
                    crop_method="nonwhite_bbox",
                    receipt_area_ratio=float(ratio),
                    crop_bbox=crop_bbox,
                    debug_image=debug_bgr,
                )

                if output_dir is not None:
                    output_dir = Path(output_dir)
                    p1 = save_image(
                        debug_bgr, output_dir, f"{prefix}crop_debug.png", "crop",
                    )
                    p2 = save_image(
                        cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR),
                        output_dir,
                        f"{prefix}cropped.png",
                        "crop",
                    )
                    result.saved_images["debug"] = str(p1)
                    result.saved_images["cropped"] = str(p2)

                return result

    # Step 7b: No meaningful crop possible — return original
    result = CropResult(
        cropped_gray=gray,
        cropped_rgb=image_rgb,
        crop_method="none",
        receipt_area_ratio=1.0,
        crop_bbox=None,
        debug_image=debug_bgr,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        p1 = save_image(debug_bgr, output_dir, f"{prefix}crop_debug.png", "crop")
        result.saved_images["debug"] = str(p1)

    return result


# ── Edge-weight and mask helpers ─────────────────────────────────────

def compute_edge_weight_map(
    gray: np.ndarray,
    k: float = 0.5,
) -> np.ndarray:
    """
    Compute an edge-based weight map: w = 1 / (1 + k * grad_mag_norm).

    Near edges (high gradient magnitude) the weight is low; in flat
    regions the weight approaches 1.  This is useful for weighting
    forensic signals so that natural edges do not dominate anomaly maps.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image.
    k : float
        Scaling factor for gradient magnitude (default 0.5).

    Returns
    -------
    np.ndarray
        Float32 weight map with values in [0, 1], same shape as input.
    """
    if gray.size == 0:
        return np.ones_like(gray, dtype=np.float32)

    # Compute Sobel gradients
    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    # Normalize gradient magnitude to [0, 1]
    mag_max = grad_mag.max()
    if mag_max < 1e-8:
        # Completely flat image — uniform weight of 1
        return np.ones_like(gray, dtype=np.float32)

    grad_mag_norm = grad_mag / mag_max

    # Weight map: high near flat areas, low near edges
    weight = 1.0 / (1.0 + k * grad_mag_norm)

    return weight.astype(np.float32)


def compute_nonwhite_mask(
    gray: np.ndarray,
    threshold: int = 240,
) -> np.ndarray:
    """
    Return a binary mask (uint8, values 0 or 255) of non-white pixels.

    A pixel is considered "non-white" if its grayscale value is strictly
    less than *threshold*.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image.
    threshold : int
        Luminance threshold (default 240).

    Returns
    -------
    np.ndarray
        Binary mask, same shape as input, dtype uint8.
    """
    if gray.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[gray < threshold] = 255
    return mask


# ── ROI helpers ──────────────────────────────────────────────────────

def compute_iou(
    bbox1: Tuple[int, int, int, int],
    bbox2: Tuple[int, int, int, int],
) -> float:
    """
    Compute Intersection over Union between two (x, y, w, h) bounding boxes.

    Parameters
    ----------
    bbox1 : (x, y, w, h)
    bbox2 : (x, y, w, h)

    Returns
    -------
    float
        IoU in [0, 1].  Returns 0.0 if either box has zero area.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    area1 = w1 * h1
    area2 = w2 * h2
    if area1 <= 0 or area2 <= 0:
        return 0.0

    # Intersection rectangle
    ix = max(x1, x2)
    iy = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)

    iw = max(0, ix2 - ix)
    ih = max(0, iy2 - iy)
    inter = iw * ih

    if inter == 0:
        return 0.0

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0

    return float(inter / union)


def merge_rois(
    rois: List[ROI],
    iou_threshold: float = 0.3,
) -> List[ROI]:
    """
    Merge overlapping ROIs by IoU, keeping the highest-scoring one from
    each overlapping group.

    Uses a greedy approach: sort ROIs by score descending, then for each
    ROI suppress any remaining ROI that overlaps it above the threshold
    (similar to Non-Maximum Suppression).

    Parameters
    ----------
    rois : list of ROI
        Input ROIs (may contain overlapping boxes).
    iou_threshold : float
        IoU above which two ROIs are considered overlapping (default 0.3).

    Returns
    -------
    list of ROI
        Merged (deduplicated) ROIs, sorted by score descending.
    """
    if not rois:
        return []

    # Sort by score descending
    sorted_rois = sorted(rois, key=lambda r: r.score, reverse=True)
    keep: List[ROI] = []
    suppressed = set()

    for i, roi_i in enumerate(sorted_rois):
        if i in suppressed:
            continue
        keep.append(roi_i)

        # Suppress all lower-scored ROIs that overlap with this one
        for j in range(i + 1, len(sorted_rois)):
            if j in suppressed:
                continue
            if compute_iou(roi_i.bbox, sorted_rois[j].bbox) >= iou_threshold:
                suppressed.add(j)

    return keep


def rank_rois(
    rois: List[ROI],
    max_k: int = 8,
) -> List[ROI]:
    """
    Sort ROIs by score descending and return the top K.

    Parameters
    ----------
    rois : list of ROI
    max_k : int
        Maximum number of ROIs to return (default 8).

    Returns
    -------
    list of ROI
        Up to *max_k* ROIs sorted by score descending.
    """
    if not rois:
        return []
    sorted_rois = sorted(rois, key=lambda r: r.score, reverse=True)
    return sorted_rois[:max_k]


def fallback_bottom_roi(
    height: int,
    width: int,
    fraction: float = 0.25,
) -> ROI:
    """
    Create a fallback ROI covering the bottom *fraction* of the image.

    Receipts typically have totals, tax lines, and payment information
    at the bottom, so this serves as a reasonable default region when
    no tool-specific ROI is available.

    Parameters
    ----------
    height : int
        Image height in pixels.
    width : int
        Image width in pixels.
    fraction : float
        Fraction of image height to cover from the bottom (default 0.25).

    Returns
    -------
    ROI
        A single ROI covering the bottom portion of the image.
    """
    fraction = max(0.0, min(1.0, fraction))
    roi_h = max(1, int(height * fraction))
    y = height - roi_h

    total_area = max(1, height * width)
    roi_area = roi_h * width
    area_pct = (roi_area / total_area) * 100.0

    return ROI(
        bbox=(0, y, width, roi_h),
        score=0.1,
        area_pct=area_pct,
        source="fallback",
        notes=f"Bottom {fraction:.0%} of image (default region for totals/payment)",
    )
