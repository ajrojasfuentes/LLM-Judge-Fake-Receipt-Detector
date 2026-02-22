"""
forensics.utils — Shared utilities for the forensic analysis
pipeline.

Provides:

* **ROI dataclass** — A region of interest detected by a forensic module.
* **Image I/O** — ``load_image_rgb``, ``save_gray_png``, ``save_rgb_png``.
* **Array helpers** — ``normalize_01``, ``clamp01``, ``tile_view``,
  ``crop_rgb``, ``robust_percentiles``.
* **Visualisation** — ``overlay_heatmap``, ``draw_bboxes``.
* **Serialisation** — ``json_sanitize``, ``save_json``.
* **JPEG round-trip** — ``pil_jpeg_bytes``, ``pil_load_jpeg_bytes``.

All functions gracefully degrade when OpenCV (``cv2``) is unavailable,
falling back to pure NumPy / PIL operations where possible.
"""

from __future__ import annotations

import io
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

# Type alias for numeric scalars accepted throughout this module.
Number = Union[int, float, np.number]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ROI:
    """A region of interest flagged by one or more forensic modules.

    Attributes
    ----------
    roi_id : str
        Unique identifier within the analysis run (e.g. ``"r1"``).
    bbox : tuple of int
        Bounding box as ``(x, y, width, height)`` in pixel coordinates.
    score : float
        Anomaly score in ``[0, 1]`` — higher means more suspicious.
    signals : list of str
        Names of the forensic signals that contributed to this ROI
        (e.g. ``["mela_hotspot"]``).
    entity_hint : str, optional
        Semantic hint about what the region might contain
        (e.g. ``"total_field"``).
    ocr_snippet : str, optional
        Short OCR text extracted from inside this bounding box.
    crop_path : str, optional
        File-system path to the saved crop image for this ROI.
    """

    roi_id: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    score: float
    signals: List[str]
    entity_hint: Optional[str] = None
    ocr_snippet: Optional[str] = None
    crop_path: Optional[str] = None


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create a directory (and parents) if it does not exist.

    Parameters
    ----------
    path : str or Path
        Target directory path.

    Returns
    -------
    Path
        The resolved ``Path`` object for the created directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image_rgb(image_path: Union[str, Path]) -> np.ndarray:
    """Load an image file as an RGB ``uint8`` NumPy array.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file (PNG, JPEG, BMP, etc.).

    Returns
    -------
    np.ndarray
        Array of shape ``(H, W, 3)`` with dtype ``uint8``.

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist.
    PIL.UnidentifiedImageError
        If the file cannot be decoded as an image.
    """
    img = Image.open(image_path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def to_gray_u8(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB ``uint8`` image to single-channel grayscale.

    Uses ``cv2.cvtColor`` when available; otherwise falls back to the
    ITU-R BT.601 luminance formula:
    ``Y = 0.299*R + 0.587*G + 0.114*B``.

    Parameters
    ----------
    rgb : np.ndarray
        Input array of shape ``(H, W, 3)`` with dtype ``uint8``.

    Returns
    -------
    np.ndarray
        Grayscale array of shape ``(H, W)`` with dtype ``uint8``.
    """
    if cv2 is None:
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        y = (0.299 * r + 0.587 * g + 0.114 * b).round().astype(np.uint8)
        return y
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


# ---------------------------------------------------------------------------
# Array manipulation
# ---------------------------------------------------------------------------

def clamp01(x: np.ndarray) -> np.ndarray:
    """Clip array values to the ``[0.0, 1.0]`` range.

    Parameters
    ----------
    x : np.ndarray
        Input array (any shape).

    Returns
    -------
    np.ndarray
        Clipped array with the same shape and dtype promotion to float.
    """
    return np.clip(x, 0.0, 1.0)


def normalize_01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalise an array to ``[0.0, 1.0]``.

    If the value range (max - min) is smaller than *eps*, returns an
    all-zeros array of the same shape to avoid division instability.

    Parameters
    ----------
    x : np.ndarray
        Input array (any shape).
    eps : float
        Minimum range below which the output is forced to zero.

    Returns
    -------
    np.ndarray
        Normalised ``float32`` array in ``[0, 1]``.
    """
    x = x.astype(np.float32, copy=False)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def robust_percentiles(
    x: np.ndarray,
    ps: Sequence[float] = (50, 75, 90, 95, 97, 99),
) -> Dict[str, float]:
    """Compute percentiles on finite values of a flattened array.

    Non-finite values (``NaN``, ``±Inf``) are silently excluded.

    Parameters
    ----------
    x : np.ndarray
        Input array (any shape).
    ps : sequence of float
        Percentile levels to compute (0-100 scale).

    Returns
    -------
    dict
        Mapping from ``"p<level>"`` to the computed percentile value.
        If no finite values exist, all values are ``NaN``.
    """
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{int(p)}": float("nan") for p in ps}
    vals = np.percentile(x, ps)
    return {f"p{int(p)}": float(v) for p, v in zip(ps, vals)}


def tile_view(
    arr: np.ndarray, tile_h: int, tile_w: int,
) -> Tuple[np.ndarray, int, int]:
    """Reshape a 2-D (or 3-D) array into non-overlapping tiles.

    The array is cropped to the largest dimensions that are exact
    multiples of the tile size.  The returned array is a **view** of
    the original data — it shares memory, so mutations propagate.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape ``(H, W)`` or ``(H, W, C)``.
    tile_h, tile_w : int
        Tile height and width in pixels.

    Returns
    -------
    tiles : np.ndarray
        Reshaped view.  For a 2-D input the shape is
        ``(n_rows, n_cols, tile_h, tile_w)``; for 3-D it is
        ``(n_rows, n_cols, tile_h, tile_w, C)``.
    n_rows : int
        Number of tile rows.
    n_cols : int
        Number of tile columns.
    """
    H, W = arr.shape[:2]
    Hc = (H // tile_h) * tile_h
    Wc = (W // tile_w) * tile_w
    cropped = arr[:Hc, :Wc]
    nh = Hc // tile_h
    nw = Wc // tile_w
    if cropped.ndim == 2:
        tiles = cropped.reshape(nh, tile_h, nw, tile_w).swapaxes(1, 2)
    else:
        C = cropped.shape[2]
        tiles = cropped.reshape(nh, tile_h, nw, tile_w, C).swapaxes(1, 2)
    return tiles, nh, nw


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------

def save_gray_png(gray01: np.ndarray, out_path: Union[str, Path]) -> str:
    """Save a ``[0, 1]`` float array as a grayscale PNG.

    Parameters
    ----------
    gray01 : np.ndarray
        2-D float array with values in ``[0, 1]``.
    out_path : str or Path
        Destination file path.

    Returns
    -------
    str
        The string representation of *out_path*.
    """
    out_path = str(out_path)
    im = (clamp01(gray01) * 255.0).round().astype(np.uint8)
    Image.fromarray(im, mode="L").save(out_path)
    return out_path


def save_rgb_png(rgb: np.ndarray, out_path: Union[str, Path]) -> str:
    """Save an RGB ``uint8`` array as a PNG.

    Parameters
    ----------
    rgb : np.ndarray
        3-D array of shape ``(H, W, 3)`` with dtype ``uint8``.
    out_path : str or Path
        Destination file path.

    Returns
    -------
    str
        The string representation of *out_path*.
    """
    out_path = str(out_path)
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def overlay_heatmap(
    rgb: np.ndarray, heat01: np.ndarray, alpha: float = 0.45,
) -> np.ndarray:
    """Blend a scalar heatmap onto an RGB image using a red colour ramp.

    This avoids a matplotlib dependency by constructing the overlay
    manually.  High heat values shift pixels toward red; low values
    leave the base colour intact.

    Parameters
    ----------
    rgb : np.ndarray
        Base image, shape ``(H, W, 3)``, dtype ``uint8``.
    heat01 : np.ndarray
        Heatmap array, shape ``(H, W)``, values in ``[0, 1]``.
    alpha : float
        Blend factor.  ``0`` = base only, ``1`` = overlay only.

    Returns
    -------
    np.ndarray
        Blended RGB image, dtype ``uint8``.
    """
    heat01 = clamp01(heat01).astype(np.float32)
    base = rgb.astype(np.float32) / 255.0

    overlay = base.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + heat01, 0.0, 1.0)
    overlay[..., 1] = np.clip(overlay[..., 1] * (1.0 - 0.25 * heat01), 0.0, 1.0)
    overlay[..., 2] = np.clip(overlay[..., 2] * (1.0 - 0.25 * heat01), 0.0, 1.0)

    out = base * (1.0 - alpha) + overlay * alpha
    return (np.clip(out, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def crop_rgb(
    rgb: np.ndarray, bbox: Tuple[int, int, int, int], pad: int = 0,
) -> np.ndarray:
    """Crop a rectangular region from an RGB image.

    Parameters
    ----------
    rgb : np.ndarray
        Source image of shape ``(H, W, 3)``.
    bbox : tuple of int
        ``(x, y, w, h)`` bounding box in pixel coordinates.
    pad : int
        Extra padding in pixels on every side (clamped to image bounds).

    Returns
    -------
    np.ndarray
        Copied crop of shape ``(h', w', 3)``.
    """
    x, y, w, h = bbox
    H, W = rgb.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    return rgb[y0:y1, x0:x1].copy()


def draw_bboxes(
    rgb: np.ndarray,
    bboxes: Sequence[Tuple[int, int, int, int]],
    thickness: int = 2,
) -> np.ndarray:
    """Draw green bounding-box rectangles onto an RGB image.

    Uses OpenCV when available; otherwise falls back to a simple
    pixel-level drawing approach using NumPy.

    Parameters
    ----------
    rgb : np.ndarray
        Source image of shape ``(H, W, 3)``, dtype ``uint8``.
    bboxes : sequence of tuple
        Each element is ``(x, y, w, h)`` in pixel coordinates.
    thickness : int
        Line thickness in pixels.

    Returns
    -------
    np.ndarray
        Copy of *rgb* with rectangles drawn.
    """
    if cv2 is None:
        out = rgb.copy()
        H, W = out.shape[:2]
        for (x, y, w, h) in bboxes:
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W - 1, x + w), min(H - 1, y + h)
            for t in range(thickness):
                if y0 + t < H:
                    out[y0 + t, x0:x1, :] = [0, 255, 0]
                if y1 - t >= 0:
                    out[y1 - t, x0:x1, :] = [0, 255, 0]
                if x0 + t < W:
                    out[y0:y1, x0 + t, :] = [0, 255, 0]
                if x1 - t >= 0:
                    out[y0:y1, x1 - t, :] = [0, 255, 0]
        return out

    out = rgb.copy()
    for (x, y, w, h) in bboxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), thickness)
    return out


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def json_sanitize(obj: Any) -> Any:
    """Recursively convert an object tree into JSON-safe Python types.

    Handles: ``numpy`` scalars/arrays, ``Path`` objects, ``dataclass``
    instances, ``NaN``/``Inf`` floats (mapped to ``None``), and nested
    dicts/lists.

    Parameters
    ----------
    obj : Any
        Arbitrary Python object.

    Returns
    -------
    Any
        JSON-serialisable equivalent.
    """
    if obj is None:
        return None
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return {k: json_sanitize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return json_sanitize(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (int, float, str, bool)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    return str(obj)


def save_json(data: Dict[str, Any], out_path: Union[str, Path]) -> str:
    """Serialise a dictionary to a pretty-printed JSON file.

    All values are passed through :func:`json_sanitize` before writing.

    Parameters
    ----------
    data : dict
        The data to serialise.
    out_path : str or Path
        Destination file path.

    Returns
    -------
    str
        The string representation of *out_path*.
    """
    out_path = str(out_path)
    safe = json_sanitize(data)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=False, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# JPEG round-trip helpers
# ---------------------------------------------------------------------------

def pil_jpeg_bytes(rgb: np.ndarray, quality: int) -> bytes:
    """Compress an RGB array to JPEG bytes in memory.

    Parameters
    ----------
    rgb : np.ndarray
        Image array of shape ``(H, W, 3)``, dtype ``uint8``.
    quality : int
        JPEG quality level (1-95).

    Returns
    -------
    bytes
        Raw JPEG data.
    """
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return buf.getvalue()


def pil_load_jpeg_bytes(jpeg_bytes: bytes) -> np.ndarray:
    """Decode in-memory JPEG bytes back to an RGB NumPy array.

    Parameters
    ----------
    jpeg_bytes : bytes
        Raw JPEG data.

    Returns
    -------
    np.ndarray
        Decoded image of shape ``(H, W, 3)``, dtype ``uint8``.
    """
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    return np.array(img, dtype=np.uint8)
