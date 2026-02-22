from __future__ import annotations

import io
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


Number = Union[int, float, np.number]


@dataclass
class ROI:
    roi_id: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    score: float
    signals: List[str]
    entity_hint: Optional[str] = None
    ocr_snippet: Optional[str] = None
    crop_path: Optional[str] = None


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_image_rgb(image_path: Union[str, Path]) -> np.ndarray:
    """Load image as RGB uint8."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def to_gray_u8(rgb: np.ndarray) -> np.ndarray:
    if cv2 is None:
        # fallback luminance
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        y = (0.299 * r + 0.587 * g + 0.114 * b).round().astype(np.uint8)
        return y
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def normalize_01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)


def robust_percentiles(x: np.ndarray, ps: Sequence[float] = (50, 75, 90, 95, 97, 99)) -> Dict[str, float]:
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{int(p)}": float("nan") for p in ps}
    vals = np.percentile(x, ps)
    return {f"p{int(p)}": float(v) for p, v in zip(ps, vals)}


def tile_view(arr: np.ndarray, tile_h: int, tile_w: int) -> Tuple[np.ndarray, int, int]:
    """Return a view of arr cropped to multiples of tile size: (tiles, Ht, Wt)."""
    H, W = arr.shape[:2]
    Hc = (H // tile_h) * tile_h
    Wc = (W // tile_w) * tile_w
    cropped = arr[:Hc, :Wc]
    nh = Hc // tile_h
    nw = Wc // tile_w
    if cropped.ndim == 2:
        tiles = cropped.reshape(nh, tile_h, nw, tile_w).swapaxes(1, 2)  # (nh, nw, th, tw)
    else:
        C = cropped.shape[2]
        tiles = cropped.reshape(nh, tile_h, nw, tile_w, C).swapaxes(1, 2)  # (nh, nw, th, tw, C)
    return tiles, nh, nw


def save_gray_png(gray01: np.ndarray, out_path: Union[str, Path]) -> str:
    """Save float map in [0,1] as grayscale PNG."""
    out_path = str(out_path)
    im = (clamp01(gray01) * 255.0).round().astype(np.uint8)
    Image.fromarray(im, mode="L").save(out_path)
    return out_path


def save_rgb_png(rgb: np.ndarray, out_path: Union[str, Path]) -> str:
    out_path = str(out_path)
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(out_path)
    return out_path


def overlay_heatmap(rgb: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Create RGB overlay using a simple red ramp (no matplotlib dependency)."""
    heat01 = clamp01(heat01).astype(np.float32)
    base = rgb.astype(np.float32) / 255.0

    # Red overlay
    overlay = base.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + heat01, 0.0, 1.0)
    overlay[..., 1] = np.clip(overlay[..., 1] * (1.0 - 0.25 * heat01), 0.0, 1.0)
    overlay[..., 2] = np.clip(overlay[..., 2] * (1.0 - 0.25 * heat01), 0.0, 1.0)

    out = (base * (1.0 - alpha) + overlay * alpha)
    return (np.clip(out, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def crop_rgb(rgb: np.ndarray, bbox: Tuple[int, int, int, int], pad: int = 0) -> np.ndarray:
    x, y, w, h = bbox
    H, W = rgb.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    return rgb[y0:y1, x0:x1].copy()


def draw_bboxes(rgb: np.ndarray, bboxes: Sequence[Tuple[int, int, int, int]], thickness: int = 2) -> np.ndarray:
    """Draw green boxes (opencv if available; else crude PIL draw fallback)."""
    if cv2 is None:
        # crude fallback: draw border pixels directly
        out = rgb.copy()
        H, W = out.shape[:2]
        for (x, y, w, h) in bboxes:
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W - 1, x + w), min(H - 1, y + h)
            for t in range(thickness):
                if y0 + t < H: out[y0 + t, x0:x1, :] = [0, 255, 0]
                if y1 - t >= 0: out[y1 - t, x0:x1, :] = [0, 255, 0]
                if x0 + t < W: out[y0:y1, x0 + t, :] = [0, 255, 0]
                if x1 - t >= 0: out[y0:y1, x1 - t, :] = [0, 255, 0]
        return out

    out = rgb.copy()
    for (x, y, w, h) in bboxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), thickness)
    return out


def json_sanitize(obj: Any) -> Any:
    """Convert numpy types + Path + dataclasses to JSON-safe Python types."""
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
        # normalize NaN/inf
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    return str(obj)


def save_json(data: Dict[str, Any], out_path: Union[str, Path]) -> str:
    out_path = str(out_path)
    safe = json_sanitize(data)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=False, indent=2)
    return out_path


def pil_jpeg_bytes(rgb: np.ndarray, quality: int) -> bytes:
    """Compress RGB to JPEG bytes in-memory."""
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return buf.getvalue()


def pil_load_jpeg_bytes(jpeg_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    return np.array(img, dtype=np.uint8)