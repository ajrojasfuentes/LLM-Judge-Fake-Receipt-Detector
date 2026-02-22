from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .utils import normalize_01, to_gray_u8

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def shannon_entropy(gray_u8: np.ndarray) -> float:
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def edge_density(rgb: np.ndarray) -> Dict[str, Any]:
    gray = to_gray_u8(rgb)
    if cv2 is None:
        return {"edge_density": None, "edge_method": "none", "edge_threshold": None}

    edges = cv2.Canny(gray, 80, 160)
    dens = float((edges > 0).mean())
    return {"edge_density": dens, "edge_method": "canny", "edge_threshold": [80, 160]}


def text_area_estimate(rgb: np.ndarray) -> Dict[str, Any]:
    """
    Very lightweight "text-like ink" estimate:
    adaptive threshold + morphology; returns % of pixels considered ink.
    """
    gray = to_gray_u8(rgb)
    if cv2 is None:
        return {"text_ink_ratio": None, "method": "none"}

    # adaptive threshold (invert: ink=1)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15
    )
    # remove tiny noise
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)

    ink_ratio = float((clean > 0).mean())
    return {"text_ink_ratio": ink_ratio, "method": "adaptive+open", "params": {"block": 35, "C": 15}}


def entropy_edge_text_metrics(rgb: np.ndarray) -> Dict[str, Any]:
    gray = to_gray_u8(rgb)
    return {
        "entropy": shannon_entropy(gray),
        **edge_density(rgb),
        **text_area_estimate(rgb),
    }