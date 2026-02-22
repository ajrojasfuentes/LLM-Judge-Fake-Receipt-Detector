"""
forensics.entropy_edges — Image complexity and texture metrics.

Provides three complementary descriptors of image content density:

* **Shannon entropy** — Measures the information content of the
  grayscale histogram.  Low entropy suggests uniform or synthetic
  backgrounds; high entropy indicates rich texture or noise.

* **Edge density** — Fraction of pixels detected as edges by the Canny
  operator.  Document images with text typically have edge densities
  between 5 % and 20 %.

* **Text-ink area estimate** — Fraction of pixels classified as "ink"
  via adaptive thresholding plus morphological noise removal.  Useful
  as a quick proxy for how much printed text the image contains.

None of these metrics produces output files; they return a single
dictionary of scalar values consumed by the evidence pack.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .utils import to_gray_u8

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def shannon_entropy(gray_u8: np.ndarray) -> float:
    """Compute the Shannon entropy of a grayscale histogram.

    Parameters
    ----------
    gray_u8 : np.ndarray
        Grayscale image of shape ``(H, W)`` with dtype ``uint8``.

    Returns
    -------
    float
        Entropy in bits (0 for a constant image; up to 8 for maximal
        uniformity across all 256 levels).
    """
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def edge_density(rgb: np.ndarray) -> Dict[str, Any]:
    """Compute the fraction of edge pixels using the Canny detector.

    Parameters
    ----------
    rgb : np.ndarray
        Input image of shape ``(H, W, 3)``, dtype ``uint8``.

    Returns
    -------
    dict
        ``"edge_density"`` (float or ``None``), ``"edge_method"``
        (str), and ``"edge_threshold"`` (list or ``None``).
    """
    gray = to_gray_u8(rgb)
    if cv2 is None:
        return {
            "edge_density": None,
            "edge_method": "none",
            "edge_threshold": None,
        }

    edges = cv2.Canny(gray, 80, 160)
    dens = float((edges > 0).mean())
    return {
        "edge_density": dens,
        "edge_method": "canny",
        "edge_threshold": [80, 160],
    }


def text_area_estimate(rgb: np.ndarray) -> Dict[str, Any]:
    """Estimate the fraction of "text-like ink" pixels.

    Uses Gaussian adaptive thresholding (inverted) followed by a single
    morphological opening pass to remove salt-and-pepper noise.

    Parameters
    ----------
    rgb : np.ndarray
        Input image of shape ``(H, W, 3)``, dtype ``uint8``.

    Returns
    -------
    dict
        ``"text_ink_ratio"`` (float or ``None``), ``"method"`` (str),
        and ``"params"`` (dict) describing the adaptive threshold
        settings.
    """
    gray = to_gray_u8(rgb)
    if cv2 is None:
        return {"text_ink_ratio": None, "method": "none"}

    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 15,
    )
    # Remove tiny noise blobs
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)

    ink_ratio = float((clean > 0).mean())
    return {
        "text_ink_ratio": ink_ratio,
        "method": "adaptive+open",
        "params": {"block": 35, "C": 15},
    }


# ---------------------------------------------------------------------------
# Combined public API
# ---------------------------------------------------------------------------

def entropy_edge_text_metrics(rgb: np.ndarray) -> Dict[str, Any]:
    """Compute entropy, edge density, and text-ink metrics in one call.

    Parameters
    ----------
    rgb : np.ndarray
        Input image of shape ``(H, W, 3)``, dtype ``uint8``.

    Returns
    -------
    dict
        Merged dictionary containing keys from :func:`shannon_entropy`,
        :func:`edge_density`, and :func:`text_area_estimate`.
    """
    gray = to_gray_u8(rgb)
    return {
        "entropy": shannon_entropy(gray),
        **edge_density(rgb),
        **text_area_estimate(rgb),
    }
