"""
forensics_analysis.skew — Document skew estimation via Hough line
detection.

Skew (rotation) of a document can indicate a scan or photograph rather
than a digitally generated image.  This module estimates the dominant
skew angle by:

1. Detecting edges with Canny.
2. Running Standard Hough Transform to find dominant line segments.
3. Converting each line's theta to a deviation angle from horizontal
   and folding the result into ``[-45, 45]`` degrees for stability.
4. Computing the median angle as the skew estimate and using the
   median absolute deviation (MAD) as an inverse confidence proxy.

A low MAD (tight angle clustering) yields high confidence, meaning the
detected lines are consistent and the skew estimate is reliable.

This module does not produce output files; it returns a single
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


def estimate_skew(rgb: np.ndarray) -> Dict[str, Any]:
    """Estimate the dominant document skew angle.

    Parameters
    ----------
    rgb : np.ndarray
        Input image of shape ``(H, W, 3)``, dtype ``uint8``.

    Returns
    -------
    dict
        * ``"skew_angle_deg"`` — Estimated skew in degrees (positive =
          clockwise).  ``None`` if OpenCV is unavailable.
        * ``"skew_confidence"`` — Confidence score in ``[0, 1]`` based
          on the inverse MAD of detected line angles.  ``None`` if
          unavailable.
        * ``"method"`` — Detection method used (``"hough"`` or
          ``"none"``).
    """
    if cv2 is None:
        return {
            "skew_angle_deg": None,
            "skew_confidence": None,
            "method": "none",
        }

    gray = to_gray_u8(rgb)
    edges = cv2.Canny(gray, 60, 140)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=130)

    if lines is None or len(lines) < 5:
        return {
            "skew_angle_deg": 0.0,
            "skew_confidence": 0.0,
            "method": "hough",
        }

    angles = []
    for i in range(min(len(lines), 200)):
        rho, theta = lines[i][0]
        # Convert theta (polar angle) to deviation from horizontal
        ang = (theta * 180.0 / np.pi) - 90.0
        # Fold into [-45, 45] for stability across near-vertical /
        # near-horizontal lines
        while ang < -45:
            ang += 90
        while ang > 45:
            ang -= 90
        angles.append(ang)

    a = np.array(angles, dtype=np.float32)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)) + 1e-6)

    # Higher confidence when angles are tightly clustered (low MAD)
    conf = float(np.clip(1.0 - (mad / 10.0), 0.0, 1.0))
    return {
        "skew_angle_deg": med,
        "skew_confidence": conf,
        "method": "hough",
    }
