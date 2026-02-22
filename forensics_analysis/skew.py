from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .utils import to_gray_u8

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def estimate_skew(rgb: np.ndarray) -> Dict[str, Any]:
    """
    Estimate dominant skew angle using Hough lines on edges.
    Returns degrees (positive=clockwise) and a confidence proxy.
    """
    if cv2 is None:
        return {"skew_angle_deg": None, "skew_confidence": None, "method": "none"}

    gray = to_gray_u8(rgb)
    edges = cv2.Canny(gray, 60, 140)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=130)

    if lines is None or len(lines) < 5:
        return {"skew_angle_deg": 0.0, "skew_confidence": 0.0, "method": "hough"}

    angles = []
    for i in range(min(len(lines), 200)):
        rho, theta = lines[i][0]
        # convert theta to line angle in degrees relative to horizontal
        ang = (theta * 180.0 / np.pi) - 90.0
        # fold into [-45,45] for stability
        while ang < -45:
            ang += 90
        while ang > 45:
            ang -= 90
        angles.append(ang)

    a = np.array(angles, dtype=np.float32)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)) + 1e-6)

    # higher confidence if angles tightly clustered
    conf = float(np.clip(1.0 - (mad / 10.0), 0.0, 1.0))
    return {"skew_angle_deg": med, "skew_confidence": conf, "method": "hough"}