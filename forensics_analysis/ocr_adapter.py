"""
forensics_analysis.ocr_adapter — Unified OCR interface with engine
fallback.

Provides a single entry-point, :func:`run_ocr`, that abstracts over two
OCR strategies:

1. **Plain-text file** — If a pre-existing ``.txt`` transcription is
   available (common in datasets like FINDIT2), it is read directly.
2. **PaddleOCR engine** — If no text file is supplied and the
   ``paddleocr`` package is installed, the image is passed through
   PaddleOCR for automatic text recognition with angle classification.

PaddleOCR is lazily initialised on first use and cached as a module-
level singleton so that repeated calls within the same process do not
pay the model-loading cost more than once.

The output always includes an ``"engine"`` tag (``"txt"``,
``"paddleocr"``, or ``"none"``) so downstream consumers know which
source produced the text lines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .utils import to_gray_u8

# ---------------------------------------------------------------------------
# Optional PaddleOCR dependency
# ---------------------------------------------------------------------------

try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:
    PaddleOCR = None

# Module-level singleton for the PaddleOCR engine.
_OCR_INSTANCE = None

# Maximum number of text lines returned from any engine to prevent
# excessively large payloads in the evidence pack.
_MAX_LINES = 300


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_ocr() -> Optional[Any]:
    """Lazily initialise and return the PaddleOCR singleton.

    Returns ``None`` if PaddleOCR is not installed.
    """
    global _OCR_INSTANCE
    if PaddleOCR is None:
        return None
    if _OCR_INSTANCE is None:
        _OCR_INSTANCE = PaddleOCR(use_angle_cls=True, lang="en")
    return _OCR_INSTANCE


# ---------------------------------------------------------------------------
# Text-file reader
# ---------------------------------------------------------------------------

def read_txt_ocr(
    txt_path: Union[str, Path], max_lines: int = _MAX_LINES,
) -> List[str]:
    """Read a plain-text OCR transcription file.

    Blank lines are skipped and leading/trailing whitespace is stripped.

    Parameters
    ----------
    txt_path : str or Path
        Path to the ``.txt`` file.
    max_lines : int
        Maximum number of non-empty lines to return.

    Returns
    -------
    list of str
        Non-empty, stripped text lines (up to *max_lines*).
    """
    p = Path(txt_path)
    if not p.exists():
        return []
    lines: List[str] = []
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if ln:
            lines.append(ln)
        if len(lines) >= max_lines:
            break
    return lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ocr(
    rgb: np.ndarray,
    txt_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run OCR on an image and return structured results.

    Strategy selection:

    * If *txt_path* is provided, reads the file directly (fastest).
    * Otherwise, if PaddleOCR is installed, runs the OCR engine.
    * If neither is available, returns an empty result with
      ``engine="none"``.

    Parameters
    ----------
    rgb : np.ndarray
        Input image of shape ``(H, W, 3)``, dtype ``uint8``.
    txt_path : str or Path, optional
        Path to a pre-existing ``.txt`` transcription.  When given,
        PaddleOCR is not invoked.

    Returns
    -------
    dict
        * ``"engine"`` — ``"txt"``, ``"paddleocr"``, or ``"none"``.
        * ``"lines"`` — List of recognised text strings (capped at
          ``_MAX_LINES``).
        * ``"boxes"`` — List of box dicts (PaddleOCR only) or ``None``.
          Each dict contains ``"box"``, ``"text"``, and ``"conf"``.
    """
    # ------------------------------------------------------------------
    # Strategy 1: pre-existing text file
    # ------------------------------------------------------------------
    if txt_path is not None:
        lines = read_txt_ocr(txt_path, max_lines=_MAX_LINES)
        return {"engine": "txt", "lines": lines, "boxes": None}

    # ------------------------------------------------------------------
    # Strategy 2: PaddleOCR engine
    # ------------------------------------------------------------------
    ocr = _get_ocr()
    if ocr is None:
        return {"engine": "none", "lines": [], "boxes": None}

    # PaddleOCR expects a BGR NumPy array
    try:
        import cv2  # type: ignore

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        bgr = rgb[..., ::-1].copy()

    res = ocr.ocr(bgr, cls=True)
    lines: List[str] = []
    boxes: List[Dict[str, Any]] = []

    # PaddleOCR output format: list of pages, each page is a list of
    # [box_coords, (text, confidence)] pairs.
    for page in res or []:
        for item in page:
            box, (text, conf) = item
            lines.append(str(text))
            boxes.append({
                "box": box,
                "text": str(text),
                "conf": float(conf),
            })

    return {
        "engine": "paddleocr",
        "lines": lines[:_MAX_LINES],
        "boxes": boxes[:_MAX_LINES],
    }
