from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .utils import to_gray_u8

# Optional PaddleOCR
try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:
    PaddleOCR = None


_OCR_INSTANCE = None


def _get_ocr() -> Optional[Any]:
    global _OCR_INSTANCE
    if PaddleOCR is None:
        return None
    if _OCR_INSTANCE is None:
        # lightweight defaults; adjust to your locale if needed
        _OCR_INSTANCE = PaddleOCR(use_angle_cls=True, lang="en")
    return _OCR_INSTANCE


def read_txt_ocr(txt_path: Union[str, Path], max_lines: int = 200) -> List[str]:
    p = Path(txt_path)
    if not p.exists():
        return []
    lines = []
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if ln:
            lines.append(ln)
        if len(lines) >= max_lines:
            break
    return lines


def run_ocr(rgb: np.ndarray, txt_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Returns:
      {engine, lines, boxes(optional), confidence(optional)}
    """
    if txt_path is not None:
        lines = read_txt_ocr(txt_path)
        return {"engine": "txt", "lines": lines, "boxes": None}

    ocr = _get_ocr()
    if ocr is None:
        return {"engine": "none", "lines": [], "boxes": None}

    # PaddleOCR expects BGR np array
    try:
        import cv2  # type: ignore
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        bgr = rgb[..., ::-1].copy()

    res = ocr.ocr(bgr, cls=True)
    lines: List[str] = []
    boxes: List[Any] = []

    # PaddleOCR output: list of [ [box, (text, conf)], ... ]
    for page in res or []:
        for item in page:
            box, (text, conf) = item
            lines.append(str(text))
            boxes.append({"box": box, "text": str(text), "conf": float(conf)})

    return {"engine": "paddleocr", "lines": lines[:300], "boxes": boxes[:300]}