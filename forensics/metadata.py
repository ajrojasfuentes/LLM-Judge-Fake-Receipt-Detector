"""
forensics.metadata — Image metadata extraction.

Extracts structural metadata from the image file header using Pillow
(PIL).  The returned dictionary includes:

* File size, name, and format (PNG / JPEG / etc.).
* Dimensions, aspect ratio, and colour mode.
* DPI (horizontal / vertical resolution) when available.
* Alpha-channel and bit-depth information.
* EXIF orientation tag (tag 274) — present mainly in JPEG files.

This module does not produce output files; it returns a single
dictionary of scalar values consumed by the evidence pack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

from PIL import Image


def extract_metadata(image_path: Union[str, Path]) -> Dict[str, Any]:
    """Extract structural metadata from an image file.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file.

    Returns
    -------
    dict
        Metadata fields.  An ``"error"`` key is present (non-``None``)
        only when the image cannot be opened.

        Notable keys:

        * ``"format"`` — Image codec (e.g. ``"PNG"``, ``"JPEG"``).
        * ``"width"`` / ``"height"`` — Pixel dimensions.
        * ``"aspect_ratio"`` — ``width / height``.
        * ``"horizontal_resolution"`` / ``"vertical_resolution"`` — DPI.
        * ``"has_alpha"`` — Whether an alpha channel is present.
        * ``"bit_depth"`` — Estimated bit depth from the colour mode.
        * ``"exif_orientation"`` — EXIF orientation tag value, or
          ``None``.
    """
    p = Path(image_path)
    meta: Dict[str, Any] = {
        "path": str(p),
        "filename": p.name,
        "bytes": p.stat().st_size if p.exists() else None,
        "error": None,
    }
    try:
        im = Image.open(p)
        meta["format"] = im.format
        meta["mode"] = im.mode
        meta["width"], meta["height"] = im.size

        # DPI — may be absent in many image files
        dpi = im.info.get("dpi")
        if dpi and isinstance(dpi, (tuple, list)) and len(dpi) >= 2:
            meta["horizontal_resolution"] = float(dpi[0])
            meta["vertical_resolution"] = float(dpi[1])
        else:
            meta["horizontal_resolution"] = None
            meta["vertical_resolution"] = None

        # Alpha channel detection
        meta["has_alpha"] = (
            "A" in (im.mode or "")
        ) or (
            "transparency" in im.info
        )

        # Bit depth — best-effort estimation from the colour mode
        mode = im.mode or ""
        if mode in ("1", "L", "P"):
            meta["bit_depth"] = 8
        elif mode == "RGB":
            meta["bit_depth"] = 24
        elif mode == "RGBA":
            meta["bit_depth"] = 32
        elif mode in ("I;16", "I;16B", "I;16L"):
            meta["bit_depth"] = 16
        else:
            meta["bit_depth"] = None

        # EXIF orientation (tag 274) — often absent in PNG files
        exif = im.getexif() if hasattr(im, "getexif") else None
        orientation = None
        if exif:
            orientation = exif.get(274, None)
        meta["exif_orientation"] = orientation

        # Aspect ratio
        w, h = im.size
        meta["aspect_ratio"] = float(w) / float(h) if h else None

    except Exception as e:
        meta["error"] = f"{type(e).__name__}: {e}"
    return meta
