from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image


def extract_metadata(image_path: Union[str, Path]) -> Dict[str, Any]:
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

        # DPI if present
        dpi = im.info.get("dpi")
        if dpi and isinstance(dpi, (tuple, list)) and len(dpi) >= 2:
            meta["horizontal_resolution"] = float(dpi[0])
            meta["vertical_resolution"] = float(dpi[1])
        else:
            meta["horizontal_resolution"] = None
            meta["vertical_resolution"] = None

        # Alpha channel
        meta["has_alpha"] = ("A" in (im.mode or "")) or ("transparency" in im.info)

        # Bit depth (best effort)
        # PNG bit depth can be inferred loosely from mode
        mode = im.mode or ""
        if mode in ("1", "L", "P"):
            meta["bit_depth"] = 8
        elif mode in ("RGB",):
            meta["bit_depth"] = 24
        elif mode in ("RGBA",):
            meta["bit_depth"] = 32
        elif mode in ("I;16", "I;16B", "I;16L"):
            meta["bit_depth"] = 16
        else:
            meta["bit_depth"] = None

        # EXIF orientation (often absent in PNG)
        exif = im.getexif() if hasattr(im, "getexif") else None
        orientation = None
        if exif:
            orientation = exif.get(274, None)  # 274 = Orientation
        meta["exif_orientation"] = orientation

        # Aspect ratio
        w, h = im.size
        meta["aspect_ratio"] = float(w) / float(h) if h else None

    except Exception as e:
        meta["error"] = f"{type(e).__name__}: {e}"
    return meta