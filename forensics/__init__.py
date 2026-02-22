"""
forensics — Modular forensic image analysis toolkit for receipt
forgery detection.

This package provides a pipeline of independent forensic analysis modules
that examine a receipt image for evidence of tampering, manipulation, or
forgery.  Each module produces structured results that are aggregated into
a single JSON evidence pack by the orchestration layer.

Modules
-------
pipeline            Orchestration entry-point: ``build_evidence_pack()``.
mela                Multi-Quality Error Level Analysis (JPEG compression
                    artifact detection via cross-quality variance fusion).
copy_move           Copy-Move forgery detection using ORB keypoint matching
                    and translation-vector clustering.
noise_inconsistency Residual-noise-based per-tile inconsistency analysis.
entropy_edges       Shannon entropy, Canny edge density, and adaptive
                    text-ink-area estimation.
quality             Global and per-tile quality metrics (brightness,
                    contrast, Laplacian blur variance).
skew                Document skew estimation via Hough line detection.
metadata            Image metadata extraction (format, DPI, EXIF, etc.).
semantic_checks     OCR-based accounting validation (subtotal + tax vs.
                    total cross-check).
ocr_adapter         Unified OCR interface with PaddleOCR engine and
                    plain-text file fallback.
utils               Shared image I/O helpers, dataclasses, and
                    JSON-serialization utilities.

Usage
-----
    from forensics_analysis.pipeline import build_evidence_pack

    pack = build_evidence_pack(
        image_path="receipt.png",
        output_dir="outputs/forensic",
        ocr_txt="receipt.txt",        # optional pre-existing OCR text
    )
"""

__all__ = [
    "build_evidence_pack",
    "pipeline",
    "ROI",
]

from .pipeline import build_evidence_pack
from .utils import ROI

def __getattr__(name: str):
    """Lazy imports to avoid circular/premature loading.

    This prevents the RuntimeWarning that occurs when running
    ``python -m forensics_analysis.pipeline`` — Python imports the
    package first (triggering ``__init__.py``), and an eager import
    of ``pipeline`` would place it in ``sys.modules`` before the
    ``-m`` runner can execute it as ``__main__``.

    Supported lazy names:
        build_evidence_pack — from .pipeline
        ROI                 — from .utils
    """
    if name == "build_evidence_pack":
        from .pipeline import build_evidence_pack
        return build_evidence_pack
    if name == "ROI":
        from .utils import ROI
        return ROI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")