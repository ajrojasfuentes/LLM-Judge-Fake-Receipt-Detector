"""
forensics — Modular forensic image analysis toolkit for receipt forgery detection.

This package provides a pipeline of independent forensic analysis modules
that examine a receipt image for evidence of tampering, manipulation, or
forgery.  Each module produces structured results that are aggregated into
a single JSON evidence pack by the orchestration layer.
"""

from .utils import ROI


def build_evidence_pack(*args, **kwargs):
    """Lazy wrapper to avoid importing `forensics.pipeline` at package import time."""
    from .pipeline import build_evidence_pack as _build
    return _build(*args, **kwargs)


__all__ = [
    "build_evidence_pack",
    "pipeline",
    "ROI",
]
