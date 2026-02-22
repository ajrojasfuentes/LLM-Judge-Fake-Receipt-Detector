"""forensics — modular forensic toolkit for receipt fraud analysis.

Primary entry-point:
    from forensics.pipeline import build_evidence_pack
"""

from .pipeline import build_evidence_pack
from .utils import ROI

__all__ = ["build_evidence_pack", "ROI"]
