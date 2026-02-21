"""
Modular forensic analysis toolkit for receipt forgery detection.

Each analysis tool lives in its own module and shares common utilities
from ``forensic.utils``.  The ``pipeline.forensic_pipeline`` module
orchestrates them, collects results, and formats them for the VLM judges.

Modules
-------
utils          Shared image I/O, receipt crop/deskew, ROI helpers
mela           Enhanced Multi-Quality Error Level Analysis
noisemap       Block-based noise variance analysis
frequencydct   Per-block DCT + global FFT frequency analysis
cpi            Dense block copy-move (CPI) detection
"""

from .utils import CropResult, ROI, crop_receipt
from .mela import MELAResult, mela_analyze
from .noisemap import NoiseResult, noise_analyze
from .frequencydct import FrequencyResult, frequency_analyze
from .cpi import CPIResult, CPIPair, cpi_analyze

__all__ = [
    "CropResult", "ROI", "crop_receipt",
    "MELAResult", "mela_analyze",
    "NoiseResult", "noise_analyze",
    "FrequencyResult", "frequency_analyze",
    "CPIResult", "CPIPair", "cpi_analyze",
]
