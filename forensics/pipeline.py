"""
forensics.pipeline — Orchestration layer for the modular forensic toolkit.

This module is the main entry-point for running all forensic analyses
on a single receipt image.  It coordinates the following steps:

1. **Global metrics** — Image metadata (format, DPI, EXIF), quality
   descriptors (brightness, contrast, blur), entropy / edge / text-ink
   metrics, and document skew estimation.
2. **Forensic analyses** — Multi-Quality Error Level Analysis (MELA),
   residual-noise inconsistency mapping, and ORB-based copy-move
   detection.
3. **OCR and semantic checks** — Text extraction via PaddleOCR or a
   pre-existing ``.txt`` file, followed by accounting cross-checks
   (subtotal + tax vs. total).
4. **Evidence packing** — All results are merged into a single
   JSON-serialisable dictionary (the *evidence pack*) and persisted to
   ``evidence_pack.json`` inside the output directory.

Each analysis step is wrapped in its own ``try / except`` block so that
a failure in one module does not prevent the remaining modules from
running.  Errors are recorded in the ``"errors"`` section of the
evidence pack.

Usage
-----
    from forensics.pipeline import build_evidence_pack

    pack = build_evidence_pack(
        image_path="receipt.png",
        output_dir="forensics/evidence/receipt_001",
        ocr_txt="receipt.txt",
    )

CLI
---
    python -m forensics.pipeline --image receipt.png
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .copy_move import copy_move_detect
from .entropy_edges import entropy_edge_text_metrics
from .metadata import extract_metadata
from .mela import mela_analyze
from .noise_inconsistency import noise_inconsistency
from .ocr_adapter import run_ocr
from .quality import blur_tile_stats, quality_metrics
from .semantic_checks import semantic_checks
from .skew import estimate_skew
from .utils import ROI, ensure_dir, load_image_rgb, save_json

logger = logging.getLogger(__name__)

_FORENSICS_DIR = Path(__file__).resolve().parent
_DEFAULT_EVIDENCE_ROOT = _FORENSICS_DIR / "evidence"
_LEGACY_TEMPLATE_PATH = _FORENSICS_DIR / "evidence_pack_template.json"
_V2_TEMPLATE_PATH = _FORENSICS_DIR / "evidence_template_v2.json"


def _resolve_evidence_dir(image_path: Path, output_dir: Optional[Union[str, Path]]) -> Path:
    """Resolve evidence directory as `forensics/evidence/<image_id>/` by default."""
    image_id = image_path.stem
    if output_dir is None:
        return ensure_dir(_DEFAULT_EVIDENCE_ROOT / image_id)
    return ensure_dir(output_dir)


def _resolve_ocr_txt_path(image_path: Path, ocr_txt: Optional[Union[str, Path]]) -> Optional[Path]:
    """Prefer explicit OCR txt path; otherwise try `<image>.txt` beside image."""
    if ocr_txt is not None:
        p = Path(ocr_txt)
        return p if p.exists() else None

    inferred = image_path.with_suffix(".txt")
    if inferred.exists():
        return inferred
    return None


def _load_template_version() -> str:
    """Return evidence template version available in repo (v2 preferred)."""
    for candidate in (_V2_TEMPLATE_PATH, _LEGACY_TEMPLATE_PATH):
        if not candidate.exists():
            continue
        try:
            with open(candidate, encoding="utf-8") as f:
                data = json.load(f)
            return str(data.get("template_version", candidate.stem))
        except Exception:
            return candidate.stem
    return "custom"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_evidence_pack(
    image_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    ocr_txt: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run every forensic module and assemble the evidence pack.

    Parameters
    ----------
    image_path : str or Path
        Path to the receipt image (PNG, JPEG, etc.).
    output_dir : str or Path, optional
        Directory where output artifacts and ``evidence_pack.json`` are saved.
        Default: ``forensics/evidence/<image_id>/``.
    ocr_txt : str or Path, optional
        Path to a pre-existing plain-text OCR transcription.  When
        provided, PaddleOCR is skipped and lines are read from this
        file instead.

    Returns
    -------
    dict
        The evidence pack — a JSON-serialisable dictionary with the
        following top-level keys:

        * ``"image_id"`` — Filename of the input image.
        * ``"input_path"`` — Absolute string path to the input image.
        * ``"output_dir"`` — Absolute string path to the output
          directory.
        * ``"global"`` — Merged metadata, quality, entropy, edge, and
          skew metrics.
        * ``"forensic"`` — Per-module summary dicts for MELA, noise,
          and copy-move analyses.
        * ``"suspect_rois"`` — List of :class:`ROI` instances flagged
          by MELA.
        * ``"copy_move_pairs"`` — List of copy-move pair dicts.
        * ``"ocr"`` — OCR engine info, line count, and sample lines.
        * ``"semantic_checks"`` — List of accounting check results.
        * ``"artifacts"`` — Paths to all saved image artifacts.
        * ``"errors"`` — Dict of module-name to error message for any
          step that failed.
        * ``"timing_ms"`` — Execution time in milliseconds.
        * ``"evidence_json"`` — Path to the saved JSON file.
    """
    t0 = time.time()
    image_path = Path(image_path)
    outp = _resolve_evidence_dir(image_path, output_dir)
    ocr_txt_path = _resolve_ocr_txt_path(image_path, ocr_txt)

    errors: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Load the source image
    # ------------------------------------------------------------------
    rgb = load_image_rgb(image_path)

    # ------------------------------------------------------------------
    # 1) Global metrics (metadata, quality, entropy/edges, skew)
    # ------------------------------------------------------------------
    meta: Dict[str, Any] = {}
    try:
        meta = extract_metadata(image_path)
    except Exception as exc:
        errors["metadata"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Metadata extraction failed: %s", exc)

    q: Dict[str, Any] = {}
    try:
        q = quality_metrics(rgb)
    except Exception as exc:
        errors["quality"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Quality metrics failed: %s", exc)

    q_tiles: Dict[str, Any] = {}
    try:
        q_tiles = blur_tile_stats(rgb, tile=64)
    except Exception as exc:
        errors["blur_tiles"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Blur tile stats failed: %s", exc)

    eet: Dict[str, Any] = {}
    try:
        eet = entropy_edge_text_metrics(rgb)
    except Exception as exc:
        errors["entropy_edges"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Entropy/edge metrics failed: %s", exc)

    skew_result: Dict[str, Any] = {}
    try:
        skew_result = estimate_skew(rgb)
    except Exception as exc:
        errors["skew"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Skew estimation failed: %s", exc)

    # ------------------------------------------------------------------
    # 2) Forensic analyses (MELA, noise, copy-move)
    # ------------------------------------------------------------------
    mela: Dict[str, Any] = {"summary": {}, "rois": [], "artifacts": {}}
    try:
        mela = mela_analyze(rgb, out_dir=str(outp))
    except Exception as exc:
        errors["mela"] = f"{type(exc).__name__}: {exc}"
        logger.warning("MELA analysis failed: %s", exc)

    noise: Dict[str, Any] = {"summary": {}, "artifacts": {}}
    try:
        noise = noise_inconsistency(rgb, out_dir=str(outp))
    except Exception as exc:
        errors["noise"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Noise analysis failed: %s", exc)

    cm: Dict[str, Any] = {"summary": {}, "pairs": [], "artifacts": {}}
    try:
        cm = copy_move_detect(rgb, out_dir=str(outp))
    except Exception as exc:
        errors["copy_move"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Copy-move detection failed: %s", exc)

    # ------------------------------------------------------------------
    # 3) OCR + semantic checks
    # ------------------------------------------------------------------
    ocr: Dict[str, Any] = {"engine": "none", "lines": [], "boxes": None}
    try:
        ocr = run_ocr(rgb, txt_path=ocr_txt_path)
    except Exception as exc:
        errors["ocr"] = f"{type(exc).__name__}: {exc}"
        logger.warning("OCR failed: %s", exc)

    checks: List[Dict[str, Any]] = []
    try:
        checks = semantic_checks(ocr.get("lines", []) or [])
    except Exception as exc:
        errors["semantic_checks"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Semantic checks failed: %s", exc)

    # ------------------------------------------------------------------
    # 4) Collect suspect ROIs from MELA
    # ------------------------------------------------------------------
    rois: List[ROI] = []
    for r in mela.get("rois", []):
        if isinstance(r, ROI):
            rois.append(r)

    # ------------------------------------------------------------------
    # 5) Assemble the evidence pack
    # ------------------------------------------------------------------
    artifacts: Dict[str, Any] = {
        "input_image": str(image_path),
        "mela": mela.get("artifacts", {}),
        "noise": noise.get("artifacts", {}),
        "copymove": cm.get("artifacts", {}),
    }

    ocr_lines = ocr.get("lines", []) or []

    pack: Dict[str, Any] = {
        "image_id": image_path.name,
        "input_path": str(image_path),
        "output_dir": str(outp),
        "template_version": _load_template_version(),
        "global": {
            **meta,
            **q,
            "blur_tiles": q_tiles,
            **eet,
            **skew_result,
        },
        "forensic": {
            "mela": mela.get("summary", {}),
            "noise": noise.get("summary", {}),
            "copy_move": cm.get("summary", {}),
        },
        "suspect_rois": rois,
        "copy_move_pairs": cm.get("pairs", []),
        "ocr": {
            "engine": ocr.get("engine"),
            "source_txt": str(ocr_txt_path) if ocr_txt_path else None,
            "num_lines": len(ocr_lines),
            "sample_lines": ocr_lines[:25],
            "has_boxes": bool(ocr.get("boxes")),
        },
        "semantic_checks": checks,
        "artifacts": artifacts,
        "errors": errors,
        "timing_ms": {"total": int((time.time() - t0) * 1000)},
    }

    # Persist the evidence pack to disk
    json_path = save_json(pack, outp / "evidence_pack.json")
    pack["evidence_json"] = json_path
    return pack


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line interface for running the forensic pipeline."""
    ap = argparse.ArgumentParser(
        description="Run modular forensic analysis on a receipt image.",
    )
    ap.add_argument(
        "--image", required=True,
        help="Path to the receipt image (PNG/JPG).",
    )
    ap.add_argument(
        "--out", default=None,
        help=(
            "Output directory for artifacts and JSON. "
            "Default: forensics/evidence/<image_id>/"
        ),
    )
    ap.add_argument(
        "--ocr_txt", default=None,
        help="Optional path to a pre-existing OCR .txt transcription.",
    )
    args = ap.parse_args()

    pack = build_evidence_pack(args.image, args.out, ocr_txt=args.ocr_txt)
    print(pack["evidence_json"])


if __name__ == "__main__":
    main()
