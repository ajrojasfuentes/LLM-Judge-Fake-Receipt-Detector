"""forensics.pipeline — main orchestration for forensic evidence packs.

This pipeline runs graphic forensic analyzers (metadata/quality/entropy/skew,
MELA/noise/copy-move) and reading analyzers (OCR post-processing + semantic
checks from dataset-provided OCR text), then emits an evidence pack.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .copy_move import copy_move_detect
from .entropy_edges import entropy_edge_text_metrics
from .metadata import extract_metadata
from .mela import mela_analyze
from .noise_inconsistency import noise_inconsistency
from .ocr_postprocess import extract_ocr_from_txt
from .quality import blur_tile_stats, quality_metrics
from .semantic_check import semantic_checks_from_result
from .skew import estimate_skew
from .utils import ROI, ensure_dir, load_image_rgb, save_json

logger = logging.getLogger(__name__)

BASE_EVIDENCE_DIR = Path(__file__).parent / "evidence"
VALID_MODES = {"GRAPHIC", "READING", "FULL"}


def _default_output_dir(image_path: Path) -> Path:
    image_id = image_path.stem
    return ensure_dir(BASE_EVIDENCE_DIR / image_id)


def _build_mode_payload(mode: str, pack: Dict[str, Any]) -> Dict[str, Any]:
    mode = mode.upper()
    if mode == "GRAPHIC":
        return {
            "mode": "GRAPHIC",
            "image_id": pack["image_id"],
            "input_path": pack["input_path"],
            "output_dir": pack["output_dir"],
            "global": pack["global"],
            "forensic": pack["forensic"],
            "suspect_rois": pack["suspect_rois"],
            "copy_move_pairs": pack["copy_move_pairs"],
            "artifacts": pack["artifacts"],
            "errors": pack["errors"],
            "timing_ms": pack["timing_ms"],
        }

    if mode == "READING":
        return {
            "mode": "READING",
            "image_id": pack["image_id"],
            "input_path": pack["input_path"],
            "output_dir": pack["output_dir"],
            "reading": pack["reading"],
            "errors": pack["errors"],
            "timing_ms": pack["timing_ms"],
        }

    return {
        "mode": "FULL",
        **pack,
    }


def build_evidence_pack(
    image_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    ocr_txt: Optional[Union[str, Path]] = None,
    mode: str = "FULL",
) -> Dict[str, Any]:
    """Run forensics and build an evidence pack.

    If output_dir is omitted, artifacts are saved at:
      forensics/evidence/<image_id>/
    """
    mode = mode.upper()
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Valid: {sorted(VALID_MODES)}")

    t0 = time.time()
    image_path = Path(image_path)
    outp = ensure_dir(output_dir) if output_dir else _default_output_dir(image_path)
    errors: Dict[str, str] = {}

    rgb = load_image_rgb(image_path)

    # 1) Global/graphic metrics
    meta: Dict[str, Any] = {}
    q: Dict[str, Any] = {}
    q_tiles: Dict[str, Any] = {}
    eet: Dict[str, Any] = {}
    skew_result: Dict[str, Any] = {}

    for key, fn in [
        ("metadata", lambda: extract_metadata(image_path)),
        ("quality", lambda: quality_metrics(rgb)),
        ("blur_tiles", lambda: blur_tile_stats(rgb, tile=64)),
        ("entropy_edges", lambda: entropy_edge_text_metrics(rgb)),
        ("skew", lambda: estimate_skew(rgb)),
    ]:
        try:
            val = fn()
            if key == "metadata":
                meta = val
            elif key == "quality":
                q = val
            elif key == "blur_tiles":
                q_tiles = val
            elif key == "entropy_edges":
                eet = val
            elif key == "skew":
                skew_result = val
        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"
            logger.warning("%s failed: %s", key, exc)

    # 2) Forensic image analyses
    mela: Dict[str, Any] = {"summary": {}, "rois": [], "artifacts": {}}
    noise: Dict[str, Any] = {"summary": {}, "artifacts": {}}
    cm: Dict[str, Any] = {"summary": {}, "pairs": [], "artifacts": {}}

    try:
        mela = mela_analyze(rgb, out_dir=str(outp / "mela"))
    except Exception as exc:
        errors["mela"] = f"{type(exc).__name__}: {exc}"
        logger.warning("MELA failed: %s", exc)

    try:
        noise = noise_inconsistency(rgb, out_dir=str(outp / "noise"))
    except Exception as exc:
        errors["noise"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Noise failed: %s", exc)

    try:
        cm = copy_move_detect(rgb, out_dir=str(outp / "copymove"))
    except Exception as exc:
        errors["copy_move"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Copy-move failed: %s", exc)

    # 3) OCR postprocess + semantic checks (dataset-provided pseudo OCR)
    ocr_txt_path: Optional[Path] = Path(ocr_txt) if ocr_txt else None
    reading: Dict[str, Any] = {
        "source": "ocr_postprocess",
        "txt_path": str(ocr_txt_path) if ocr_txt_path else None,
        "structured": None,
        "arithmetic_report": None,
        "semantic_checks": [],
    }

    if ocr_txt_path is not None and ocr_txt_path.exists():
        try:
            ocr_res = extract_ocr_from_txt(ocr_txt_path)
            reading["structured"] = ocr_res.structured.to_dict()
            reading["arithmetic_report"] = ocr_res.arithmetic_report.to_dict()
            reading["semantic_checks"] = semantic_checks_from_result(ocr_res)
            reading["quality_score"] = ocr_res.structured.quality_score
        except Exception as exc:
            errors["ocr_postprocess"] = f"{type(exc).__name__}: {exc}"
            logger.warning("OCR postprocess failed: %s", exc)
    else:
        errors["ocr_postprocess"] = "OCR .txt path not provided or file does not exist"

    rois: List[ROI] = [r for r in mela.get("rois", []) if isinstance(r, ROI)]

    artifacts: Dict[str, Any] = {
        "input_image": str(image_path),
        "mela": mela.get("artifacts", {}),
        "noise": noise.get("artifacts", {}),
        "copymove": cm.get("artifacts", {}),
    }

    core_pack: Dict[str, Any] = {
        "image_id": image_path.name,
        "input_path": str(image_path),
        "output_dir": str(outp),
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
        "reading": reading,
        "artifacts": artifacts,
        "errors": errors,
        "timing_ms": {"total": int((time.time() - t0) * 1000)},
    }

    pack = _build_mode_payload(mode, core_pack)
    json_path = str(outp / "evidence_pack.json")
    pack["evidence_json"] = json_path
    save_json(pack, json_path)
    return pack


def main() -> None:
    ap = argparse.ArgumentParser(description="Run modular forensic analysis on a receipt image.")
    ap.add_argument("--image", required=True, help="Path to receipt image (PNG/JPG).")
    ap.add_argument(
        "--out",
        default=None,
        help="Output dir (default: forensics/evidence/<image_id>/).",
    )
    ap.add_argument("--ocr_txt", default=None, help="Path to paired OCR .txt file.")
    ap.add_argument("--mode", choices=sorted(VALID_MODES), default="FULL")
    args = ap.parse_args()

    pack = build_evidence_pack(args.image, args.out, ocr_txt=args.ocr_txt, mode=args.mode)
    print(pack["evidence_json"])


if __name__ == "__main__":
    main()
