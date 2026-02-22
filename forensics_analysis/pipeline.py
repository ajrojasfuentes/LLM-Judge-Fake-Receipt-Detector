from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

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


def build_evidence_pack(
    image_path: Union[str, Path],
    output_dir: Union[str, Path],
    ocr_txt: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    image_path = Path(image_path)
    outp = ensure_dir(output_dir)

    rgb = load_image_rgb(image_path)

    # 1) Global
    meta = extract_metadata(image_path)
    q = quality_metrics(rgb)
    q_tiles = blur_tile_stats(rgb, tile=64)
    eet = entropy_edge_text_metrics(rgb)
    skew = estimate_skew(rgb)

    # 2) Forensic
    mela = mela_analyze(rgb, out_dir=str(outp / "mela"))
    noise = noise_inconsistency(rgb, out_dir=str(outp / "noise"))
    cm = copy_move_detect(rgb, out_dir=str(outp / "copymove"))

    # 3) OCR + checks
    ocr = run_ocr(rgb, txt_path=ocr_txt)
    checks = semantic_checks(ocr.get("lines", []))

    # 4) Suspect ROIs: start with MELA ROIs, optionally attach OCR snippets later
    rois = []
    mela_rois = mela.get("rois", [])
    for r in mela_rois:
        if hasattr(r, "roi_id"):
            rois.append(r)

    # 5) Collect artifacts
    artifacts: Dict[str, Any] = {
        "input_image": str(image_path),
        "mela": mela.get("artifacts", {}),
        "noise": noise.get("artifacts", {}),
        "copymove": cm.get("artifacts", {}),
    }

    pack: Dict[str, Any] = {
        "image_id": image_path.name,
        "input_path": str(image_path),
        "output_dir": str(outp),
        "global": {
            **meta,
            **q,
            "blur_tiles": q_tiles,
            **eet,
            **skew,
        },
        "forensic": {
            "mela": mela.get("summary", {}),
            "noise": noise.get("summary", {}),
            "copy_move": cm.get("summary", {}),
        },
        "suspect_rois": rois,  # ROI dataclasses -> json_sanitize in utils.save_json
        "copy_move_pairs": cm.get("pairs", []),
        "ocr": {
            "engine": ocr.get("engine"),
            "num_lines": len(ocr.get("lines", []) or []),
            "sample_lines": (ocr.get("lines", []) or [])[:25],
            # boxes can be huge; keep optional
            "has_boxes": bool(ocr.get("boxes")),
        },
        "semantic_checks": checks,
        "artifacts": artifacts,
        "timing_ms": {"total": int((time.time() - t0) * 1000)},
    }

    # save pack json
    json_path = save_json(pack, outp / "evidence_pack.json")
    pack["evidence_json"] = json_path
    return pack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to receipt PNG/JPG")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--ocr_txt", default=None, help="Optional pseudo-OCR .txt path")
    args = ap.parse_args()

    pack = build_evidence_pack(args.image, args.out, ocr_txt=args.ocr_txt)
    print(pack["evidence_json"])


if __name__ == "__main__":
    main()