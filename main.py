"""High-level API + CLI for receipt judging pipeline.

This module can be imported from notebooks and used as an API, while still
providing CLI commands for dataset preparation, batch execution and evaluation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from forensics.pipeline import build_evidence_pack
from judges import VotingEngine, load_judges_from_config
from pipeline.dataset import DatasetManager
from pipeline.evaluator import Evaluator
from pipeline.sampler import ReceiptSampler


@dataclass
class RunConfig:
    judges_config_path: str = "configs/judges.yaml"
    results_dir: str = "outputs/results"
    forensic_mode: str = "full"  # graphic|reading|full


def _load_runtime_config(config_path: str = "configs/judges.yaml") -> Dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _build_forensic_prompt_section(pack: Dict[str, Any]) -> str:
    lines = [
        "=== FORENSIC PRE-ANALYSIS (SYSTEM GENERATED) ===",
        f"Mode: {pack.get('mode', 'FULL')}",
    ]

    if "forensic" in pack:
        forensic = pack.get("forensic", {})
        mela = forensic.get("mela", {})
        noise = forensic.get("noise", {})
        cm = forensic.get("copy_move", {})
        lines += [
            f"MELA suspicious_ratio: {mela.get('suspicious_ratio', 'n/a')}",
            f"MELA peak_score: {mela.get('peak_score', 'n/a')}",
            f"Noise max_tile_z: {noise.get('max_tile_z', 'n/a')}",
            f"CopyMove pairs_found: {cm.get('pairs_found', 'n/a')}",
        ]

    reading = pack.get("reading")
    if isinstance(reading, dict):
        ar = reading.get("arithmetic_report") or {}
        lines += [
            f"OCR quality_score: {reading.get('quality_score', 'n/a')}",
            f"Arithmetic consistent: {ar.get('arithmetic_consistent', 'n/a')}",
            f"Arithmetic best_explanation: {ar.get('best_explanation', 'n/a')}",
        ]

    if pack.get("errors"):
        lines.append(f"Pipeline errors: {pack['errors']}")

    lines.append("=== END FORENSIC PRE-ANALYSIS ===")
    return "\n".join(lines)


def _collect_aux_images(image_path: Path, mode: str) -> List[Path]:
    mode = mode.lower()
    if mode not in {"graphic", "full"}:
        return []

    image_id = image_path.stem
    base = Path("forensics/evidence") / image_id
    candidates = [
        base / "mela" / "mela_heat.png",
        base / "mela" / "mela_overlay.png",
        base / "mela" / "mela_rois.png",
        base / "noise" / "noise_overlay.png",
        base / "noise" / "noise_heat.png",
        base / "copymove" / "copymove_rois.png",
    ]
    return [p for p in candidates if p.exists()]


def run_single_receipt(
    receipt_id: str,
    split: Optional[str] = None,
    ground_truth: Optional[str] = None,
    config: Optional[RunConfig] = None,
) -> Dict[str, Any]:
    cfg = config or RunConfig()
    raw_cfg = _load_runtime_config(cfg.judges_config_path)
    runtime = raw_cfg.get("runtime", {})
    forensic_mode = runtime.get("forensic_mode", cfg.forensic_mode)

    dm = DatasetManager()

    found_split = split
    image_path = None
    ocr_path = None

    if split:
        image_path = dm.find_image(receipt_id, split)
        ocr_path = dm.find_ocr_txt(receipt_id, split)
    else:
        for s in ["train", "val", "test"]:
            image_path = dm.find_image(receipt_id, s)
            if image_path is not None:
                found_split = s
                ocr_path = dm.find_ocr_txt(receipt_id, s)
                break

    if image_path is None:
        raise FileNotFoundError(f"Image not found for receipt_id={receipt_id}")

    evidence_pack = build_evidence_pack(
        image_path=image_path,
        output_dir=None,
        ocr_txt=ocr_path,
        mode=forensic_mode,
    )
    forensic_prompt = _build_forensic_prompt_section(evidence_pack)
    aux_images = _collect_aux_images(image_path, forensic_mode)

    judges = load_judges_from_config(cfg.judges_config_path)
    voting_cfg = raw_cfg.get("voting", {})
    engine = VotingEngine(
        strategy=voting_cfg.get("strategy", "majority_simple"),
        uncertain_threshold=voting_cfg.get("uncertain_threshold", 2),
    )

    judge_results = []
    for judge in judges:
        result = judge.judge(
            receipt_id=receipt_id,
            image_path=image_path,
            forensic_context=forensic_prompt,
            extra_image_paths=aux_images,
        )
        judge_results.append(result)

    verdict = engine.aggregate(judge_results)
    out = verdict.to_dict()
    out["forensic_mode"] = forensic_mode
    out["forensic_evidence_json"] = evidence_pack.get("evidence_json")
    out["forensic_aux_images"] = [str(p) for p in aux_images]
    out["split"] = found_split
    if ground_truth:
        out["ground_truth"] = ground_truth
    return out


def run_sample_batch(config: Optional[RunConfig] = None) -> List[Dict[str, Any]]:
    cfg = config or RunConfig()
    sampler = ReceiptSampler()
    sample = sampler.load()

    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[Dict[str, Any]] = []
    for rec in sample:
        result = run_single_receipt(
            receipt_id=rec["id"],
            split=rec.get("split"),
            ground_truth=rec.get("label"),
            config=cfg,
        )
        outputs.append(result)
        with open(out_dir / f"{rec['id']}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return outputs


def evaluate_results() -> Dict[str, Any]:
    sampler = ReceiptSampler()
    sample = sampler.load()
    ground_truth = {r["id"]: r["label"] for r in sample}

    ev = Evaluator()
    ev.load_results()
    return ev.summary(ground_truth)


# ---------------- CLI ---------------- #

def cmd_download(_args):
    dm = DatasetManager()
    dm.download()
    dm.extract()
    print("[download] Done")


def cmd_sample(_args):
    dm = DatasetManager()
    sampler = ReceiptSampler()
    labels = dm.load_labels(sampler.split)
    sample = sampler.sample(labels, dataset_manager=dm)
    sampler.save(sample)
    print(f"[sample] Saved {len(sample)} receipts")


def cmd_run(_args):
    outputs = run_sample_batch()
    print(f"[run] Processed {len(outputs)} receipts")


def cmd_evaluate(_args):
    print(json.dumps(evaluate_results(), indent=2, ensure_ascii=False))


def cmd_demo(args):
    result = run_single_receipt(receipt_id=args.receipt_id)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_forensic(args):
    dm = DatasetManager()
    image_path = None
    found_split = None
    for s in ["train", "val", "test"]:
        image_path = dm.find_image(args.receipt_id, s)
        if image_path is not None:
            found_split = s
            break
    if image_path is None:
        raise FileNotFoundError(f"Image not found for {args.receipt_id}")

    ocr = dm.find_ocr_txt(args.receipt_id, found_split)
    cfg = _load_runtime_config()
    mode = cfg.get("runtime", {}).get("forensic_mode", "full")
    pack = build_evidence_pack(image_path=image_path, output_dir=None, ocr_txt=ocr, mode=mode)
    print(json.dumps(pack, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-Judge Fake Receipt Detector")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download")
    sub.add_parser("sample")
    sub.add_parser("run")
    sub.add_parser("evaluate")

    demo_p = sub.add_parser("demo")
    demo_p.add_argument("receipt_id")

    forensic_p = sub.add_parser("forensic")
    forensic_p.add_argument("receipt_id")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    commands = {
        "download": cmd_download,
        "sample": cmd_sample,
        "run": cmd_run,
        "evaluate": cmd_evaluate,
        "demo": cmd_demo,
        "forensic": cmd_forensic,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
