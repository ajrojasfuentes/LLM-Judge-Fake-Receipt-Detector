"""
main.py — CLI entry point for the LLM-Judge Fake Receipt Detector.

Commands:
  download    Download and extract the dataset
  sample      Select 20 receipts (10 REAL + 10 FAKE) with a fixed seed
  run         Run all 3 judges on the sampled receipts and save results
  evaluate    Compute accuracy + confusion matrix + disagreement cases
  demo        Run a single receipt through all 3 judges (quick demo)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_download(args):
    from pipeline.dataset import DatasetManager
    dm = DatasetManager()
    dm.download()
    dm.extract()
    print("[download] Done.")


def cmd_sample(args):
    from pipeline.dataset import DatasetManager
    from pipeline.sampler import ReceiptSampler
    dm = DatasetManager()
    labels = dm.load_labels()
    sampler = ReceiptSampler()
    sample = sampler.sample(labels)
    sampler.save(sample)
    print(f"[sample] Selected {len(sample)} receipts:")
    for r in sample:
        print(f"  {r['id']:40s} {r['label']}")


def cmd_run(args):
    from pipeline.dataset import DatasetManager
    from pipeline.sampler import ReceiptSampler
    from judges.qwen_judge import make_forensic_accountant, make_document_examiner
    from judges.internvl_judge import InternVLJudge
    from judges.voting import VotingEngine
    import yaml

    dm = DatasetManager()
    sampler = ReceiptSampler()
    sample = sampler.load()

    judges = [
        make_forensic_accountant(),
        make_document_examiner(),
        InternVLJudge(),
    ]

    cfg_path = Path("configs/judges.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    voting_cfg = cfg.get("voting", {})
    engine = VotingEngine(
        strategy=voting_cfg.get("strategy", "majority"),
        uncertain_threshold=voting_cfg.get("uncertain_threshold", 2),
    )

    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for receipt in sample:
        receipt_id = receipt["id"]
        image_path = dm.find_image(receipt_id)

        if image_path is None:
            print(f"[run] WARNING: Image not found for {receipt_id}. Skipping.")
            continue

        print(f"\n[run] Processing receipt: {receipt_id} (GT: {receipt['label']})")
        judge_results = []
        for judge in judges:
            print(f"  → {judge.judge_name} ...", end=" ", flush=True)
            result = judge.judge(receipt_id=receipt_id, image_path=image_path)
            print(f"{result.label} ({result.confidence:.1f}%)")
            judge_results.append(result)

        verdict = engine.aggregate(judge_results)
        output = verdict.to_dict()
        output["ground_truth"] = receipt["label"]

        out_path = results_dir / f"{receipt_id}.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"  → VERDICT: {verdict.tally}  (GT: {receipt['label']})")

    print("\n[run] All receipts processed.")


def cmd_evaluate(args):
    from pipeline.evaluator import Evaluator
    from pipeline.sampler import ReceiptSampler

    sampler = ReceiptSampler()
    sample = sampler.load()
    ground_truth = {r["id"]: r["label"] for r in sample}

    ev = Evaluator()
    ev.load_results()

    summary = ev.summary(ground_truth)
    print("\n=== EVALUATION SUMMARY ===")
    print(json.dumps(summary, indent=2))

    print("\n=== DISAGREEMENT CASES ===")
    for case in ev.disagreement_cases(n=3):
        print(f"\nReceipt: {case['receipt_id']}  GT: {case.get('ground_truth', '?')}  Verdict: {case['label']}")
        for j in case.get("judges", []):
            print(f"  [{j['judge_name']}] {j['label']} ({j['confidence']:.1f}%) — {j['reasons'][:2]}")


def cmd_demo(args):
    """Quick demo: run 3 judges on a single receipt and print results."""
    from pipeline.dataset import DatasetManager
    from judges.qwen_judge import make_forensic_accountant, make_document_examiner
    from judges.internvl_judge import InternVLJudge
    from judges.voting import VotingEngine

    receipt_id = args.receipt_id
    dm = DatasetManager()
    image_path = dm.find_image(receipt_id)

    if image_path is None:
        print(f"[demo] ERROR: Image not found for ID '{receipt_id}'")
        sys.exit(1)

    print(f"\n=== DEMO: {receipt_id} ===")
    print(f"Image: {image_path}\n")

    judges = [
        make_forensic_accountant(),
        make_document_examiner(),
        InternVLJudge(),
    ]
    engine = VotingEngine()

    results = []
    for judge in judges:
        print(f"Running {judge.judge_name}...")
        r = judge.judge(receipt_id=receipt_id, image_path=image_path)
        results.append(r)
        print(json.dumps(r.to_dict(), indent=2))
        print()

    verdict = engine.aggregate(results)
    print("=== FINAL VERDICT ===")
    print(json.dumps(verdict.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM-Judge Fake Receipt Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="Download and extract the dataset")
    sub.add_parser("sample", help="Select 20 receipts (10 REAL + 10 FAKE)")
    sub.add_parser("run", help="Run 3 judges on all sampled receipts")
    sub.add_parser("evaluate", help="Compute evaluation metrics")

    demo_p = sub.add_parser("demo", help="Run judges on a single receipt (quick demo)")
    demo_p.add_argument("receipt_id", help="Receipt filename stem (without extension)")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "download": cmd_download,
        "sample": cmd_sample,
        "run": cmd_run,
        "evaluate": cmd_evaluate,
        "demo": cmd_demo,
    }
    commands[args.command](args)
