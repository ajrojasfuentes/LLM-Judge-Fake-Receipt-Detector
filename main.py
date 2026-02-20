"""
main.py — CLI entry point for the LLM-Judge Fake Receipt Detector.

Dataset: Find it again! — Receipt Dataset for Document Forgery Detection
         https://l3i-share.univ-lr.fr/2023Finditagain/index.html

Commands:
  download    Download and extract the Find-It-Again dataset
  sample      Select 20 receipts (10 REAL + 10 FAKE) from the train split
  run         Run all 3 LLM judges on the sampled receipts, with optional forensic pre-analysis
  evaluate    Compute accuracy, precision, recall, F1, and confusion matrix
  demo        Run a single receipt through all 3 judges (quick demo)
  forensic    Run forensic pre-analysis on a single receipt and print the context report

Usage examples:
  python main.py download
  python main.py sample
  python main.py run
  python main.py run --forensic              # Run with forensic pre-analysis
  python main.py evaluate
  python main.py demo X00016469622
  python main.py forensic X00016469622
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_download(args):
    from pipeline.dataset import DatasetManager
    dm = DatasetManager()
    dm.download()
    dm.extract()
    print("[download] Done. Dataset extracted to data/raw/findit2/")


def cmd_sample(args):
    from pipeline.dataset import DatasetManager
    from pipeline.sampler import ReceiptSampler

    dm = DatasetManager()
    sampler = ReceiptSampler()

    print(f"[sample] Loading labels from '{sampler.split}' split ...")
    labels = dm.load_labels(sampler.split)

    print(f"[sample] Selecting {sampler.real_count} REAL + {sampler.fake_count} FAKE "
          f"(seed={sampler.random_seed}) ...")
    sample = sampler.sample(labels, dataset_manager=dm)
    sampler.save(sample)

    print(f"\n[sample] Selected {len(sample)} receipts:")
    for r in sample:
        img_status = "found" if r.get("image_path") else "NOT FOUND"
        txt_status = "found" if r.get("ocr_txt_path") else "not found"
        print(f"  {r['id']:40s} {r['label']}  img:{img_status}  ocr:{txt_status}")


def cmd_run(args):
    from pipeline.dataset import DatasetManager
    from pipeline.sampler import ReceiptSampler
    from judges.qwen_judge import make_forensic_accountant, make_document_examiner
    from judges.internvl_judge import InternVLJudge
    from judges.voting import VotingEngine
    import yaml

    use_forensic = getattr(args, "forensic", False)

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

    # Initialize forensic pipeline if requested
    forensic_pipeline = None
    if use_forensic:
        from pipeline.forensic_pipeline import ForensicPipeline
        forensic_pipeline = ForensicPipeline(
            output_dir="outputs/forensic",
            save_images=True,
            verbose=False,
        )
        print("[run] Forensic pre-analysis ENABLED — signals prepended to judge prompts.")

    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for receipt in sample:
        receipt_id = receipt["id"]
        split = receipt.get("split", "train")

        # Resolve image path (try from sample dict first, then search)
        image_path = None
        if receipt.get("image_path"):
            p = Path(receipt["image_path"])
            if p.exists():
                image_path = p
        if image_path is None:
            image_path = dm.find_image(receipt_id, split)

        if image_path is None:
            print(f"[run] WARNING: Image not found for {receipt_id}. Skipping.")
            continue

        print(f"\n[run] Processing: {receipt_id} (GT: {receipt['label']})")

        # Run forensic pre-analysis
        forensic_context = None
        if forensic_pipeline is not None:
            ocr_path = receipt.get("ocr_txt_path")
            if ocr_path is None:
                ocr_path = dm.find_ocr_txt(receipt_id, split)
            print(f"  → Forensic analysis ...", end=" ", flush=True)
            forensic_context = forensic_pipeline.analyze(
                image_path,
                ocr_txt_path=ocr_path,
            )
            mela_info = (f"MELA={forensic_context.multi_ela_suspicious_ratio:.0%}"
                         if forensic_context.multi_ela_suspicious_ratio is not None
                         else "MELA=N/A")
            cm_info = (f"CM={forensic_context.cm_confidence:.2f}"
                       if forensic_context.cm_confidence is not None else "CM=N/A")
            arith = forensic_context.ocr_arithmetic_report
            arith_info = ""
            if arith is not None:
                consistent = arith.get("arithmetic_consistent")
                if consistent is False:
                    arith_info = "  ARITH=⚠"
                elif consistent is True:
                    arith_info = "  ARITH=✓"
            print(f"{mela_info}  {cm_info}{arith_info}")

        # Run judges
        judge_results = []
        for judge in judges:
            print(f"  → {judge.judge_name} ...", end=" ", flush=True)
            result = judge.judge(
                receipt_id=receipt_id,
                image_path=image_path,
                forensic_context=forensic_context,
            )
            print(f"{result.label} ({result.confidence:.1f}%)")
            judge_results.append(result)

        verdict = engine.aggregate(judge_results)
        output = verdict.to_dict()
        output["ground_truth"] = receipt["label"]
        output["forensic_used"] = use_forensic

        out_path = results_dir / f"{receipt_id}.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        match = "CORRECT" if verdict.label == receipt["label"] else "WRONG"
        u_str = f"  u={verdict.verdict_uncertainty:.2f}" if hasattr(verdict, "verdict_uncertainty") else ""
        print(f"  → VERDICT: {verdict.tally}{u_str}  GT: {receipt['label']}  [{match}]")

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
        print(f"\nReceipt: {case['receipt_id']}  GT: {case.get('ground_truth', '?')}  "
              f"Verdict: {case['label']}")
        for j in case.get("judges", []):
            print(f"  [{j['judge_name']}] {j['label']} ({j['confidence']:.1f}%) "
                  f"— {j['reasons'][:2]}")


def cmd_demo(args):
    """Quick demo: run 3 judges on a single receipt and print results."""
    from pipeline.dataset import DatasetManager
    from judges.qwen_judge import make_forensic_accountant, make_document_examiner
    from judges.internvl_judge import InternVLJudge
    from judges.voting import VotingEngine

    receipt_id = args.receipt_id
    use_forensic = getattr(args, "forensic", False)

    dm = DatasetManager()

    # Try all splits to find the image
    image_path = None
    found_split = None
    for split in ["train", "val", "test"]:
        p = dm.find_image(receipt_id, split)
        if p is not None:
            image_path = p
            found_split = split
            break

    if image_path is None:
        print(f"[demo] ERROR: Image not found for ID '{receipt_id}'")
        sys.exit(1)

    print(f"\n=== DEMO: {receipt_id} ===")
    print(f"Image : {image_path}")
    print(f"Split : {found_split}")

    # Forensic pre-analysis (optional)
    forensic_context = None
    if use_forensic:
        from pipeline.forensic_pipeline import ForensicPipeline
        fp = ForensicPipeline(output_dir="outputs/forensic", save_images=True, verbose=True)
        ocr_path = dm.find_ocr_txt(receipt_id, found_split)
        print(f"\nRunning forensic pre-analysis ...")
        forensic_context = fp.analyze(image_path, ocr_txt_path=ocr_path)
        print(forensic_context.to_prompt_section())

    judges = [
        make_forensic_accountant(),
        make_document_examiner(),
        InternVLJudge(),
    ]
    engine = VotingEngine()

    results = []
    for judge in judges:
        print(f"\nRunning {judge.judge_name}...")
        r = judge.judge(
            receipt_id=receipt_id,
            image_path=image_path,
            forensic_context=forensic_context,
        )
        results.append(r)
        print(json.dumps(r.to_dict(), indent=2))

    verdict = engine.aggregate(results)
    print("\n=== FINAL VERDICT ===")
    print(json.dumps(verdict.to_dict(), indent=2))


def cmd_forensic(args):
    """Run forensic pre-analysis on a single receipt and print the full context report."""
    from pipeline.dataset import DatasetManager
    from pipeline.forensic_pipeline import ForensicPipeline

    receipt_id = args.receipt_id
    dm = DatasetManager()

    image_path = None
    found_split = None
    for split in ["train", "val", "test"]:
        p = dm.find_image(receipt_id, split)
        if p is not None:
            image_path = p
            found_split = split
            break

    if image_path is None:
        print(f"[forensic] ERROR: Image not found for ID '{receipt_id}'")
        sys.exit(1)

    ocr_path = dm.find_ocr_txt(receipt_id, found_split)

    print(f"\n=== FORENSIC ANALYSIS: {receipt_id} ===")
    print(f"Image : {image_path}")
    print(f"OCR   : {ocr_path or 'not found'}")
    print(f"Split : {found_split}\n")

    fp = ForensicPipeline(
        output_dir="outputs/forensic",
        save_images=True,
        verbose=True,
    )
    ctx = fp.analyze(image_path, ocr_txt_path=ocr_path)

    print("\n" + ctx.to_prompt_section())

    if ctx.errors:
        print(f"\n[forensic] Analysis errors: {ctx.errors}")
    else:
        print("\n[forensic] Analysis complete. Forensic images saved to outputs/forensic/")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM-Judge Fake Receipt Detector — Find it again! dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py download\n"
            "  python main.py sample\n"
            "  python main.py run\n"
            "  python main.py run --forensic\n"
            "  python main.py evaluate\n"
            "  python main.py demo X00016469622\n"
            "  python main.py demo X00016469622 --forensic\n"
            "  python main.py forensic X00016469622\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="Download and extract the Find-It-Again dataset")
    sub.add_parser("sample", help="Select 20 receipts (10 REAL + 10 FAKE) from train split")

    run_p = sub.add_parser("run", help="Run 3 LLM judges on all sampled receipts")
    run_p.add_argument(
        "--forensic", action="store_true",
        help="Pre-compute forensic signals (Multi-ELA, noise, copy-move, OCR + arithmetic) and include in prompts",
    )

    sub.add_parser("evaluate", help="Compute evaluation metrics (accuracy, F1, confusion matrix)")

    demo_p = sub.add_parser("demo", help="Run judges on a single receipt (quick demo)")
    demo_p.add_argument("receipt_id", help="Receipt filename stem (without extension)")
    demo_p.add_argument(
        "--forensic", action="store_true",
        help="Include forensic pre-analysis in the demo",
    )

    forensic_p = sub.add_parser("forensic", help="Run forensic pre-analysis on a single receipt")
    forensic_p.add_argument("receipt_id", help="Receipt filename stem (without extension)")

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
        "forensic": cmd_forensic,
    }
    commands[args.command](args)
