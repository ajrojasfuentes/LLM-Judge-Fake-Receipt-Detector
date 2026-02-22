"""High-level API + CLI for LLM-Judge Fake Receipt Detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from judges.glm_judge import GLMJudge
from judges.qwen_judge import QwenJudge
from judges.voting import VotingEngine
from pipeline import DatasetManager, Evaluator, ForensicPipeline, ReceiptSampler

CONFIG_JUDGES = Path("configs/judges.yaml")


class ReceiptDetectorAPI:
    """High-level orchestration API usable from CLI or notebooks."""

    def __init__(self, judges_config_path: Path = CONFIG_JUDGES):
        self.judges_config_path = judges_config_path
        self.dataset_manager = DatasetManager()
        self.sampler = ReceiptSampler()
        self.forensic_pipeline = ForensicPipeline(output_dir="forensics/evidence")

        with open(judges_config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self._cfg = cfg
        self.judges = None
        self.voting_engine = None

    def _ensure_runtime(self) -> None:
        if self.judges is not None and self.voting_engine is not None:
            return
        self.judges = self._build_judges(self._cfg.get("judges", []))
        voting_cfg = self._cfg.get("voting", {})
        self.voting_engine = VotingEngine(
            strategy=voting_cfg.get("strategy", "majority_simple"),
            uncertain_threshold=voting_cfg.get("uncertain_threshold", 2),
        )

    def download_dataset(self) -> None:
        self.dataset_manager.download()
        self.dataset_manager.extract()

    def sample_receipts(self) -> list[dict[str, Any]]:
        labels = self.dataset_manager.load_labels(self.sampler.split)
        sample = self.sampler.sample(labels, dataset_manager=self.dataset_manager)
        self.sampler.save(sample)
        return sample

    def run_sample(self, save_dir: str | Path = "outputs/results") -> list[dict[str, Any]]:
        sample = self.sampler.load()
        return self.run_receipts(sample, save_dir=save_dir)

    def run_receipts(
        self,
        receipts: list[dict[str, Any]],
        save_dir: str | Path = "outputs/results",
    ) -> list[dict[str, Any]]:
        self._ensure_runtime()
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        outputs = []
        for receipt in receipts:
            result = self.run_single_receipt(
                receipt_id=receipt["id"],
                ground_truth=receipt.get("label"),
                split_hint=receipt.get("split", "train"),
                image_hint=receipt.get("image_path"),
                ocr_hint=receipt.get("ocr_txt_path"),
            )

            with open(out_dir / f"{receipt['id']}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            outputs.append(result)

        return outputs

    def run_single_receipt(
        self,
        receipt_id: str,
        *,
        ground_truth: str | None = None,
        split_hint: str | None = None,
        image_hint: str | None = None,
        ocr_hint: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_runtime()
        split = split_hint or "train"

        image_path = self._resolve_image_path(receipt_id, split, image_hint)
        if image_path is None:
            raise FileNotFoundError(f"Image not found for receipt '{receipt_id}'")

        ocr_path = self._resolve_ocr_path(receipt_id, split, ocr_hint)

        # Cache forensic outputs by mode to avoid duplicate analysis calls per receipt
        forensic_cache: dict[str, Any] = {}

        judge_results = []
        for judge in self.judges:
            mode = judge.forensic_mode
            if mode not in forensic_cache:
                forensic_cache[mode] = self.forensic_pipeline.analyze(
                    image_path=image_path,
                    ocr_txt_path=ocr_path,
                    mode=mode,
                )

            ctx = forensic_cache[mode]
            result = judge.judge(
                receipt_id=receipt_id,
                image_path=image_path,
                evidence_pack=ctx.evidence_pack,
                supporting_image_paths=ctx.supporting_image_paths,
            )
            judge_results.append(result)

        verdict = self.voting_engine.aggregate(judge_results)
        output = verdict.to_dict()
        if ground_truth is not None:
            output["ground_truth"] = ground_truth
        output["forensic_modes_used"] = sorted(forensic_cache.keys())
        output["forensics_output_dir"] = str(self.forensic_pipeline.output_dir / image_path.stem)
        return output

    def evaluate_results(self) -> dict[str, Any]:
        sample = self.sampler.load()
        ground_truth = {r["id"]: r["label"] for r in sample}
        evaluator = Evaluator()
        evaluator.load_results()
        return evaluator.summary(ground_truth)

    def forensic_report(self, receipt_id: str) -> dict[str, Any]:
        image_path, split = self._find_image_across_splits(receipt_id)
        if image_path is None:
            raise FileNotFoundError(f"Image not found for receipt '{receipt_id}'")
        ocr_path = self.dataset_manager.find_ocr_txt(receipt_id, split)

        return {
            "READING": self.forensic_pipeline.analyze(image_path, ocr_txt_path=ocr_path, mode="READING").evidence_pack,
            "GRAPHIC": self.forensic_pipeline.analyze(image_path, ocr_txt_path=ocr_path, mode="GRAPHIC").evidence_pack,
            "FULL": self.forensic_pipeline.analyze(image_path, ocr_txt_path=ocr_path, mode="FULL").evidence_pack,
        }

    def _build_judges(self, judges_cfg: list[dict[str, Any]]) -> list[Any]:
        judges = []
        for jc in judges_cfg:
            model = str(jc.get("model", ""))
            common = dict(
                judge_id=jc.get("id"),
                judge_name=jc.get("name"),
                persona_description=jc.get("persona", ""),
                temperature=float(jc.get("temperature", 0.3)),
                max_tokens=int(jc.get("max_tokens", 1024)),
                focus_skills=jc.get("focus_skills"),
                forensic_mode=jc.get("forensic_mode", "FULL"),
            )
            if "Qwen" in model:
                judges.append(QwenJudge(**common))
            else:
                judges.append(GLMJudge(**common))
        return judges

    def _resolve_image_path(self, receipt_id: str, split: str, image_hint: str | None) -> Path | None:
        if image_hint:
            p = Path(image_hint)
            if p.exists():
                return p
        return self.dataset_manager.find_image(receipt_id, split)

    def _resolve_ocr_path(self, receipt_id: str, split: str, ocr_hint: str | None) -> Path | None:
        if ocr_hint:
            p = Path(ocr_hint)
            if p.exists():
                return p
        return self.dataset_manager.find_ocr_txt(receipt_id, split)

    def _find_image_across_splits(self, receipt_id: str) -> tuple[Path | None, str | None]:
        for split in ["train", "val", "test"]:
            p = self.dataset_manager.find_image(receipt_id, split)
            if p is not None:
                return p, split
        return None, None


# -------------------- CLI commands --------------------

def cmd_download(_: argparse.Namespace) -> None:
    api = ReceiptDetectorAPI()
    api.download_dataset()
    print("[download] Done. Dataset extracted to data/raw/findit2/")


def cmd_sample(_: argparse.Namespace) -> None:
    api = ReceiptDetectorAPI()
    sample = api.sample_receipts()
    print(f"[sample] Selected {len(sample)} receipts and saved outputs/samples.json")


def cmd_run(_: argparse.Namespace) -> None:
    api = ReceiptDetectorAPI()
    outputs = api.run_sample()
    print(f"[run] Processed {len(outputs)} receipts. Results in outputs/results/")


def cmd_evaluate(_: argparse.Namespace) -> None:
    api = ReceiptDetectorAPI()
    summary = api.evaluate_results()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_demo(args: argparse.Namespace) -> None:
    api = ReceiptDetectorAPI()
    out = api.run_single_receipt(args.receipt_id)
    print(json.dumps(out, indent=2, ensure_ascii=False))


def cmd_forensic(args: argparse.Namespace) -> None:
    api = ReceiptDetectorAPI()
    rep = api.forensic_report(args.receipt_id)
    print(json.dumps(rep, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-Judge Fake Receipt Detector")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="Download and extract the Find-It-Again dataset")
    sub.add_parser("sample", help="Select 20 receipts (10 REAL + 10 FAKE) from configured split")
    sub.add_parser("run", help="Run 3 judges with always-on forensic evidence")
    sub.add_parser("evaluate", help="Compute evaluation metrics from outputs/results")

    demo_p = sub.add_parser("demo", help="Run judges on a single receipt")
    demo_p.add_argument("receipt_id", help="Receipt filename stem")

    forensic_p = sub.add_parser("forensic", help="Run forensic packs on a single receipt")
    forensic_p.add_argument("receipt_id", help="Receipt filename stem")

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
