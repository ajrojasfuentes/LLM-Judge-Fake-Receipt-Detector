"""
ReceiptSampler: Reproducibly selects 10 REAL + 10 FAKE receipts from a dataset split.

Sampling strategy:
  - Draws from the train split by default (94 forged, 483 real available).
  - Fixed seed (42) guarantees reproducibility across runs.
  - Includes image path and OCR txt path in the sample output for the pipeline.
  - Records dataset imbalance metadata for transparency in evaluation reports.

Dataset imbalance note:
  The Find-It-Again dataset is heavily imbalanced: ~16.5% forged (163/988).
  The sampler enforces equal 10+10 representation for unbiased judge evaluation,
  but the imbalance context is preserved in the sample metadata.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).parent.parent / "configs" / "sampling.yaml"
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "samples.json"


class ReceiptSampler:
    """
    Selects a stratified random sample of receipts.

    Usage:
        dm = DatasetManager()
        sampler = ReceiptSampler()
        labels = dm.load_labels()                  # {stem: "REAL"|"FAKE"}
        sample = sampler.sample(labels, dm)         # list of sample dicts
        sampler.save(sample)
        sample = sampler.load()                     # reload later
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        s_cfg = cfg["sampling"]
        ds_cfg = cfg["dataset"]

        self.random_seed: int = s_cfg["random_seed"]
        self.total_samples: int = s_cfg["total_samples"]
        self.real_count: int = s_cfg["real_count"]
        self.fake_count: int = s_cfg["fake_count"]
        self.split: str = s_cfg.get("split", "train")
        self.output_file: Path = Path(s_cfg["output_file"])

        # Split statistics for metadata
        self.splits_cfg: dict = ds_cfg.get("splits", {})

    def sample(
        self,
        labels: dict[str, str],
        dataset_manager: Any | None = None,
    ) -> list[dict]:
        """
        Randomly select real_count REAL + fake_count FAKE receipts.

        Args:
            labels: dict mapping filename stem → "REAL" | "FAKE"
            dataset_manager: optional DatasetManager instance. When provided,
                             image_path and ocr_txt_path are resolved and
                             included in each sample dict.

        Returns:
            List of dicts with keys:
                - id:           filename stem (e.g. "X00016469622")
                - label:        "REAL" or "FAKE"
                - split:        which split this came from
                - image_path:   absolute path to image (str) or None
                - ocr_txt_path: absolute path to OCR .txt (str) or None
        """
        rng = random.Random(self.random_seed)

        real_ids = sorted(k for k, v in labels.items() if v == "REAL")
        fake_ids = sorted(k for k, v in labels.items() if v == "FAKE")

        if len(real_ids) < self.real_count:
            raise ValueError(
                f"Not enough REAL receipts in '{self.split}' split: "
                f"need {self.real_count}, found {len(real_ids)}"
            )
        if len(fake_ids) < self.fake_count:
            raise ValueError(
                f"Not enough FAKE receipts in '{self.split}' split: "
                f"need {self.fake_count}, found {len(fake_ids)}"
            )

        selected_real = rng.sample(real_ids, self.real_count)
        selected_fake = rng.sample(fake_ids, self.fake_count)

        sample = []
        for stem in selected_real:
            entry = self._build_entry(stem, "REAL", dataset_manager)
            sample.append(entry)
        for stem in selected_fake:
            entry = self._build_entry(stem, "FAKE", dataset_manager)
            sample.append(entry)

        # Shuffle combined list reproducibly
        rng.shuffle(sample)
        return sample

    def save(self, sample: list[dict]) -> Path:
        """Persist the selected sample to outputs/samples.json."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        split_cfg = self.splits_cfg.get(self.split, {})
        meta = {
            "random_seed": self.random_seed,
            "split": self.split,
            "total": len(sample),
            "real_count": sum(1 for s in sample if s["label"] == "REAL"),
            "fake_count": sum(1 for s in sample if s["label"] == "FAKE"),
            # Dataset context for evaluation transparency
            "dataset_info": {
                "split_total": split_cfg.get("total"),
                "split_forged": split_cfg.get("forged"),
                "split_real": split_cfg.get("real"),
                "class_imbalance_note": (
                    "Full dataset: 163/988 forged (~16.5%). "
                    "Sample uses equal 10+10 for unbiased judge evaluation."
                ),
            },
            "receipts": sample,
        }

        with open(self.output_file, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[sampler] Sample saved to {self.output_file}")
        return self.output_file

    def load(self) -> list[dict]:
        """Load a previously saved sample."""
        if not self.output_file.exists():
            raise FileNotFoundError(
                f"No sample found at {self.output_file}. Run sample() first."
            )
        with open(self.output_file) as f:
            meta = json.load(f)
        return meta["receipts"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_entry(
        self,
        stem: str,
        label: str,
        dataset_manager: Any | None,
    ) -> dict:
        """Build a sample entry dict, resolving paths if dm is available."""
        entry: dict = {
            "id": stem,
            "label": label,
            "split": self.split,
            "image_path": None,
            "ocr_txt_path": None,
        }

        if dataset_manager is not None:
            img = dataset_manager.find_image(stem, self.split)
            txt = dataset_manager.find_ocr_txt(stem, self.split)
            entry["image_path"] = str(img) if img else None
            entry["ocr_txt_path"] = str(txt) if txt else None

        return entry
