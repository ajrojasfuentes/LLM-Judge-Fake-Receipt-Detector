"""
ReceiptSampler: Reproducibly selects 10 REAL + 10 FAKE receipts from the dataset.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "sampling.yaml"
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "samples.json"


class ReceiptSampler:
    """
    Selects a stratified random sample of receipts.

    Usage:
        sampler = ReceiptSampler()
        sample = sampler.sample(labels)  # labels: {stem: "REAL"|"FAKE"}
        sampler.save(sample)
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        s_cfg = cfg["sampling"]
        self.random_seed: int = s_cfg["random_seed"]
        self.total_samples: int = s_cfg["total_samples"]
        self.real_count: int = s_cfg["real_count"]
        self.fake_count: int = s_cfg["fake_count"]
        self.output_file: Path = Path(s_cfg["output_file"])

    def sample(self, labels: dict[str, str]) -> list[dict]:
        """
        Randomly select real_count REAL + fake_count FAKE receipts.

        Args:
            labels: dict mapping filename stem → "REAL" | "FAKE"
        Returns:
            List of dicts with keys: id, label
        """
        rng = random.Random(self.random_seed)

        real_ids = sorted(k for k, v in labels.items() if v == "REAL")
        fake_ids = sorted(k for k, v in labels.items() if v == "FAKE")

        if len(real_ids) < self.real_count:
            raise ValueError(
                f"Not enough REAL receipts: need {self.real_count}, found {len(real_ids)}"
            )
        if len(fake_ids) < self.fake_count:
            raise ValueError(
                f"Not enough FAKE receipts: need {self.fake_count}, found {len(fake_ids)}"
            )

        selected_real = rng.sample(real_ids, self.real_count)
        selected_fake = rng.sample(fake_ids, self.fake_count)

        sample = (
            [{"id": s, "label": "REAL"} for s in selected_real]
            + [{"id": s, "label": "FAKE"} for s in selected_fake]
        )

        # Shuffle the combined list (reproducible)
        rng.shuffle(sample)
        return sample

    def save(self, sample: list[dict]) -> Path:
        """Persist the selected sample to outputs/samples.json."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "random_seed": self.random_seed,
            "total": len(sample),
            "real_count": sum(1 for s in sample if s["label"] == "REAL"),
            "fake_count": sum(1 for s in sample if s["label"] == "FAKE"),
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
