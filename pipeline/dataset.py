"""
DatasetManager: Handles downloading, extracting, and loading the Find-It-Again dataset.

Dataset:  https://l3i-share.univ-lr.fr/2023Finditagain/index.html
Download: http://l3i-share.univ-lr.fr/2023Finditagain/findit2.zip

=== Dataset Structure (after extraction) ===
data/raw/findit2/
    train/          ← PNG images + OCR .txt files for training split
    val/            ← PNG images + OCR .txt files for validation split
    test/           ← PNG images + OCR .txt files for test split
    train.txt       ← CSV label file: image, digital annotation, handwritten annotation,
                        forged, forgery annotations
    val.txt         ← same format
    test.txt        ← same format

=== Label Format ===
The split files (train.txt, val.txt, test.txt) are CSV files where:
  - "forged" column: "True" (FAKE) or "False" (REAL)
  - "digital annotation" / "handwritten annotation": True/False flags for
    non-fraudulent marks — these are NOT forgeries (important hard negatives).
  - "forgery annotations": JSON with region-level forgery details (for forged=True rows).

=== Pre-collected Metadata ===
data/dataset/findit2/
    train_data.csv  ← enriched CSV with image metadata (width, height, blur, etc.)
    val_data.csv
    test_data.csv

=== Key Statistics ===
Total: 988 receipts | Forged: 163 (~16.5%) | Real: 825 (~83.5%)
Dominant forgery type: CPI - Copy-Paste Inside (~77.6% of all modifications)
Most targeted entity: Total/Payment (~51.4% of all modifications)
"""

from __future__ import annotations

import ast
import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import requests
import yaml


CONFIG_PATH = Path(__file__).parent.parent / "configs" / "sampling.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class DatasetManager:
    """
    Manages the Find-It-Again receipt dataset.

    Primary usage:
        dm = DatasetManager()
        dm.download()                     # Download findit2.zip if not present
        dm.extract()                      # Extract to data/raw/findit2/
        labels = dm.load_labels()         # {stem: "REAL"|"FAKE"} from train split
        labels = dm.load_labels("test")   # Load a specific split

    Advanced:
        info = dm.load_split_info("train")  # Full dict with metadata for each image
        img  = dm.find_image("X000...", "train")   # Locate image in split directory
        txt  = dm.find_ocr_txt("X000...", "train") # Locate paired OCR text
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        ds_cfg = cfg["dataset"]
        self.download_url: str = ds_cfg["download_url"]
        self.raw_dir: Path = Path(ds_cfg["raw_dir"])
        self.findit2_dir: Path = Path(ds_cfg["findit2_dir"])
        self.metadata_dir: Path = Path(ds_cfg["metadata_dir"])
        self.samples_dir: Path = Path(ds_cfg["samples_dir"])
        self.labels_file: Path = Path(ds_cfg["labels_file"])
        self.splits_cfg: dict = ds_cfg.get("splits", {})

        s_cfg = cfg["sampling"]
        self.default_split: str = s_cfg.get("split", "train")

    # ------------------------------------------------------------------
    # Download & extract
    # ------------------------------------------------------------------

    def download(self, force: bool = False) -> Path:
        """Download findit2.zip if not already present."""
        zip_path = self.raw_dir / "findit2.zip"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        if zip_path.exists() and not force:
            print(f"[dataset] ZIP already exists at {zip_path}. Skipping download.")
            return zip_path

        print(f"[dataset] Downloading dataset from {self.download_url} ...")
        response = requests.get(self.download_url, stream=True, timeout=300)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"[dataset] Downloaded to {zip_path} ({size_mb:.1f} MB)")
        return zip_path

    def extract(self, force: bool = False) -> Path:
        """Extract the ZIP archive into data/raw/, producing data/raw/findit2/."""
        zip_path = self.raw_dir / "findit2.zip"
        extracted_marker = self.raw_dir / ".extracted"

        if extracted_marker.exists() and not force:
            print(f"[dataset] Already extracted at {self.findit2_dir}. Skipping.")
            return self.findit2_dir

        if not zip_path.exists():
            raise FileNotFoundError(
                f"ZIP not found at {zip_path}. Run download() first."
            )

        print(f"[dataset] Extracting {zip_path} → {self.raw_dir} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.raw_dir)

        extracted_marker.touch()
        print(f"[dataset] Extracted. Dataset root: {self.findit2_dir}")
        return self.findit2_dir

    # ------------------------------------------------------------------
    # Label loading (primary interface)
    # ------------------------------------------------------------------

    def load_labels(self, split: str | None = None) -> dict[str, str]:
        """
        Load ground-truth labels for a given split.

        Returns a dict mapping filename stem → "REAL" | "FAKE".

        Priority order for reading labels:
          1. Pre-collected metadata CSV (data/dataset/findit2/<split>_data.csv)
          2. Raw split file from extracted dataset (data/raw/findit2/<split>.txt)

        Args:
            split: "train", "val", or "test". Defaults to sampling config split.
        """
        split = split or self.default_split
        self._validate_split(split)

        # Try pre-collected CSV first (already processed and enriched)
        meta_csv = self.metadata_dir / self.splits_cfg[split]["metadata_csv"]
        if meta_csv.exists():
            labels = self._load_labels_from_metadata_csv(meta_csv)
            self._print_label_summary(labels, split, source=str(meta_csv))
            return labels

        # Fall back to raw split txt file
        split_txt = self.findit2_dir / self.splits_cfg[split]["txt"]
        if split_txt.exists():
            labels = self._load_labels_from_split_txt(split_txt)
            self._print_label_summary(labels, split, source=str(split_txt))
            return labels

        raise FileNotFoundError(
            f"Cannot find label file for split '{split}'. "
            f"Looked for:\n  {meta_csv}\n  {split_txt}\n"
            f"Run `dm.download()` and `dm.extract()` first, or ensure "
            f"data/dataset/findit2/{split}_data.csv exists."
        )

    def load_all_splits(self) -> dict[str, dict[str, str]]:
        """
        Load labels for all three splits.

        Returns:
            {"train": {...}, "val": {...}, "test": {...}}
        """
        return {split: self.load_labels(split) for split in ["train", "val", "test"]}

    def load_split_info(self, split: str | None = None) -> list[dict[str, Any]]:
        """
        Load full per-image metadata for a split.

        Returns a list of dicts with all CSV columns, plus:
          - "label": "REAL" or "FAKE"
          - "split": split name

        Args:
            split: "train", "val", or "test".
        """
        split = split or self.default_split
        self._validate_split(split)

        meta_csv = self.metadata_dir / self.splits_cfg[split]["metadata_csv"]
        if not meta_csv.exists():
            raise FileNotFoundError(
                f"Metadata CSV not found: {meta_csv}. "
                "Ensure data/dataset/findit2/ contains the pre-collected CSVs."
            )

        rows = []
        with open(meta_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                forged_raw = row.get("forged", "False").strip()
                label = "FAKE" if forged_raw in ("True", "1", "true") else "REAL"
                row["label"] = label
                row["split"] = split
                rows.append(dict(row))

        return rows

    # ------------------------------------------------------------------
    # Image and OCR text lookup
    # ------------------------------------------------------------------

    def find_image(self, stem: str, split: str | None = None) -> Path | None:
        """
        Find the PNG image for a given filename stem in a split directory.

        Searches in order:
          1. data/raw/findit2/<split>/<stem>.png  (extracted dataset)
          2. data/raw/findit2/<split>/<stem>.*    (other extensions)
          3. data/raw/**/<stem>.*                 (fallback: anywhere in raw/)

        Args:
            stem: Filename without extension (e.g. "X00016469622").
            split: "train", "val", or "test". Defaults to sampling config split.
        """
        split = split or self.default_split

        # Primary: look in the split subdirectory
        if split in self.splits_cfg:
            split_dir = self.findit2_dir / self.splits_cfg[split]["images_subdir"]
            for ext in IMAGE_EXTENSIONS:
                candidate = split_dir / f"{stem}{ext}"
                if candidate.exists():
                    return candidate

        # Fallback: search anywhere in raw_dir
        for ext in IMAGE_EXTENSIONS:
            for candidate in self.raw_dir.rglob(f"{stem}{ext}"):
                return candidate

        return None

    def find_ocr_txt(self, stem: str, split: str | None = None) -> Path | None:
        """
        Find the paired OCR .txt file for a given image stem.

        In the Find-It-Again dataset, each PNG has a paired .txt transcription
        in the same directory. For forged receipts, the transcription has been
        updated to match the forged content.

        Args:
            stem: Filename without extension.
            split: "train", "val", or "test".
        """
        split = split or self.default_split

        if split in self.splits_cfg:
            split_dir = self.findit2_dir / self.splits_cfg[split]["images_subdir"]
            candidate = split_dir / f"{stem}.txt"
            if candidate.exists():
                return candidate

        # Fallback
        for candidate in self.raw_dir.rglob(f"{stem}.txt"):
            return candidate

        return None

    def all_images(self, split: str | None = None) -> list[Path]:
        """
        Return all image paths for a given split (or all images if split is None).
        """
        if split is not None:
            self._validate_split(split)
            split_dir = self.findit2_dir / self.splits_cfg[split]["images_subdir"]
            return sorted(
                p for p in split_dir.rglob("*")
                if p.suffix.lower() in IMAGE_EXTENSIONS
            )

        return sorted(
            p for p in self.raw_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_labels_from_metadata_csv(self, csv_path: Path) -> dict[str, str]:
        """
        Parse labels from a pre-collected metadata CSV.

        The 'forged' column contains Python boolean strings: "True" or "False".
        """
        labels: dict[str, str] = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image = row.get("image", "").strip()
                if not image:
                    continue
                stem = Path(image).stem
                forged_raw = row.get("forged", "False").strip()
                # Handle both Python bool strings and 0/1
                if forged_raw in ("True", "1", "true", "TRUE"):
                    labels[stem] = "FAKE"
                else:
                    labels[stem] = "REAL"
        return labels

    def _load_labels_from_split_txt(self, txt_path: Path) -> dict[str, str]:
        """
        Parse labels from a raw split .txt file (CSV-formatted).

        Format: image, digital annotation, handwritten annotation, forged, forgery annotations
        The 'forged' column: True / False
        """
        labels: dict[str, str] = {}
        with open(txt_path, newline="", encoding="utf-8") as f:
            # These files can have complex quoted JSON in the last column
            reader = csv.DictReader(f, quotechar='"')
            for row in reader:
                image = row.get("image", "").strip()
                if not image:
                    continue
                stem = Path(image).stem
                forged_raw = row.get("forged", "False").strip()
                if forged_raw in ("True", "1", "true", "TRUE"):
                    labels[stem] = "FAKE"
                else:
                    labels[stem] = "REAL"
        return labels

    def _validate_split(self, split: str) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Invalid split '{split}'. Must be one of: train, val, test"
            )

    @staticmethod
    def _print_label_summary(
        labels: dict[str, str], split: str, source: str
    ) -> None:
        real = sum(1 for v in labels.values() if v == "REAL")
        fake = sum(1 for v in labels.values() if v == "FAKE")
        print(
            f"[dataset] Loaded {len(labels)} labels from {split} split "
            f"({real} REAL, {fake} FAKE) ← {source}"
        )
