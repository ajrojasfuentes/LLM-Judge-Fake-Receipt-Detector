"""
DatasetManager: Handles downloading, extracting, and loading the Find-It-Again dataset.

Dataset: https://l3i-share.univ-lr.fr/2023Finditagain/index.html
Download: https://l3i-share.univ-lr.fr/2023Finditagain/findit2.zip
"""

from __future__ import annotations

import csv
import re
import shutil
import zipfile
from pathlib import Path

import requests
import yaml


CONFIG_PATH = Path(__file__).parent.parent / "configs" / "sampling.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class DatasetManager:
    """
    Manages the Find-It-Again receipt dataset.

    Usage:
        dm = DatasetManager()
        dm.download()       # downloads findit2.zip if not present
        dm.extract()        # extracts to data/raw/
        labels = dm.load_labels()  # returns {filename: "REAL"|"FAKE"}
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        ds_cfg = cfg["dataset"]
        self.download_url: str = ds_cfg["download_url"]
        self.raw_dir: Path = Path(ds_cfg["raw_dir"])
        self.samples_dir: Path = Path(ds_cfg["samples_dir"])
        self.labels_file: Path = Path(ds_cfg["labels_file"])

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
        response = requests.get(self.download_url, stream=True, timeout=120)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"[dataset] Downloaded to {zip_path} ({zip_path.stat().st_size // 1024} KB)")
        return zip_path

    def extract(self, force: bool = False) -> Path:
        """Extract the ZIP archive into data/raw/."""
        zip_path = self.raw_dir / "findit2.zip"
        extracted_marker = self.raw_dir / ".extracted"

        if extracted_marker.exists() and not force:
            print(f"[dataset] Already extracted. Skipping.")
            return self.raw_dir

        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP not found at {zip_path}. Run download() first.")

        print(f"[dataset] Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.raw_dir)

        extracted_marker.touch()
        print(f"[dataset] Extracted to {self.raw_dir}")
        return self.raw_dir

    # ------------------------------------------------------------------
    # Label loading
    # ------------------------------------------------------------------

    def load_labels(self) -> dict[str, str]:
        """
        Load ground-truth labels from the dataset split files.
        Returns a dict mapping filename (stem) → "REAL" or "FAKE".

        The Find-It-Again dataset uses text files listing image IDs with labels.
        We parse those files and also generate data/labels.csv for convenience.
        """
        if self.labels_file.exists():
            return self._read_labels_csv()

        labels = self._parse_dataset_splits()
        self._write_labels_csv(labels)
        return labels

    def _parse_dataset_splits(self) -> dict[str, str]:
        """
        Parse the dataset's own split/label files.
        The Find-It-Again dataset ships with files like:
          - train.txt / val.txt / test.txt  with columns: filename label
          - Or a single gt.txt / ground_truth.txt
        We try common patterns and fall back to inferring from directory names.
        """
        labels: dict[str, str] = {}

        # Pattern 1: files with two columns (filename, label) or (filename, 0/1)
        for candidate in self.raw_dir.rglob("*.txt"):
            found = self._try_parse_label_file(candidate)
            if found:
                labels.update(found)

        # Pattern 2: infer from subdirectory names (fake/ and real/ folders)
        if not labels:
            for img_path in self.raw_dir.rglob("*"):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    parts = [p.lower() for p in img_path.parts]
                    if any("fake" in p for p in parts):
                        labels[img_path.stem] = "FAKE"
                    elif any("real" in p or "original" in p or "authentic" in p for p in parts):
                        labels[img_path.stem] = "REAL"

        if not labels:
            raise RuntimeError(
                "Could not parse labels from dataset. "
                "Please inspect data/raw/ and update _parse_dataset_splits()."
            )

        print(f"[dataset] Loaded {len(labels)} labels "
              f"({sum(1 for v in labels.values() if v=='REAL')} REAL, "
              f"{sum(1 for v in labels.values() if v=='FAKE')} FAKE)")
        return labels

    @staticmethod
    def _try_parse_label_file(path: Path) -> dict[str, str] | None:
        """Try to parse a text file as a label list. Returns None if not parseable."""
        results: dict[str, str] = {}
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                filename, raw_label = parts[0], parts[1].upper()

                if raw_label in {"FAKE", "FORGED", "1"}:
                    label = "FAKE"
                elif raw_label in {"REAL", "AUTHENTIC", "ORIGINAL", "0"}:
                    label = "REAL"
                else:
                    continue

                stem = Path(filename).stem
                results[stem] = label
        except Exception:
            return None
        return results if results else None

    def _write_labels_csv(self, labels: dict[str, str]) -> None:
        self.labels_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.labels_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            for stem, label in sorted(labels.items()):
                writer.writerow([stem, label])
        print(f"[dataset] Labels written to {self.labels_file}")

    def _read_labels_csv(self) -> dict[str, str]:
        labels: dict[str, str] = {}
        with open(self.labels_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["filename"]] = row["label"]
        return labels

    # ------------------------------------------------------------------
    # Image lookup
    # ------------------------------------------------------------------

    def find_image(self, stem: str) -> Path | None:
        """Find the image file for a given filename stem."""
        for ext in IMAGE_EXTENSIONS:
            for candidate in self.raw_dir.rglob(f"{stem}{ext}"):
                return candidate
        return None

    def all_images(self) -> list[Path]:
        """Return all image paths in the raw dataset directory."""
        return [
            p for p in self.raw_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
