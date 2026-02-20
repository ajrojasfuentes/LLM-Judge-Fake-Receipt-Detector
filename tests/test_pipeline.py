"""Tests for dataset loading and sampling logic."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from pipeline.sampler import ReceiptSampler


FAKE_LABELS = {
    **{f"real_{i:03d}": "REAL" for i in range(50)},
    **{f"fake_{i:03d}": "FAKE" for i in range(30)},
}


def test_sampler_produces_correct_counts():
    sampler = ReceiptSampler()
    # dataset_manager=None means image/ocr paths are not resolved (returns None)
    sample = sampler.sample(FAKE_LABELS)
    real_count = sum(1 for s in sample if s["label"] == "REAL")
    fake_count = sum(1 for s in sample if s["label"] == "FAKE")
    assert real_count == sampler.real_count
    assert fake_count == sampler.fake_count
    assert len(sample) == sampler.total_samples


def test_sampler_is_reproducible():
    sampler = ReceiptSampler()
    s1 = sampler.sample(FAKE_LABELS)
    s2 = sampler.sample(FAKE_LABELS)
    assert [r["id"] for r in s1] == [r["id"] for r in s2]


def test_sampler_raises_on_insufficient_real():
    sampler = ReceiptSampler()
    few_labels = {f"real_{i}": "REAL" for i in range(3)}  # only 3 REAL, need 10
    few_labels.update({f"fake_{i}": "FAKE" for i in range(20)})
    with pytest.raises(ValueError, match="Not enough REAL"):
        sampler.sample(few_labels)


def test_sampler_raises_on_insufficient_fake():
    sampler = ReceiptSampler()
    few_labels = {f"real_{i}": "REAL" for i in range(20)}
    few_labels.update({f"fake_{i}": "FAKE" for i in range(2)})  # only 2 FAKE, need 10
    with pytest.raises(ValueError, match="Not enough FAKE"):
        sampler.sample(few_labels)


def test_sampler_output_includes_new_fields():
    """Each sample entry must include id, label, split, image_path, ocr_txt_path."""
    sampler = ReceiptSampler()
    sample = sampler.sample(FAKE_LABELS)
    for entry in sample:
        assert "id" in entry
        assert "label" in entry
        assert "split" in entry
        assert "image_path" in entry
        assert "ocr_txt_path" in entry
        assert entry["label"] in ("REAL", "FAKE")


def test_sampler_with_dataset_manager_resolves_paths():
    """When a DatasetManager is passed, image_path should be populated (or None if not found)."""
    sampler = ReceiptSampler()
    mock_dm = MagicMock()
    mock_dm.find_image.return_value = Path("/tmp/fake_image.png")
    mock_dm.find_ocr_txt.return_value = Path("/tmp/fake_ocr.txt")

    sample = sampler.sample(FAKE_LABELS, dataset_manager=mock_dm)
    for entry in sample:
        assert entry["image_path"] == "/tmp/fake_image.png"
        assert entry["ocr_txt_path"] == "/tmp/fake_ocr.txt"


def test_dataset_manager_loads_labels_from_csv():
    """DatasetManager must load correct label counts from the pre-collected CSVs."""
    from pipeline.dataset import DatasetManager
    dm = DatasetManager()

    # Train split: 577 total, 94 FAKE, 483 REAL (from paper + our CSV)
    train_labels = dm.load_labels("train")
    assert len(train_labels) == 577
    assert sum(1 for v in train_labels.values() if v == "FAKE") == 94
    assert sum(1 for v in train_labels.values() if v == "REAL") == 483


def test_dataset_manager_val_split():
    from pipeline.dataset import DatasetManager
    dm = DatasetManager()
    val_labels = dm.load_labels("val")
    assert len(val_labels) == 193
    assert sum(1 for v in val_labels.values() if v == "FAKE") == 34


def test_dataset_manager_test_split():
    from pipeline.dataset import DatasetManager
    dm = DatasetManager()
    test_labels = dm.load_labels("test")
    assert len(test_labels) == 218
    assert sum(1 for v in test_labels.values() if v == "FAKE") == 35


def test_dataset_manager_invalid_split_raises():
    from pipeline.dataset import DatasetManager
    dm = DatasetManager()
    with pytest.raises(ValueError, match="Invalid split"):
        dm.load_labels("invalid_split")
