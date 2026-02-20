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
