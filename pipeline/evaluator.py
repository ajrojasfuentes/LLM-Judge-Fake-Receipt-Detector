"""
Evaluator: Computes metrics and identifies disagreement cases from judge outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict


RESULTS_DIR = Path(__file__).parent.parent / "outputs" / "results"


class Evaluator:
    """
    Loads FinalVerdict JSON files and computes evaluation metrics.

    Usage:
        ev = Evaluator()
        ev.load_results()
        print(ev.accuracy())
        print(ev.confusion_matrix())
        cases = ev.disagreement_cases(n=3)
    """

    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = results_dir
        self._results: list[dict] = []

    def load_results(self) -> None:
        """Load all result JSON files from outputs/results/."""
        self._results = []
        for path in sorted(self.results_dir.glob("*.json")):
            with open(path) as f:
                self._results.append(json.load(f))
        print(f"[evaluator] Loaded {len(self._results)} result files.")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def accuracy(self, ground_truth: dict[str, str] | None = None) -> float:
        """
        Compute accuracy of final verdicts against ground-truth labels.
        If ground_truth is None, looks for 'ground_truth' field in each result dict.
        """
        correct = 0
        total = 0
        for r in self._results:
            gt = (ground_truth or {}).get(r["receipt_id"]) or r.get("ground_truth")
            if gt is None:
                continue
            predicted = r["label"]
            # UNCERTAIN counts as wrong
            if predicted == gt:
                correct += 1
            total += 1

        if total == 0:
            return 0.0
        return correct / total

    def confusion_matrix(self, ground_truth: dict[str, str] | None = None) -> dict:
        """
        Returns a confusion matrix dict:
        {
          "TP": int,  # predicted FAKE, gt FAKE
          "TN": int,  # predicted REAL, gt REAL
          "FP": int,  # predicted FAKE, gt REAL
          "FN": int,  # predicted REAL, gt FAKE
          "UNCERTAIN": int,
        }
        """
        cm = defaultdict(int)
        for r in self._results:
            gt = (ground_truth or {}).get(r["receipt_id"]) or r.get("ground_truth")
            if gt is None:
                continue
            pred = r["label"]
            if pred == "UNCERTAIN":
                cm["UNCERTAIN"] += 1
            elif pred == "FAKE" and gt == "FAKE":
                cm["TP"] += 1
            elif pred == "REAL" and gt == "REAL":
                cm["TN"] += 1
            elif pred == "FAKE" and gt == "REAL":
                cm["FP"] += 1
            elif pred == "REAL" and gt == "FAKE":
                cm["FN"] += 1
        return dict(cm)

    def disagreement_cases(self, n: int = 3) -> list[dict]:
        """
        Return up to n cases where judges disagreed (not all voting the same).
        """
        cases = []
        for r in self._results:
            judges = r.get("judges", [])
            if len(judges) < 2:
                continue
            vote_set = set(j["label"] for j in judges)
            if len(vote_set) > 1:  # judges disagree
                cases.append(r)
        return cases[:n]

    def summary(self, ground_truth: dict[str, str] | None = None) -> dict:
        """Return a full evaluation summary dict."""
        cm = self.confusion_matrix(ground_truth)
        acc = self.accuracy(ground_truth)

        total = sum(cm.get(k, 0) for k in ["TP", "TN", "FP", "FN", "UNCERTAIN"])
        fake_total = cm.get("TP", 0) + cm.get("FN", 0)
        real_total = cm.get("TN", 0) + cm.get("FP", 0)

        precision = cm.get("TP", 0) / max(cm.get("TP", 0) + cm.get("FP", 0), 1)
        recall = cm.get("TP", 0) / max(fake_total, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-9)

        return {
            "total_evaluated": total,
            "accuracy": round(acc, 4),
            "precision_fake": round(precision, 4),
            "recall_fake": round(recall, 4),
            "f1_fake": round(f1, 4),
            "confusion_matrix": cm,
            "disagreement_count": len(self.disagreement_cases(n=len(self._results))),
        }
