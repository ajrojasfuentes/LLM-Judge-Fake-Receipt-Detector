"""
VotingEngine: Aggregates 3 JudgeResult objects into a single FinalVerdict.

Strategy 1 (default): Majority vote
  - 2+ FAKE  → FAKE
  - 2+ REAL  → REAL
  - otherwise → UNCERTAIN

Strategy 2 (extension): Confidence-weighted vote
  - Sum confidence scores per label; highest wins (with a minimum threshold).
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

from .base_judge import JudgeResult


@dataclass
class FinalVerdict:
    """The aggregated decision from all judges."""

    receipt_id: str
    label: str                  # "REAL" | "FAKE" | "UNCERTAIN"
    tally: str                  # e.g. "FAKE (2/3)"
    vote_counts: dict[str, int] # {"FAKE": 2, "REAL": 1, "UNCERTAIN": 0}
    avg_confidence: float
    strategy_used: str          # "majority" | "confidence_weighted"
    all_flags: list[str]        # union of all judge flags
    judge_results: list[JudgeResult]

    def to_dict(self) -> dict:
        return {
            "receipt_id": self.receipt_id,
            "label": self.label,
            "tally": self.tally,
            "vote_counts": self.vote_counts,
            "avg_confidence": round(self.avg_confidence, 2),
            "strategy_used": self.strategy_used,
            "all_flags": self.all_flags,
            "judges": [jr.to_dict() for jr in self.judge_results],
        }


class VotingEngine:
    """
    Combines multiple JudgeResult objects into a FinalVerdict.

    Args:
        strategy: "majority" (default) or "confidence_weighted"
        uncertain_threshold: if >= N votes are UNCERTAIN, force UNCERTAIN
    """

    def __init__(self, strategy: str = "majority", uncertain_threshold: int = 2):
        self.strategy = strategy
        self.uncertain_threshold = uncertain_threshold

    def aggregate(self, results: list[JudgeResult]) -> FinalVerdict:
        if not results:
            raise ValueError("No judge results provided to VotingEngine.")

        receipt_id = results[0].receipt_id
        vote_counts = Counter(r.label for r in results)
        n = len(results)

        # Collect all unique flags across judges
        all_flags = list(dict.fromkeys(f for r in results for f in r.flags))

        avg_confidence = sum(r.confidence for r in results) / n

        if self.strategy == "majority":
            label, strategy_used = self._majority_vote(vote_counts, n)
        elif self.strategy == "confidence_weighted":
            label, strategy_used = self._confidence_weighted_vote(results)
        else:
            label, strategy_used = self._majority_vote(vote_counts, n)

        winning_count = vote_counts.get(label, 0)
        tally = f"{label} ({winning_count}/{n})"

        return FinalVerdict(
            receipt_id=receipt_id,
            label=label,
            tally=tally,
            vote_counts=dict(vote_counts),
            avg_confidence=avg_confidence,
            strategy_used=strategy_used,
            all_flags=all_flags,
            judge_results=results,
        )

    # ------------------------------------------------------------------
    # Voting strategies
    # ------------------------------------------------------------------

    def _majority_vote(self, vote_counts: Counter, n: int) -> tuple[str, str]:
        # If too many UNCERTAIN votes, return UNCERTAIN
        if vote_counts.get("UNCERTAIN", 0) >= self.uncertain_threshold:
            return "UNCERTAIN", "majority (uncertain threshold)"

        fake_votes = vote_counts.get("FAKE", 0)
        real_votes = vote_counts.get("REAL", 0)
        majority = n // 2 + 1  # e.g., 2 out of 3

        if fake_votes >= majority:
            return "FAKE", "majority"
        if real_votes >= majority:
            return "REAL", "majority"
        return "UNCERTAIN", "majority (no majority)"

    def _confidence_weighted_vote(self, results: list[JudgeResult]) -> tuple[str, str]:
        """
        Weighted vote: sum confidence scores per label; highest total wins.
        Falls back to UNCERTAIN if the winner's total confidence is below 50.
        """
        weighted: dict[str, float] = {"FAKE": 0.0, "REAL": 0.0, "UNCERTAIN": 0.0}
        for r in results:
            weighted[r.label] += r.confidence

        # Exclude UNCERTAIN from winning unless it's the only option
        candidates = {k: v for k, v in weighted.items() if k != "UNCERTAIN"}
        if not candidates:
            return "UNCERTAIN", "confidence_weighted (all uncertain)"

        winner = max(candidates, key=lambda k: candidates[k])
        total_confidence = candidates[winner]

        # If winner's combined confidence is too low, fall back to UNCERTAIN
        if total_confidence < 50.0:
            return "UNCERTAIN", "confidence_weighted (low confidence)"

        return winner, "confidence_weighted"
