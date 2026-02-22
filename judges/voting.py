"""Voting engine (majority-simple decision + uncertainty diagnostics)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

from .base_judge import JudgeResult


@dataclass
class FinalVerdict:
    receipt_id: str
    label: str
    tally: str
    vote_counts: Dict[str, int]
    avg_confidence: float
    strategy_used: str
    all_flags: List[str]
    judge_results: List[JudgeResult]
    verdict_uncertainty: float = 0.0
    judge_uncertainties: Dict[str, float] = field(default_factory=dict)
    weighted_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "receipt_id": self.receipt_id,
            "label": self.label,
            "tally": self.tally,
            "vote_counts": self.vote_counts,
            "avg_confidence": round(self.avg_confidence, 2),
            "verdict_uncertainty": round(self.verdict_uncertainty, 4),
            "strategy_used": self.strategy_used,
            "all_flags": self.all_flags,
            "judge_uncertainties": {k: round(v, 4) for k, v in self.judge_uncertainties.items()},
            "weighted_scores": {k: round(v, 4) for k, v in self.weighted_scores.items()},
            "judges": [jr.to_dict() for jr in self.judge_results],
        }


class VotingEngine:
    """Label is chosen by simple majority only; uncertainty diagnostics are preserved."""

    def __init__(self, strategy: str = "majority_simple", uncertain_threshold: int = 2):
        self.strategy = strategy
        self.uncertain_threshold = uncertain_threshold

    def aggregate(self, results: List[JudgeResult]) -> FinalVerdict:
        if not results:
            raise ValueError("No judge results provided to VotingEngine.")

        receipt_id = results[0].receipt_id
        n = len(results)
        vote_counts = dict(Counter(r.label for r in results))
        all_flags = list(dict.fromkeys(f for r in results for f in r.flags))
        avg_confidence = sum(r.confidence for r in results) / n

        judge_uncertainties = {r.judge_id: self._compute_uncertainty(r) for r in results}
        weights = {r.judge_id: max(0.1, 1.0 - judge_uncertainties[r.judge_id]) for r in results}

        weighted_scores: Dict[str, float] = {"FAKE": 0.0, "REAL": 0.0, "UNCERTAIN": 0.0}
        for r in results:
            weighted_scores[r.label] += weights[r.judge_id]

        label = self._majority_simple_vote(vote_counts)
        tally = f"{label} ({vote_counts.get(label, 0)}/{n})"

        verdict_uncertainty = self._compute_verdict_uncertainty_simple(
            results, judge_uncertainties, label
        )

        return FinalVerdict(
            receipt_id=receipt_id,
            label=label,
            tally=tally,
            vote_counts=vote_counts,
            avg_confidence=avg_confidence,
            strategy_used="majority_simple",
            all_flags=all_flags,
            judge_results=results,
            verdict_uncertainty=verdict_uncertainty,
            judge_uncertainties=judge_uncertainties,
            weighted_scores=weighted_scores,
        )

    @staticmethod
    def _majority_simple_vote(vote_counts: Dict[str, int]) -> str:
        fake_votes = vote_counts.get("FAKE", 0)
        real_votes = vote_counts.get("REAL", 0)
        uncertain_votes = vote_counts.get("UNCERTAIN", 0)

        top = max(fake_votes, real_votes, uncertain_votes)
        winners = []
        if fake_votes == top:
            winners.append("FAKE")
        if real_votes == top:
            winners.append("REAL")
        if uncertain_votes == top:
            winners.append("UNCERTAIN")

        if len(winners) == 1:
            return winners[0]
        return "UNCERTAIN"

    @staticmethod
    def _compute_uncertainty(result: JudgeResult) -> float:
        if result.label == "UNCERTAIN":
            return 1.0

        c = result.confidence / 100.0
        if c >= 0.75:
            u_conf = 1.0 - c
        else:
            u_conf = min(1.0, 0.25 + (0.75 - c))

        n_skills = max(len(result.skill_results), 1)
        n_uncertain = sum(1 for v in result.skill_results.values() if v == "uncertain")
        u_skill = n_uncertain / n_skills

        consistency_penalty = 0.0
        if result.label == "FAKE" and result.risk_level == "low":
            consistency_penalty = 0.12
        elif result.label == "REAL" and result.risk_level == "high":
            consistency_penalty = 0.12
        elif result.label == "UNCERTAIN" and result.confidence > 75.0:
            consistency_penalty = 0.08

        u = 0.55 * u_conf + 0.35 * u_skill + consistency_penalty
        return round(min(1.0, max(0.0, u)), 4)

    @staticmethod
    def _compute_verdict_uncertainty_simple(
        results: List[JudgeResult],
        judge_uncertainties: Dict[str, float],
        winner_label: str,
    ) -> float:
        n = len(results)
        if n == 0:
            return 1.0
        disagreements = sum(
            1 for r in results if r.label != winner_label and r.label != "UNCERTAIN"
        )
        u_spread = disagreements / n
        u_avg = sum(judge_uncertainties.values()) / max(len(judge_uncertainties), 1)
        return round(min(1.0, 0.5 * u_spread + 0.5 * u_avg), 4)
