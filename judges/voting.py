"""
VotingEngine: Aggregates 3 JudgeResult objects into a single FinalVerdict.

=== Dynamic Uncertainty-Weighted Voting ===

Each judge's vote weight is computed dynamically from its *uncertainty score*:
    weight = max(0.1, 1.0 - uncertainty)

Uncertainty is calculated from three components (more severe than raw confidence):
  1. Confidence component (55%): nonlinear penalty — medium/low confidence is
     penalised more harshly than pure (100 - conf).
  2. Skill component (35%): fraction of skill_results that are "uncertain".
  3. Consistency penalty (up to 12%): penalises judges that claim high risk
     but also very high confidence (overconfident), or low confidence on
     clear-cut cases (overcautious).

Vote weights replace the binary 0/1 majority count:
    FAKE_score = sum of weights for all judges that voted FAKE
    REAL_score = sum of weights for all judges that voted REAL
    Winner = label with > 40% of total weight (avoids a single high-weight
    judge dominating when the other two disagree but are uncertain).

Verdict uncertainty is then computed from:
  1. Vote spread (40%): how decisive was the weighted vote (winner_weight / total)?
  2. Average judge uncertainty (35%): individual certainty of all judges.
  3. Dissent weight (25%): weight of judges who dissented from the winner.

This gives a continuous [0, 1] uncertainty estimate for the final verdict
that reflects both the judges' confidence and their agreement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import Dict, List

from .base_judge import JudgeResult


@dataclass
class FinalVerdict:
    """The aggregated decision from all judges."""

    receipt_id: str
    label: str                          # "REAL" | "FAKE" | "UNCERTAIN"
    tally: str                          # e.g. "FAKE (w=1.52/2.30)"
    vote_counts: Dict[str, int]         # raw label counts {"FAKE": 2, "REAL": 1}
    avg_confidence: float               # simple mean of stated confidences
    strategy_used: str                  # strategy identifier
    all_flags: List[str]                # union of all judge flags
    judge_results: List[JudgeResult]

    # Uncertainty information
    verdict_uncertainty: float = 0.0    # [0, 1] — 0=certain, 1=fully uncertain
    judge_uncertainties: Dict[str, float] = field(default_factory=dict)
                                        # judge_id → uncertainty score [0, 1]

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
            "judge_uncertainties": {
                k: round(v, 4) for k, v in self.judge_uncertainties.items()
            },
            "judges": [jr.to_dict() for jr in self.judge_results],
        }


class VotingEngine:
    """
    Combines multiple JudgeResult objects into a FinalVerdict using
    dynamic uncertainty-weighted voting.

    Args:
        strategy: "dynamic_weighted" (default) | "majority" | "confidence_weighted"
        uncertain_threshold: if >= N raw votes are UNCERTAIN, force UNCERTAIN
    """

    def __init__(self, strategy: str = "dynamic_weighted", uncertain_threshold: int = 2):
        self.strategy = strategy
        self.uncertain_threshold = uncertain_threshold

    def aggregate(self, results: List[JudgeResult]) -> FinalVerdict:
        if not results:
            raise ValueError("No judge results provided to VotingEngine.")

        receipt_id = results[0].receipt_id
        n = len(results)

        # Collect all unique flags across judges
        all_flags = list(dict.fromkeys(f for r in results for f in r.flags))
        avg_confidence = sum(r.confidence for r in results) / n
        vote_counts = dict(Counter(r.label for r in results))

        # Compute per-judge uncertainty scores
        judge_uncertainties = {
            r.judge_id: self._compute_uncertainty(r) for r in results
        }

        if self.strategy == "dynamic_weighted":
            label, strategy_used, tally, verdict_uncertainty = self._dynamic_weighted_vote(
                results, judge_uncertainties
            )
        elif self.strategy == "majority":
            label, strategy_used = self._majority_vote(vote_counts, n)
            tally = f"{label} ({vote_counts.get(label, 0)}/{n})"
            verdict_uncertainty = self._compute_verdict_uncertainty_simple(
                results, judge_uncertainties, label
            )
        elif self.strategy == "confidence_weighted":
            label, strategy_used = self._confidence_weighted_vote(results)
            tally = f"{label} ({vote_counts.get(label, 0)}/{n})"
            verdict_uncertainty = self._compute_verdict_uncertainty_simple(
                results, judge_uncertainties, label
            )
        else:
            # Unknown strategy -> fall back to dynamic_weighted
            label, strategy_used, tally, verdict_uncertainty = self._dynamic_weighted_vote(
                results, judge_uncertainties
            )

        return FinalVerdict(
            receipt_id=receipt_id,
            label=label,
            tally=tally,
            vote_counts=vote_counts,
            avg_confidence=avg_confidence,
            strategy_used=strategy_used,
            all_flags=all_flags,
            judge_results=results,
            verdict_uncertainty=verdict_uncertainty,
            judge_uncertainties=judge_uncertainties,
        )

    # ------------------------------------------------------------------
    # Uncertainty calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_uncertainty(result: JudgeResult) -> float:
        """
        Compute a [0, 1] uncertainty score for a single judge result.

        More severe and precise than simply (100 - confidence) / 100:
          - UNCERTAIN label -> always 1.0.
          - Confidence component (55%): nonlinear; medium-low confidence is
            penalised more than a linear scale would suggest.
          - Skill component (35%): fraction of skill_results that are "uncertain".
          - Consistency penalty: penalises overconfidence on high-risk cases
            and over-caution on clear-cut assessments.

        Returns a float in [0, 1] where 0 = completely certain, 1 = no idea.
        """
        if result.label == "UNCERTAIN":
            return 1.0

        # -- Confidence component (nonlinear) --
        c = result.confidence / 100.0   # 0.0 to 1.0
        if c >= 0.75:
            # High confidence: linear mapping -> u_conf in [0.0, 0.25]
            u_conf = 1.0 - c
        else:
            # Medium/low confidence: steeper penalty -> u_conf in [0.25, 1.0]
            # At c=0.75 -> u_conf=0.25; at c=0.0 -> u_conf=1.0
            u_conf = 0.25 + (0.75 - c)
            u_conf = min(1.0, u_conf)

        # -- Skill component: fraction of uncertain skills --
        n_skills = max(len(result.skill_results), 1)
        n_uncertain = sum(1 for v in result.skill_results.values() if v == "uncertain")
        u_skill = n_uncertain / n_skills

        # -- Consistency penalty --
        consistency_penalty = 0.0
        if result.risk_level == "high" and result.confidence > 80.0:
            # Claims high risk but also very confident -> suspicious overconfidence
            consistency_penalty = 0.12
        elif result.risk_level == "low" and result.confidence < 65.0:
            # Low risk but not confident -> unnecessary caution signals uncertainty
            consistency_penalty = 0.08

        # -- Weighted combination --
        u = 0.55 * u_conf + 0.35 * u_skill + consistency_penalty
        return round(min(1.0, max(0.0, u)), 4)

    # ------------------------------------------------------------------
    # Dynamic weighted voting
    # ------------------------------------------------------------------

    def _dynamic_weighted_vote(
        self,
        results: List[JudgeResult],
        judge_uncertainties: Dict[str, float],
    ) -> tuple:
        """
        Weighted vote where each judge's weight = max(0.1, 1 - uncertainty).

        Winner must capture > 40% of total weight to avoid a single very
        certain judge dominating when others disagree.
        Returns (label, strategy_used, tally, verdict_uncertainty).
        """
        n = len(results)

        # Forced UNCERTAIN if too many raw UNCERTAIN votes
        uncertain_count = sum(1 for r in results if r.label == "UNCERTAIN")
        if uncertain_count >= self.uncertain_threshold:
            u = self._compute_verdict_uncertainty_dynamic(
                results, judge_uncertainties, "UNCERTAIN", {}
            )
            tally = f"UNCERTAIN ({uncertain_count}/{n} uncertain votes)"
            return "UNCERTAIN", "dynamic_weighted (uncertain threshold)", tally, u

        # Compute per-judge vote weights
        weights = {
            r.judge_id: max(0.1, 1.0 - judge_uncertainties[r.judge_id])
            for r in results
        }

        # Accumulate weighted scores per label
        weighted_scores: Dict[str, float] = {"FAKE": 0.0, "REAL": 0.0, "UNCERTAIN": 0.0}
        for r in results:
            weighted_scores[r.label] += weights[r.judge_id]

        total_weight = sum(weights.values())

        # Determine winner (exclude UNCERTAIN from candidacy)
        candidates = {k: v for k, v in weighted_scores.items() if k != "UNCERTAIN"}
        if not candidates:
            u = 1.0
            tally = f"UNCERTAIN (w=0.00/{total_weight:.2f})"
            return "UNCERTAIN", "dynamic_weighted (no valid votes)", tally, u

        winner = max(candidates, key=lambda k: candidates[k])
        winner_weight = candidates[winner]
        winner_ratio = winner_weight / total_weight if total_weight > 0 else 0.0

        if winner_ratio > 0.40:
            label = winner
            strategy_used = "dynamic_weighted"
        else:
            label = "UNCERTAIN"
            strategy_used = "dynamic_weighted (insufficient margin)"

        tally = f"{label} (w={weighted_scores.get(label, 0.0):.2f}/{total_weight:.2f})"

        verdict_uncertainty = self._compute_verdict_uncertainty_dynamic(
            results, judge_uncertainties, label, weights
        )
        return label, strategy_used, tally, verdict_uncertainty

    @staticmethod
    def _compute_verdict_uncertainty_dynamic(
        results: List[JudgeResult],
        judge_uncertainties: Dict[str, float],
        winner_label: str,
        weights: Dict[str, float],
    ) -> float:
        """
        Compute final verdict uncertainty from:
          - Vote spread (40%): how decisively did weights favour the winner?
          - Average judge uncertainty (35%): mean individual uncertainty.
          - Dissent weight (25%): weight of judges who actively voted against winner.

        Returns a float in [0, 1].
        """
        if not results:
            return 1.0

        uncertainties = list(judge_uncertainties.values())
        u_avg = sum(uncertainties) / len(uncertainties) if uncertainties else 1.0

        if not weights or winner_label == "UNCERTAIN":
            return round(min(1.0, 0.40 * 1.0 + 0.35 * u_avg + 0.25 * 1.0), 4)

        total_weight = sum(weights.values())
        if total_weight == 0:
            return 1.0

        winner_w = sum(
            weights.get(r.judge_id, 0.0) for r in results if r.label == winner_label
        )
        u_spread = 1.0 - (winner_w / total_weight)

        dissent_w = sum(
            weights.get(r.judge_id, 0.0)
            for r in results
            if r.label not in (winner_label, "UNCERTAIN")
        )
        u_dissent = dissent_w / total_weight

        verdict_uncertainty = 0.40 * u_spread + 0.35 * u_avg + 0.25 * u_dissent
        return round(min(1.0, max(0.0, verdict_uncertainty)), 4)

    # ------------------------------------------------------------------
    # Legacy strategies (kept for backwards compatibility)
    # ------------------------------------------------------------------

    def _majority_vote(self, vote_counts: dict, n: int) -> tuple:
        if vote_counts.get("UNCERTAIN", 0) >= self.uncertain_threshold:
            return "UNCERTAIN", "majority (uncertain threshold)"

        fake_votes = vote_counts.get("FAKE", 0)
        real_votes = vote_counts.get("REAL", 0)
        majority = n // 2 + 1

        if fake_votes >= majority:
            return "FAKE", "majority"
        if real_votes >= majority:
            return "REAL", "majority"
        return "UNCERTAIN", "majority (no majority)"

    def _confidence_weighted_vote(self, results: List[JudgeResult]) -> tuple:
        weighted: Dict[str, float] = {"FAKE": 0.0, "REAL": 0.0, "UNCERTAIN": 0.0}
        for r in results:
            weighted[r.label] += r.confidence

        candidates = {k: v for k, v in weighted.items() if k != "UNCERTAIN"}
        if not candidates:
            return "UNCERTAIN", "confidence_weighted (all uncertain)"

        winner = max(candidates, key=lambda k: candidates[k])
        if candidates[winner] < 50.0:
            return "UNCERTAIN", "confidence_weighted (low confidence)"

        return winner, "confidence_weighted"

    @staticmethod
    def _compute_verdict_uncertainty_simple(
        results: List[JudgeResult],
        judge_uncertainties: Dict[str, float],
        winner_label: str,
    ) -> float:
        """Simple verdict uncertainty estimate for legacy strategies."""
        n = len(results)
        if n == 0:
            return 1.0
        disagreements = sum(
            1 for r in results
            if r.label != winner_label and r.label != "UNCERTAIN"
        )
        u_spread = disagreements / n
        u_avg = sum(judge_uncertainties.values()) / max(len(judge_uncertainties), 1)
        return round(min(1.0, 0.5 * u_spread + 0.5 * u_avg), 4)
