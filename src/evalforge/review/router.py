"""Confidence-based routing to human review."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import random

from evalforge.review.queue import ReviewQueue, ReviewItem


class RoutingAction(Enum):
    AUTO_ACCEPT = "auto_accept"
    ROUTE_TO_HUMAN = "route_to_human"
    SAMPLE_FOR_CALIBRATION = "sample_for_calibration"


@dataclass
class RoutingDecision:
    """Decision on whether to route to human review."""

    action: RoutingAction
    reason: str
    confidence: float
    metric_name: str
    auto_score: float

    @property
    def needs_human(self) -> bool:
        return self.action in (RoutingAction.ROUTE_TO_HUMAN, RoutingAction.SAMPLE_FOR_CALIBRATION)


class ConfidenceRouter:
    """Routes evaluations to human review based on confidence.

    Low-confidence evaluations go to humans. A random sample of
    high-confidence evaluations also goes for calibration.

    Usage:
        router = ConfidenceRouter(threshold=0.6, sample_rate=0.05)
        decision = router.route(metric="faithfulness", score=0.72, confidence=0.45)
        if decision.needs_human:
            queue.add(...)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        sample_rate: float = 0.05,
        queue: Optional[ReviewQueue] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.sample_rate = sample_rate
        self.queue = queue or ReviewQueue()
        self._routing_log: List[RoutingDecision] = []

    def route(
        self,
        metric_name: str,
        auto_score: float,
        confidence: float,
    ) -> RoutingDecision:
        """Decide whether to route to human review.

        Args:
            metric_name: Which metric was evaluated
            auto_score: Automated evaluation score
            confidence: Confidence in the automated score (0-1)

        Returns:
            RoutingDecision with action and reason
        """
        # Low confidence → always route to human
        if confidence < self.confidence_threshold:
            decision = RoutingDecision(
                action=RoutingAction.ROUTE_TO_HUMAN,
                reason=f"Low confidence ({confidence:.2f} < {self.confidence_threshold})",
                confidence=confidence,
                metric_name=metric_name,
                auto_score=auto_score,
            )
        # Random sample for calibration
        elif random.random() < self.sample_rate:
            decision = RoutingDecision(
                action=RoutingAction.SAMPLE_FOR_CALIBRATION,
                reason=f"Random calibration sample ({self.sample_rate:.0%} rate)",
                confidence=confidence,
                metric_name=metric_name,
                auto_score=auto_score,
            )
        # High confidence → auto accept
        else:
            decision = RoutingDecision(
                action=RoutingAction.AUTO_ACCEPT,
                reason=f"High confidence ({confidence:.2f} >= {self.confidence_threshold})",
                confidence=confidence,
                metric_name=metric_name,
                auto_score=auto_score,
            )

        self._routing_log.append(decision)
        return decision

    def route_and_queue(
        self,
        query: str,
        response: str,
        metric_name: str,
        auto_score: float,
        confidence: float,
        context: str = "",
    ) -> RoutingDecision:
        """Route and automatically add to queue if needed."""
        decision = self.route(metric_name, auto_score, confidence)
        if decision.needs_human:
            self.queue.add_from_evaluation(
                query=query,
                response=response,
                metric_name=metric_name,
                auto_score=auto_score,
                confidence=confidence,
                context=context,
            )
        return decision

    def get_routing_stats(self) -> Dict[str, int]:
        """Get routing statistics."""
        stats = {"auto_accept": 0, "route_to_human": 0, "sample_for_calibration": 0}
        for d in self._routing_log:
            stats[d.action.value] = stats.get(d.action.value, 0) + 1
        return stats

    @property
    def human_review_rate(self) -> float:
        """Percentage of evaluations routed to humans."""
        if not self._routing_log:
            return 0.0
        human = sum(1 for d in self._routing_log if d.needs_human)
        return human / len(self._routing_log)
