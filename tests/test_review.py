"""Tests for human-in-the-loop review."""

import pytest
from evalforge.review.queue import ReviewQueue, ReviewItem, ReviewStatus
from evalforge.review.router import ConfidenceRouter, RoutingAction


class TestReviewQueue:
    def test_add_and_get_pending(self):
        queue = ReviewQueue()
        queue.add_from_evaluation("What is X?", "X is Y", "faithfulness", 0.7, 0.4)
        pending = queue.get_pending()
        assert len(pending) == 1
        assert pending[0].metric_name == "faithfulness"

    def test_complete_review(self):
        queue = ReviewQueue()
        item_id = queue.add_from_evaluation("Q", "A", "quality", 0.6, 0.3)
        assert queue.complete(item_id, score=0.85, feedback="Good", reviewer="alice")
        completed = queue.get_completed()
        assert len(completed) == 1
        assert completed[0].human_score == 0.85

    def test_assign_reviewer(self):
        queue = ReviewQueue()
        item_id = queue.add_from_evaluation("Q", "A", "quality", 0.6, 0.3)
        assert queue.assign(item_id, "bob")
        item = queue._items[item_id]
        assert item.status == ReviewStatus.ASSIGNED
        assert item.assigned_to == "bob"

    def test_skip_item(self):
        queue = ReviewQueue()
        item_id = queue.add_from_evaluation("Q", "A", "quality", 0.6, 0.3)
        assert queue.skip(item_id)
        assert queue._items[item_id].status == ReviewStatus.SKIPPED

    def test_pending_ordered_by_confidence(self):
        queue = ReviewQueue()
        queue.add_from_evaluation("Q1", "A1", "m", 0.5, 0.8)
        queue.add_from_evaluation("Q2", "A2", "m", 0.5, 0.2)
        queue.add_from_evaluation("Q3", "A3", "m", 0.5, 0.5)
        pending = queue.get_pending()
        confidences = [p.confidence for p in pending]
        assert confidences == sorted(confidences)  # lowest first

    def test_agreement_score(self):
        queue = ReviewQueue()
        id1 = queue.add_from_evaluation("Q1", "A1", "m", 0.80, 0.5)
        id2 = queue.add_from_evaluation("Q2", "A2", "m", 0.70, 0.5)
        queue.complete(id1, score=0.82)  # close to auto (0.80)
        queue.complete(id2, score=0.30)  # far from auto (0.70)
        agreement = queue.get_agreement_score()
        assert agreement == 0.5  # 1 out of 2 agree

    def test_stats(self):
        queue = ReviewQueue()
        queue.add_from_evaluation("Q1", "A1", "m", 0.5, 0.3)
        queue.add_from_evaluation("Q2", "A2", "m", 0.5, 0.3)
        id3 = queue.add_from_evaluation("Q3", "A3", "m", 0.5, 0.3)
        queue.complete(id3, score=0.9)
        stats = queue.get_stats()
        assert stats["total"] == 3
        assert stats["pending"] == 2
        assert stats["completed"] == 1


class TestConfidenceRouter:
    def test_low_confidence_routes_to_human(self):
        router = ConfidenceRouter(confidence_threshold=0.6)
        decision = router.route("faithfulness", auto_score=0.7, confidence=0.3)
        assert decision.action == RoutingAction.ROUTE_TO_HUMAN
        assert decision.needs_human

    def test_high_confidence_auto_accepts(self):
        router = ConfidenceRouter(confidence_threshold=0.6, sample_rate=0.0)
        decision = router.route("faithfulness", auto_score=0.9, confidence=0.95)
        assert decision.action == RoutingAction.AUTO_ACCEPT
        assert not decision.needs_human

    def test_route_and_queue(self):
        router = ConfidenceRouter(confidence_threshold=0.6)
        decision = router.route_and_queue(
            query="What is X?", response="X is Y",
            metric_name="faithfulness", auto_score=0.5, confidence=0.3,
        )
        assert decision.needs_human
        assert len(router.queue.get_pending()) == 1

    def test_routing_stats(self):
        router = ConfidenceRouter(confidence_threshold=0.6, sample_rate=0.0)
        router.route("m", 0.9, 0.9)  # auto
        router.route("m", 0.5, 0.3)  # human
        router.route("m", 0.8, 0.8)  # auto
        stats = router.get_routing_stats()
        assert stats["auto_accept"] == 2
        assert stats["route_to_human"] == 1

    def test_human_review_rate(self):
        router = ConfidenceRouter(confidence_threshold=0.6, sample_rate=0.0)
        for _ in range(7):
            router.route("m", 0.9, 0.9)
        for _ in range(3):
            router.route("m", 0.5, 0.3)
        assert abs(router.human_review_rate - 0.3) < 0.01
