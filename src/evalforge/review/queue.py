"""Human review queue management."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ReviewStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"
    EXPIRED = "expired"
    SKIPPED = "skipped"


@dataclass
class ReviewItem:
    """A single item in the review queue."""

    item_id: str
    query: str
    response: str
    context: str = ""
    metric_name: str = ""
    auto_score: float = 0.0
    confidence: float = 0.0
    status: ReviewStatus = ReviewStatus.PENDING
    assigned_to: Optional[str] = None
    human_score: Optional[float] = None
    human_feedback: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_pending(self) -> bool:
        return self.status == ReviewStatus.PENDING

    @property
    def wait_hours(self) -> float:
        return (time.time() - self.created_at) / 3600

    def complete(self, score: float, feedback: str = "", reviewer: str = "") -> None:
        self.human_score = score
        self.human_feedback = feedback
        self.assigned_to = reviewer or self.assigned_to
        self.status = ReviewStatus.COMPLETED
        self.completed_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "query": self.query[:100],
            "response": self.response[:100],
            "metric_name": self.metric_name,
            "auto_score": self.auto_score,
            "confidence": self.confidence,
            "status": self.status.value,
            "human_score": self.human_score,
            "wait_hours": round(self.wait_hours, 1),
        }


class ReviewQueue:
    """Manages the human review queue.

    Usage:
        queue = ReviewQueue(max_wait_hours=24)
        queue.add(ReviewItem(...))
        pending = queue.get_pending()
        queue.complete("item-1", score=0.9, feedback="Good answer")
    """

    def __init__(self, max_wait_hours: int = 24, max_queue_size: int = 1000):
        self._items: Dict[str, ReviewItem] = {}
        self.max_wait_hours = max_wait_hours
        self.max_queue_size = max_queue_size
        self._counter = 0

    def add(self, item: ReviewItem) -> str:
        """Add an item to the review queue. Returns item_id."""
        if not item.item_id:
            self._counter += 1
            item.item_id = f"review-{self._counter:05d}"
        self._items[item.item_id] = item
        self._expire_old()
        return item.item_id

    def add_from_evaluation(
        self,
        query: str,
        response: str,
        metric_name: str,
        auto_score: float,
        confidence: float,
        context: str = "",
    ) -> str:
        """Convenience method to add from evaluation results."""
        self._counter += 1
        item = ReviewItem(
            item_id=f"review-{self._counter:05d}",
            query=query,
            response=response,
            context=context,
            metric_name=metric_name,
            auto_score=auto_score,
            confidence=confidence,
        )
        return self.add(item)

    def get_pending(self, limit: int = 50) -> List[ReviewItem]:
        """Get pending items ordered by priority (lowest confidence first)."""
        pending = [i for i in self._items.values() if i.status == ReviewStatus.PENDING]
        pending.sort(key=lambda x: x.confidence)
        return pending[:limit]

    def assign(self, item_id: str, reviewer: str) -> bool:
        """Assign an item to a reviewer."""
        item = self._items.get(item_id)
        if item and item.status == ReviewStatus.PENDING:
            item.status = ReviewStatus.ASSIGNED
            item.assigned_to = reviewer
            return True
        return False

    def complete(self, item_id: str, score: float, feedback: str = "", reviewer: str = "") -> bool:
        """Complete a review with human judgment."""
        item = self._items.get(item_id)
        if item and item.status in (ReviewStatus.PENDING, ReviewStatus.ASSIGNED):
            item.complete(score, feedback, reviewer)
            return True
        return False

    def skip(self, item_id: str) -> bool:
        """Skip a review item."""
        item = self._items.get(item_id)
        if item and item.status in (ReviewStatus.PENDING, ReviewStatus.ASSIGNED):
            item.status = ReviewStatus.SKIPPED
            return True
        return False

    def get_completed(self) -> List[ReviewItem]:
        """Get all completed reviews."""
        return [i for i in self._items.values() if i.status == ReviewStatus.COMPLETED]

    def get_agreement_score(self) -> float:
        """Calculate agreement between auto and human scores."""
        completed = self.get_completed()
        if not completed:
            return 0.0
        agreements = sum(
            1 for i in completed
            if i.human_score is not None and abs(i.auto_score - i.human_score) < 0.15
        )
        return agreements / len(completed)

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        items = list(self._items.values())
        return {
            "total": len(items),
            "pending": sum(1 for i in items if i.status == ReviewStatus.PENDING),
            "assigned": sum(1 for i in items if i.status == ReviewStatus.ASSIGNED),
            "completed": sum(1 for i in items if i.status == ReviewStatus.COMPLETED),
            "expired": sum(1 for i in items if i.status == ReviewStatus.EXPIRED),
            "agreement_score": round(self.get_agreement_score(), 4),
        }

    def _expire_old(self) -> None:
        """Expire items that exceeded max wait time."""
        for item in self._items.values():
            if item.status == ReviewStatus.PENDING and item.wait_hours > self.max_wait_hours:
                item.status = ReviewStatus.EXPIRED
