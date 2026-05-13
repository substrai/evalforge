"""Base metric interface for EvalForge."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MetricInput:
    """Input to a metric evaluation."""

    query: str = ""
    response: str = ""
    context: str = ""
    reference: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricOutput:
    """Output from a metric evaluation."""

    score: float  # 0.0 to 1.0
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


class BaseMetric(ABC):
    """Base class for all evaluation metrics.

    Subclass this to create custom metrics.
    """

    name: str = "base"
    description: str = ""
    category: str = "general"  # rag, text, safety, classification, code

    @abstractmethod
    def evaluate(self, input: MetricInput, threshold: float = 0.8) -> MetricOutput:
        """Evaluate a single sample.

        Args:
            input: MetricInput with query, response, context, reference
            threshold: Pass/fail threshold

        Returns:
            MetricOutput with score and details
        """
        pass

    def evaluate_batch(self, inputs: List[MetricInput], threshold: float = 0.8) -> List[MetricOutput]:
        """Evaluate a batch of samples."""
        return [self.evaluate(inp, threshold) for inp in inputs]

    def aggregate(self, outputs: List[MetricOutput]) -> float:
        """Aggregate scores from multiple evaluations."""
        if not outputs:
            return 0.0
        return sum(o.score for o in outputs) / len(outputs)
