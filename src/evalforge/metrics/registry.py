"""Metric registry - auto-selects metrics based on use case type."""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from evalforge.metrics.base import BaseMetric
from evalforge.metrics.rag import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from evalforge.metrics.text import RougeL, Bleu, Coherence, Conciseness, Fluency
from evalforge.metrics.safety import Toxicity, BiasDetection, InjectionResistance
from evalforge.metrics.classification import Accuracy, Precision, Recall, F1Score


# Global metric registry
_REGISTRY: Dict[str, Type[BaseMetric]] = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "context_precision": ContextPrecision,
    "context_recall": ContextRecall,
    "rouge_l": RougeL,
    "bleu": Bleu,
    "coherence": Coherence,
    "conciseness": Conciseness,
    "fluency": Fluency,
    "toxicity": Toxicity,
    "bias_detection": BiasDetection,
    "injection_resistance": InjectionResistance,
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "f1_score": F1Score,
}

# Metrics auto-selected per use case
_USE_CASE_METRICS: Dict[str, List[str]] = {
    "rag": ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "toxicity"],
    "summarization": ["rouge_l", "bleu", "coherence", "conciseness", "fluency"],
    "classification": ["accuracy", "precision", "recall", "f1_score"],
    "generation": ["fluency", "coherence", "toxicity", "bias_detection"],
    "chat": ["coherence", "toxicity", "injection_resistance", "fluency"],
    "code": ["accuracy", "coherence"],
}


class MetricRegistry:
    """Registry of available metrics with auto-selection.

    Usage:
        registry = MetricRegistry()
        metrics = registry.get_metrics_for("rag")
        metric = registry.get("faithfulness")
    """

    def __init__(self):
        self._metrics = dict(_REGISTRY)

    def get(self, name: str) -> BaseMetric:
        """Get a metric instance by name."""
        if name not in self._metrics:
            available = list(self._metrics.keys())
            raise KeyError(f"Metric '{name}' not found. Available: {available}")
        return self._metrics[name]()

    def get_metrics_for(self, use_case: str) -> List[BaseMetric]:
        """Get auto-selected metrics for a use case type."""
        metric_names = _USE_CASE_METRICS.get(use_case, [])
        return [self._metrics[name]() for name in metric_names if name in self._metrics]

    def register(self, name: str, metric_class: Type[BaseMetric]) -> None:
        """Register a custom metric."""
        self._metrics[name] = metric_class

    def list_metrics(self) -> List[str]:
        """List all available metric names."""
        return sorted(self._metrics.keys())

    def list_by_category(self, category: str) -> List[str]:
        """List metrics by category."""
        return [name for name, cls in self._metrics.items() if cls().category == category]


def get_metrics_for_use_case(use_case: str) -> List[BaseMetric]:
    """Convenience function to get metrics for a use case."""
    registry = MetricRegistry()
    return registry.get_metrics_for(use_case)
