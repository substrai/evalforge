"""Classification metrics: accuracy, precision, recall, F1."""

from __future__ import annotations

from collections import Counter
from typing import List

from evalforge.metrics.base import BaseMetric, MetricInput, MetricOutput


class Accuracy(BaseMetric):
    """Classification accuracy."""
    name = "accuracy"
    description = "Proportion of correct predictions"
    category = "classification"

    def evaluate(self, input: MetricInput, threshold: float = 0.90) -> MetricOutput:
        # For single sample: check if response matches reference
        if not input.response or not input.reference:
            return MetricOutput(score=0.0, passed=False, explanation="Missing response or reference")
        match = input.response.strip().lower() == input.reference.strip().lower()
        score = 1.0 if match else 0.0
        return MetricOutput(score=score, passed=score >= threshold, details={"match": match})


class Precision(BaseMetric):
    """Classification precision."""
    name = "precision"
    description = "Proportion of positive predictions that are correct"
    category = "classification"

    def evaluate(self, input: MetricInput, threshold: float = 0.85) -> MetricOutput:
        match = input.response.strip().lower() == input.reference.strip().lower()
        score = 1.0 if match else 0.0
        return MetricOutput(score=score, passed=score >= threshold, details={"match": match})


class Recall(BaseMetric):
    """Classification recall."""
    name = "recall"
    description = "Proportion of actual positives correctly identified"
    category = "classification"

    def evaluate(self, input: MetricInput, threshold: float = 0.85) -> MetricOutput:
        match = input.response.strip().lower() == input.reference.strip().lower()
        score = 1.0 if match else 0.0
        return MetricOutput(score=score, passed=score >= threshold, details={"match": match})


class F1Score(BaseMetric):
    """F1 score (harmonic mean of precision and recall)."""
    name = "f1_score"
    description = "Harmonic mean of precision and recall"
    category = "classification"

    def evaluate(self, input: MetricInput, threshold: float = 0.85) -> MetricOutput:
        match = input.response.strip().lower() == input.reference.strip().lower()
        score = 1.0 if match else 0.0
        return MetricOutput(score=score, passed=score >= threshold, details={"match": match})
