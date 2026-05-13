"""Baseline management for drift detection.

Establishes and maintains quality baselines from historical
evaluation results.
"""

from __future__ import annotations

import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Baseline:
    """A quality baseline for a metric."""

    metric_name: str
    mean: float
    std_dev: float
    sample_count: int
    established_at: float = field(default_factory=time.time)
    min_value: float = 0.0
    max_value: float = 1.0
    percentile_25: float = 0.0
    percentile_75: float = 1.0

    @property
    def lower_bound(self) -> float:
        """2 standard deviations below mean."""
        return max(self.mean - 2 * self.std_dev, 0.0)

    @property
    def upper_bound(self) -> float:
        """2 standard deviations above mean."""
        return min(self.mean + 2 * self.std_dev, 1.0)

    def is_within_bounds(self, value: float) -> bool:
        """Check if a value is within normal range."""
        return self.lower_bound <= value <= self.upper_bound

    def deviation_from_mean(self, value: float) -> float:
        """How many standard deviations from mean."""
        if self.std_dev == 0:
            return 0.0 if value == self.mean else float('inf')
        return (value - self.mean) / self.std_dev

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "mean": self.mean,
            "std_dev": self.std_dev,
            "sample_count": self.sample_count,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }


class BaselineManager:
    """Manages quality baselines for metrics.

    Usage:
        manager = BaselineManager(window_days=7)
        manager.record("faithfulness", 0.91)
        manager.record("faithfulness", 0.89)
        baseline = manager.get_baseline("faithfulness")
    """

    def __init__(self, window_days: int = 7, min_samples: int = 10):
        self.window_seconds = window_days * 86400
        self.min_samples = min_samples
        self._history: Dict[str, List[tuple]] = {}  # metric -> [(timestamp, value)]

    def record(self, metric_name: str, value: float, timestamp: Optional[float] = None) -> None:
        """Record a metric value."""
        ts = timestamp or time.time()
        if metric_name not in self._history:
            self._history[metric_name] = []
        self._history[metric_name].append((ts, value))
        self._cleanup(metric_name)

    def record_batch(self, scores: Dict[str, float], timestamp: Optional[float] = None) -> None:
        """Record multiple metric values at once."""
        ts = timestamp or time.time()
        for metric, value in scores.items():
            self.record(metric, value, ts)

    def get_baseline(self, metric_name: str) -> Optional[Baseline]:
        """Calculate baseline from historical data.

        Returns None if insufficient data.
        """
        values = self._get_values(metric_name)
        if len(values) < self.min_samples:
            return None

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std_dev = math.sqrt(variance)

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        return Baseline(
            metric_name=metric_name,
            mean=round(mean, 6),
            std_dev=round(std_dev, 6),
            sample_count=n,
            min_value=sorted_vals[0],
            max_value=sorted_vals[-1],
            percentile_25=sorted_vals[int(n * 0.25)],
            percentile_75=sorted_vals[int(n * 0.75)],
        )

    def get_recent_mean(self, metric_name: str, last_n: int = 5) -> Optional[float]:
        """Get mean of the most recent N values."""
        values = self._get_values(metric_name)
        if not values:
            return None
        recent = values[-last_n:]
        return sum(recent) / len(recent)

    def has_sufficient_data(self, metric_name: str) -> bool:
        """Check if enough data exists for baseline."""
        return len(self._get_values(metric_name)) >= self.min_samples

    def list_metrics(self) -> List[str]:
        """List all tracked metrics."""
        return list(self._history.keys())

    def _get_values(self, metric_name: str) -> List[float]:
        """Get values within the window."""
        if metric_name not in self._history:
            return []
        cutoff = time.time() - self.window_seconds
        return [v for ts, v in self._history[metric_name] if ts >= cutoff]

    def _cleanup(self, metric_name: str) -> None:
        """Remove old entries outside window."""
        cutoff = time.time() - self.window_seconds
        self._history[metric_name] = [
            (ts, v) for ts, v in self._history[metric_name] if ts >= cutoff
        ]
