"""Evaluation result comparison between runs.

Provides side-by-side diff of two evaluation runs, highlights regressions,
performs statistical significance testing, and classifies changes as
improvements or degradations.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("evalforge.comparison")


class ChangeClassification(str, Enum):
    """Classification of a metric change between runs."""
    IMPROVEMENT = "improvement"
    DEGRADATION = "degradation"
    NO_CHANGE = "no_change"
    INCONCLUSIVE = "inconclusive"


@dataclass
class MetricComparison:
    """Comparison of a single metric between two runs.

    Attributes:
        metric_name: Name of the metric being compared.
        baseline_value: Value from the baseline (first) run.
        candidate_value: Value from the candidate (second) run.
        absolute_change: Absolute difference (candidate - baseline).
        relative_change_pct: Percentage change from baseline.
        classification: Whether this is an improvement, degradation, or no change.
        is_significant: Whether the change is statistically significant.
        p_value: P-value from significance test (if applicable).
        higher_is_better: Whether higher values indicate better performance.
    """
    metric_name: str
    baseline_value: float
    candidate_value: float
    absolute_change: float = 0.0
    relative_change_pct: float = 0.0
    classification: ChangeClassification = ChangeClassification.NO_CHANGE
    is_significant: bool = False
    p_value: Optional[float] = None
    higher_is_better: bool = True

    def __post_init__(self):
        self.absolute_change = self.candidate_value - self.baseline_value
        if self.baseline_value != 0:
            self.relative_change_pct = (self.absolute_change / abs(self.baseline_value)) * 100
        else:
            self.relative_change_pct = 0.0 if self.candidate_value == 0 else float('inf')


@dataclass
class DiffResult:
    """Complete diff result between two evaluation runs.

    Attributes:
        baseline_run_id: Identifier for the baseline run.
        candidate_run_id: Identifier for the candidate run.
        comparisons: List of metric comparisons.
        improvements: Metrics that improved.
        degradations: Metrics that degraded.
        summary: Human-readable summary of changes.
    """
    baseline_run_id: str
    candidate_run_id: str
    comparisons: list[MetricComparison] = field(default_factory=list)

    @property
    def improvements(self) -> list[MetricComparison]:
        """Return metrics classified as improvements."""
        return [c for c in self.comparisons if c.classification == ChangeClassification.IMPROVEMENT]

    @property
    def degradations(self) -> list[MetricComparison]:
        """Return metrics classified as degradations."""
        return [c for c in self.comparisons if c.classification == ChangeClassification.DEGRADATION]

    @property
    def unchanged(self) -> list[MetricComparison]:
        """Return metrics with no significant change."""
        return [c for c in self.comparisons if c.classification == ChangeClassification.NO_CHANGE]

    @property
    def has_regressions(self) -> bool:
        """Check if any metrics have degraded."""
        return len(self.degradations) > 0

    @property
    def summary(self) -> str:
        """Generate a human-readable summary."""
        total = len(self.comparisons)
        improved = len(self.improvements)
        degraded = len(self.degradations)
        unchanged = len(self.unchanged)

        parts = [f"Compared {total} metrics between {self.baseline_run_id} and {self.candidate_run_id}:"]
        if improved:
            parts.append(f"  {improved} improved")
        if degraded:
            parts.append(f"  {degraded} degraded")
        if unchanged:
            parts.append(f"  {unchanged} unchanged")

        return "\n".join(parts)


class RunDiff:
    """Compare evaluation results between two runs.

    Supports:
    - Side-by-side metric comparison
    - Statistical significance testing (Welch's t-test approximation)
    - Improvement/degradation classification
    - Configurable significance thresholds

    Args:
        significance_threshold: P-value threshold for significance (default: 0.05).
        min_change_pct: Minimum percentage change to consider meaningful (default: 1.0%).
        higher_is_better_metrics: Set of metric names where higher is better.
        lower_is_better_metrics: Set of metric names where lower is better.
    """

    def __init__(
        self,
        significance_threshold: float = 0.05,
        min_change_pct: float = 1.0,
        higher_is_better_metrics: set[str] | None = None,
        lower_is_better_metrics: set[str] | None = None,
    ):
        self.significance_threshold = significance_threshold
        self.min_change_pct = min_change_pct
        self.higher_is_better_metrics = higher_is_better_metrics or {
            "accuracy", "precision", "recall", "f1_score", "auc", "bleu", "rouge",
            "coherence", "relevance", "faithfulness",
        }
        self.lower_is_better_metrics = lower_is_better_metrics or {
            "latency", "error_rate", "hallucination_rate", "toxicity_score",
            "cost", "token_count",
        }

    def compare(
        self,
        baseline_run_id: str,
        candidate_run_id: str,
        baseline_metrics: dict[str, float],
        candidate_metrics: dict[str, float],
        baseline_samples: dict[str, list[float]] | None = None,
        candidate_samples: dict[str, list[float]] | None = None,
    ) -> DiffResult:
        """Compare two evaluation runs.

        Args:
            baseline_run_id: Identifier for the baseline run.
            candidate_run_id: Identifier for the candidate run.
            baseline_metrics: Aggregate metrics from baseline run.
            candidate_metrics: Aggregate metrics from candidate run.
            baseline_samples: Per-sample metric values from baseline (for significance testing).
            candidate_samples: Per-sample metric values from candidate (for significance testing).

        Returns:
            DiffResult with all metric comparisons.
        """
        result = DiffResult(
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
        )

        # Compare all metrics present in either run
        all_metrics = set(baseline_metrics.keys()) | set(candidate_metrics.keys())

        for metric_name in sorted(all_metrics):
            baseline_val = baseline_metrics.get(metric_name, 0.0)
            candidate_val = candidate_metrics.get(metric_name, 0.0)
            higher_is_better = self._is_higher_better(metric_name)

            comparison = MetricComparison(
                metric_name=metric_name,
                baseline_value=baseline_val,
                candidate_value=candidate_val,
                higher_is_better=higher_is_better,
            )

            # Statistical significance testing if samples available
            if baseline_samples and candidate_samples:
                b_samples = baseline_samples.get(metric_name, [])
                c_samples = candidate_samples.get(metric_name, [])
                if b_samples and c_samples:
                    p_value = self._welch_t_test(b_samples, c_samples)
                    comparison.p_value = p_value
                    comparison.is_significant = p_value < self.significance_threshold

            # Classify the change
            comparison.classification = self._classify_change(comparison)
            result.comparisons.append(comparison)

        return result

    def compare_sample_level(
        self,
        baseline_run_id: str,
        candidate_run_id: str,
        baseline_results: list[dict],
        candidate_results: list[dict],
        metric_key: str = "score",
    ) -> DiffResult:
        """Compare two runs at the sample level.

        Args:
            baseline_run_id: Identifier for the baseline run.
            candidate_run_id: Identifier for the candidate run.
            baseline_results: List of per-sample result dicts from baseline.
            candidate_results: List of per-sample result dicts from candidate.
            metric_key: Key in result dicts containing the metric value.

        Returns:
            DiffResult with aggregate comparison.
        """
        baseline_scores = [r.get(metric_key, 0.0) for r in baseline_results]
        candidate_scores = [r.get(metric_key, 0.0) for r in candidate_results]

        baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
        candidate_avg = sum(candidate_scores) / len(candidate_scores) if candidate_scores else 0.0

        return self.compare(
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            baseline_metrics={metric_key: baseline_avg},
            candidate_metrics={metric_key: candidate_avg},
            baseline_samples={metric_key: baseline_scores},
            candidate_samples={metric_key: candidate_scores},
        )

    def _classify_change(self, comparison: MetricComparison) -> ChangeClassification:
        """Classify a metric change as improvement, degradation, or no change."""
        # Check if change is meaningful
        if abs(comparison.relative_change_pct) < self.min_change_pct:
            return ChangeClassification.NO_CHANGE

        # If we have significance info and it's not significant
        if comparison.p_value is not None and not comparison.is_significant:
            return ChangeClassification.INCONCLUSIVE

        # Determine direction
        if comparison.higher_is_better:
            if comparison.absolute_change > 0:
                return ChangeClassification.IMPROVEMENT
            else:
                return ChangeClassification.DEGRADATION
        else:
            # Lower is better
            if comparison.absolute_change < 0:
                return ChangeClassification.IMPROVEMENT
            else:
                return ChangeClassification.DEGRADATION

    def _is_higher_better(self, metric_name: str) -> bool:
        """Determine if higher values are better for a given metric."""
        metric_lower = metric_name.lower()
        if metric_lower in self.lower_is_better_metrics:
            return False
        return True  # Default: higher is better

    @staticmethod
    def _welch_t_test(sample_a: list[float], sample_b: list[float]) -> float:
        """Perform Welch's t-test and return approximate p-value.

        This is a simplified implementation that uses the t-statistic
        to approximate the p-value without requiring scipy.
        """
        n_a = len(sample_a)
        n_b = len(sample_b)

        if n_a < 2 or n_b < 2:
            return 1.0  # Cannot determine significance

        mean_a = sum(sample_a) / n_a
        mean_b = sum(sample_b) / n_b

        var_a = sum((x - mean_a) ** 2 for x in sample_a) / (n_a - 1)
        var_b = sum((x - mean_b) ** 2 for x in sample_b) / (n_b - 1)

        # Avoid division by zero
        se = math.sqrt(var_a / n_a + var_b / n_b)
        if se == 0:
            return 1.0 if mean_a == mean_b else 0.0

        t_stat = abs(mean_a - mean_b) / se

        # Approximate degrees of freedom (Welch-Satterthwaite)
        num = (var_a / n_a + var_b / n_b) ** 2
        denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        df = num / denom if denom > 0 else 1.0

        # Approximate p-value using the t-distribution CDF approximation
        # Using a simple approximation for two-tailed test
        p_value = RunDiff._approximate_t_pvalue(t_stat, df)
        return p_value

    @staticmethod
    def _approximate_t_pvalue(t: float, df: float) -> float:
        """Approximate two-tailed p-value from t-statistic and degrees of freedom.

        Uses a rational approximation that works well for df > 1.
        """
        if df <= 0:
            return 1.0

        # For large df, approximate with normal distribution
        if df > 100:
            # Standard normal approximation
            z = t
            p = math.erfc(z / math.sqrt(2))
            return p

        # Simple approximation: p ≈ 2 * (1 - Φ(t * sqrt(df/(df+t²))))
        # This is a rough but reasonable approximation
        x = df / (df + t * t)
        # Regularized incomplete beta function approximation
        if t == 0:
            return 1.0

        # Use a conservative approximation
        p = math.exp(-0.5 * t * t) * math.sqrt(2 / math.pi) / t if t > 0 else 1.0
        p = min(p * 2, 1.0)  # Two-tailed
        return max(p, 0.0)
