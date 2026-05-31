"""Tests for evaluation result comparison between runs."""

import pytest

from evalforge.comparison.run_diff import (
    RunDiff,
    DiffResult,
    MetricComparison,
    ChangeClassification,
)


@pytest.fixture
def differ():
    """Create a RunDiff instance with default settings."""
    return RunDiff(significance_threshold=0.05, min_change_pct=1.0)


@pytest.fixture
def baseline_metrics():
    """Sample baseline metrics."""
    return {
        "accuracy": 0.85,
        "f1_score": 0.82,
        "latency": 150.0,
        "error_rate": 0.05,
    }


@pytest.fixture
def improved_metrics():
    """Sample metrics showing improvement."""
    return {
        "accuracy": 0.92,
        "f1_score": 0.89,
        "latency": 120.0,
        "error_rate": 0.02,
    }


@pytest.fixture
def degraded_metrics():
    """Sample metrics showing degradation."""
    return {
        "accuracy": 0.75,
        "f1_score": 0.70,
        "latency": 200.0,
        "error_rate": 0.10,
    }


class TestMetricComparison:
    """Test individual metric comparison logic."""

    def test_absolute_change_calculation(self):
        """Test that absolute change is computed correctly."""
        comp = MetricComparison(
            metric_name="accuracy",
            baseline_value=0.80,
            candidate_value=0.85,
        )
        assert comp.absolute_change == pytest.approx(0.05)

    def test_relative_change_calculation(self):
        """Test that relative change percentage is computed correctly."""
        comp = MetricComparison(
            metric_name="accuracy",
            baseline_value=0.80,
            candidate_value=0.88,
        )
        assert comp.relative_change_pct == pytest.approx(10.0)

    def test_zero_baseline_handles_gracefully(self):
        """Test that zero baseline doesn't cause division by zero."""
        comp = MetricComparison(
            metric_name="error_rate",
            baseline_value=0.0,
            candidate_value=0.05,
        )
        assert comp.relative_change_pct == float('inf')

    def test_zero_to_zero_is_zero_change(self):
        """Test that 0 -> 0 shows no change."""
        comp = MetricComparison(
            metric_name="error_rate",
            baseline_value=0.0,
            candidate_value=0.0,
        )
        assert comp.relative_change_pct == 0.0
        assert comp.absolute_change == 0.0


class TestRunComparison:
    """Test full run comparison."""

    def test_compare_detects_improvements(self, differ, baseline_metrics, improved_metrics):
        """Test that improvements are correctly detected."""
        result = differ.compare(
            baseline_run_id="run-001",
            candidate_run_id="run-002",
            baseline_metrics=baseline_metrics,
            candidate_metrics=improved_metrics,
        )

        assert len(result.improvements) > 0
        accuracy_comp = next(c for c in result.comparisons if c.metric_name == "accuracy")
        assert accuracy_comp.classification == ChangeClassification.IMPROVEMENT

    def test_compare_detects_degradations(self, differ, baseline_metrics, degraded_metrics):
        """Test that degradations are correctly detected."""
        result = differ.compare(
            baseline_run_id="run-001",
            candidate_run_id="run-003",
            baseline_metrics=baseline_metrics,
            candidate_metrics=degraded_metrics,
        )

        assert result.has_regressions is True
        assert len(result.degradations) > 0

    def test_compare_lower_is_better_metrics(self, differ):
        """Test that lower-is-better metrics are classified correctly."""
        result = differ.compare(
            baseline_run_id="run-a",
            candidate_run_id="run-b",
            baseline_metrics={"latency": 200.0},
            candidate_metrics={"latency": 150.0},
        )

        latency_comp = result.comparisons[0]
        # Latency decreased = improvement (lower is better)
        assert latency_comp.classification == ChangeClassification.IMPROVEMENT

    def test_compare_no_change_within_threshold(self, differ):
        """Test that changes below min_change_pct are classified as no change."""
        result = differ.compare(
            baseline_run_id="run-a",
            candidate_run_id="run-b",
            baseline_metrics={"accuracy": 0.900},
            candidate_metrics={"accuracy": 0.901},  # 0.11% change < 1% threshold
        )

        assert result.comparisons[0].classification == ChangeClassification.NO_CHANGE

    def test_compare_handles_missing_metrics(self, differ):
        """Test comparison when metrics exist in only one run."""
        result = differ.compare(
            baseline_run_id="run-a",
            candidate_run_id="run-b",
            baseline_metrics={"accuracy": 0.85, "old_metric": 0.5},
            candidate_metrics={"accuracy": 0.90, "new_metric": 0.7},
        )

        metric_names = {c.metric_name for c in result.comparisons}
        assert "accuracy" in metric_names
        assert "old_metric" in metric_names
        assert "new_metric" in metric_names


class TestStatisticalSignificance:
    """Test statistical significance testing."""

    def test_significant_difference_detected(self, differ):
        """Test that a clearly significant difference is detected."""
        baseline_samples = {"accuracy": [0.80, 0.82, 0.81, 0.79, 0.83, 0.80, 0.81, 0.82, 0.80, 0.81]}
        candidate_samples = {"accuracy": [0.92, 0.93, 0.91, 0.94, 0.92, 0.93, 0.91, 0.92, 0.93, 0.92]}

        result = differ.compare(
            baseline_run_id="run-a",
            candidate_run_id="run-b",
            baseline_metrics={"accuracy": 0.81},
            candidate_metrics={"accuracy": 0.92},
            baseline_samples=baseline_samples,
            candidate_samples=candidate_samples,
        )

        comp = result.comparisons[0]
        assert comp.is_significant is True
        assert comp.p_value is not None
        assert comp.p_value < 0.05

    def test_insignificant_difference_marked_inconclusive(self, differ):
        """Test that noisy data with small difference is inconclusive."""
        import random
        random.seed(42)
        baseline_samples = {"accuracy": [0.80 + random.uniform(-0.1, 0.1) for _ in range(10)]}
        candidate_samples = {"accuracy": [0.81 + random.uniform(-0.1, 0.1) for _ in range(10)]}

        result = differ.compare(
            baseline_run_id="run-a",
            candidate_run_id="run-b",
            baseline_metrics={"accuracy": 0.80},
            candidate_metrics={"accuracy": 0.81},
            baseline_samples=baseline_samples,
            candidate_samples=candidate_samples,
        )

        comp = result.comparisons[0]
        # With high variance and small difference, should not be significant
        assert comp.classification in (ChangeClassification.INCONCLUSIVE, ChangeClassification.NO_CHANGE)


class TestDiffResult:
    """Test DiffResult properties and summary."""

    def test_summary_generation(self, differ, baseline_metrics, improved_metrics):
        """Test that summary is generated correctly."""
        result = differ.compare(
            baseline_run_id="baseline-v1",
            candidate_run_id="candidate-v2",
            baseline_metrics=baseline_metrics,
            candidate_metrics=improved_metrics,
        )

        summary = result.summary
        assert "baseline-v1" in summary
        assert "candidate-v2" in summary
        assert "4 metrics" in summary or "Compared 4" in summary

    def test_has_regressions_false_when_all_improve(self, differ, baseline_metrics, improved_metrics):
        """Test has_regressions is False when everything improves."""
        result = differ.compare(
            baseline_run_id="run-a",
            candidate_run_id="run-b",
            baseline_metrics=baseline_metrics,
            candidate_metrics=improved_metrics,
        )

        assert result.has_regressions is False

    def test_sample_level_comparison(self, differ):
        """Test comparing runs at the sample level."""
        baseline_results = [{"score": 0.8}, {"score": 0.7}, {"score": 0.9}, {"score": 0.85}]
        candidate_results = [{"score": 0.9}, {"score": 0.85}, {"score": 0.95}, {"score": 0.92}]

        result = differ.compare_sample_level(
            baseline_run_id="run-a",
            candidate_run_id="run-b",
            baseline_results=baseline_results,
            candidate_results=candidate_results,
            metric_key="score",
        )

        assert len(result.comparisons) == 1
        assert result.comparisons[0].metric_name == "score"
        assert result.comparisons[0].candidate_value > result.comparisons[0].baseline_value
