"""Tests for response latency metric with percentile tracking."""

import pytest

from evalforge.metrics.latency import (
    LatencyMetric,
    LatencyMetricResult,
    LatencySample,
    LatencySeverity,
    PercentileResult,
    SLAComplianceResult,
    SLAThreshold,
)


@pytest.fixture
def metric():
    return LatencyMetric(
        sla=SLAThreshold(
            target_ms=500.0,
            warning_ms=1000.0,
            critical_ms=3000.0,
            timeout_ms=10000.0,
        )
    )


@pytest.fixture
def metric_with_samples(metric):
    """Metric preloaded with typical latency distribution."""
    latencies = [
        100, 150, 200, 220, 250, 280, 300, 320, 350, 380,
        400, 420, 450, 480, 500, 520, 550, 600, 700, 800,
        900, 1100, 1500, 2000, 5000,
    ]
    for latency in latencies:
        metric.record_duration(latency)
    return metric


class TestLatencySample:
    def test_sample_creation(self):
        sample = LatencySample(duration_ms=250.0, model_id="gpt-4")
        assert sample.duration_ms == 250.0
        assert sample.model_id == "gpt-4"
        assert sample.timestamp > 0

    def test_sample_default_timestamp(self):
        sample = LatencySample(duration_ms=100.0)
        assert sample.timestamp > 0


class TestPercentileCalculation:
    def test_basic_percentiles(self, metric_with_samples):
        percentiles = metric_with_samples.get_percentiles()
        assert percentiles.sample_count == 25
        assert percentiles.p50 > 0
        assert percentiles.p90 > percentiles.p50
        assert percentiles.p95 >= percentiles.p90
        assert percentiles.p99 >= percentiles.p95

    def test_min_max(self, metric_with_samples):
        percentiles = metric_with_samples.get_percentiles()
        assert percentiles.min_ms == 100.0
        assert percentiles.max_ms == 5000.0

    def test_mean_calculation(self, metric_with_samples):
        percentiles = metric_with_samples.get_percentiles()
        assert percentiles.mean_ms > 0
        assert percentiles.min_ms <= percentiles.mean_ms <= percentiles.max_ms

    def test_empty_samples(self, metric):
        percentiles = metric.get_percentiles()
        assert percentiles.sample_count == 0
        assert percentiles.p50 == 0.0

    def test_single_sample(self, metric):
        metric.record_duration(500.0)
        percentiles = metric.get_percentiles()
        assert percentiles.p50 == 500.0
        assert percentiles.p99 == 500.0

    def test_standard_deviation(self, metric_with_samples):
        percentiles = metric_with_samples.get_percentiles()
        assert percentiles.std_dev_ms > 0


class TestSLACompliance:
    def test_all_within_target(self, metric):
        for _ in range(20):
            metric.record_duration(200.0)
        result = metric.check_sla_compliance()
        assert result.compliant is True
        assert result.compliance_percentage == 100.0
        assert result.score == 1.0

    def test_some_violations(self, metric):
        # 90 within target, 10 violations
        for _ in range(90):
            metric.record_duration(300.0)
        for _ in range(10):
            metric.record_duration(2000.0)
        result = metric.check_sla_compliance()
        assert result.compliance_percentage == 90.0
        assert result.violations_count == 10

    def test_critical_violations(self, metric):
        for _ in range(50):
            metric.record_duration(5000.0)
        for _ in range(50):
            metric.record_duration(200.0)
        result = metric.check_sla_compliance()
        assert result.compliant is False
        assert result.severity in [LatencySeverity.DEGRADED, LatencySeverity.CRITICAL]

    def test_timeout_detection(self, metric):
        for _ in range(5):
            metric.record_duration(15000.0)  # Exceeds timeout
        for _ in range(95):
            metric.record_duration(200.0)
        result = metric.check_sla_compliance()
        assert result.details["timeouts"] == 5


class TestSlowResponseFlagging:
    def test_flag_slow_responses(self, metric_with_samples):
        slow = metric_with_samples.get_slow_responses()
        # All responses > 1000ms warning threshold
        assert len(slow) > 0
        assert all(s.duration_ms > 1000.0 for s in slow)

    def test_no_slow_responses(self, metric):
        for _ in range(10):
            metric.record_duration(200.0)
        slow = metric.get_slow_responses()
        assert len(slow) == 0


class TestSeverityClassification:
    def test_acceptable(self, metric):
        assert metric.classify_severity(200.0) == LatencySeverity.ACCEPTABLE

    def test_warning(self, metric):
        assert metric.classify_severity(800.0) == LatencySeverity.WARNING

    def test_degraded(self, metric):
        assert metric.classify_severity(2000.0) == LatencySeverity.DEGRADED

    def test_critical(self, metric):
        assert metric.classify_severity(5000.0) == LatencySeverity.CRITICAL

    def test_timeout(self, metric):
        assert metric.classify_severity(15000.0) == LatencySeverity.TIMEOUT


class TestFullEvaluation:
    def test_evaluate_healthy(self, metric):
        for _ in range(100):
            metric.record_duration(200.0)
        result = metric.evaluate()
        assert result.passed is True
        assert result.score >= 0.8

    def test_evaluate_degraded(self, metric):
        for _ in range(50):
            metric.record_duration(200.0)
        for _ in range(50):
            metric.record_duration(5000.0)
        result = metric.evaluate()
        assert result.passed is False
        assert result.score < 0.8

    def test_evaluate_empty(self, metric):
        result = metric.evaluate()
        assert result.passed is True
        assert result.score == 1.0

    def test_recommendation_generated(self, metric):
        for _ in range(10):
            metric.record_duration(5000.0, model_id="slow-model")
        for _ in range(90):
            metric.record_duration(200.0)
        result = metric.evaluate()
        assert len(result.recommendation) > 0


class TestSlidingWindow:
    def test_window_maintains_size(self):
        metric = LatencyMetric(window_size=10)
        for i in range(20):
            metric.record_duration(float(i * 100))
        assert metric.sample_count == 10

    def test_clear_samples(self, metric):
        metric.record_duration(100.0)
        metric.record_duration(200.0)
        assert metric.sample_count == 2
        metric.clear()
        assert metric.sample_count == 0
