"""Tests for drift detection and scheduling."""

import time
import pytest
from evalforge.drift.baseline import BaselineManager, Baseline
from evalforge.drift.detector import DriftDetector, DriftSeverity
from evalforge.drift.scheduler import EvalScheduler
from evalforge.core.result import EvalResult, MetricScore


class TestBaselineManager:
    def test_record_and_get_baseline(self):
        mgr = BaselineManager(window_days=30, min_samples=5)
        for v in [0.85, 0.87, 0.90, 0.88, 0.86, 0.89]:
            mgr.record("faithfulness", v)
        baseline = mgr.get_baseline("faithfulness")
        assert baseline is not None
        assert 0.85 <= baseline.mean <= 0.90
        assert baseline.std_dev > 0
        assert baseline.sample_count == 6

    def test_insufficient_data(self):
        mgr = BaselineManager(min_samples=10)
        mgr.record("faithfulness", 0.85)
        mgr.record("faithfulness", 0.87)
        assert mgr.get_baseline("faithfulness") is None
        assert not mgr.has_sufficient_data("faithfulness")

    def test_record_batch(self):
        mgr = BaselineManager(min_samples=3)
        for _ in range(5):
            mgr.record_batch({"faithfulness": 0.90, "toxicity": 0.02})
        assert mgr.has_sufficient_data("faithfulness")
        assert mgr.has_sufficient_data("toxicity")

    def test_recent_mean(self):
        mgr = BaselineManager(min_samples=3)
        for v in [0.80, 0.82, 0.84, 0.86, 0.88]:
            mgr.record("quality", v)
        recent = mgr.get_recent_mean("quality", last_n=2)
        assert recent is not None
        assert abs(recent - 0.87) < 0.01

    def test_baseline_bounds(self):
        mgr = BaselineManager(min_samples=5)
        for v in [0.90, 0.90, 0.90, 0.90, 0.90]:
            mgr.record("stable", v)
        baseline = mgr.get_baseline("stable")
        assert baseline.lower_bound <= baseline.mean <= baseline.upper_bound

    def test_list_metrics(self):
        mgr = BaselineManager(min_samples=1)
        mgr.record("a", 0.5)
        mgr.record("b", 0.6)
        assert set(mgr.list_metrics()) == {"a", "b"}


class TestDriftDetector:
    def setup_method(self):
        self.baseline = BaselineManager(window_days=30, min_samples=5)
        # Establish baseline
        import random as _rnd
        for _ in range(20):
            self.baseline.record("faithfulness", 0.90 + _rnd.uniform(-0.02, 0.02))
            self.baseline.record("toxicity", 0.02 + _rnd.uniform(-0.005, 0.005))
        self.detector = DriftDetector(self.baseline, sensitivity="medium")

    def test_no_drift_stable(self):
        results = self.detector.check({"faithfulness": 0.89, "toxicity": 0.02})
        assert all(r.severity == DriftSeverity.NONE for r in results)

    def test_quality_degradation_detected(self):
        results = self.detector.check({"faithfulness": 0.60})
        faith_result = [r for r in results if r.metric_name == "faithfulness"][0]
        assert faith_result.is_degradation
        assert faith_result.severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL)

    def test_safety_metric_increase_detected(self):
        results = self.detector.check({"toxicity": 0.30})
        tox_result = [r for r in results if r.metric_name == "toxicity"][0]
        assert tox_result.is_degradation
        assert tox_result.severity != DriftSeverity.NONE

    def test_needs_alert(self):
        results = self.detector.check({"faithfulness": 0.50})
        assert any(r.needs_alert for r in results)

    def test_should_rollback(self):
        assert self.detector.should_rollback({"faithfulness": 0.30})
        assert not self.detector.should_rollback({"faithfulness": 0.89})

    def test_check_and_record(self):
        initial_count = len(self.baseline._get_values("faithfulness"))
        self.detector.check_and_record({"faithfulness": 0.88})
        assert len(self.baseline._get_values("faithfulness")) == initial_count + 1

    def test_get_degraded_metrics(self):
        degraded = self.detector.get_degraded_metrics({"faithfulness": 0.50, "toxicity": 0.02})
        assert "faithfulness" in degraded

    def test_summary(self):
        summary = self.detector.summary({"faithfulness": 0.50, "toxicity": 0.02})
        assert "faithfulness" in summary
        assert "degraded" in summary.lower() or "⚠" in summary or "🔴" in summary

    def test_insufficient_baseline(self):
        fresh_detector = DriftDetector(BaselineManager(min_samples=100))
        results = fresh_detector.check({"faithfulness": 0.50})
        assert all(r.severity == DriftSeverity.NONE for r in results)

    def test_alert_history(self):
        self.detector.check({"faithfulness": 0.30})
        history = self.detector.get_alert_history()
        assert len(history) > 0


class TestEvalScheduler:
    def setup_method(self):
        self.scheduler = EvalScheduler()

    def _make_result(self, pass_rate: float) -> EvalResult:
        scores = [
            MetricScore(name="faithfulness", score=pass_rate, threshold=0.85, passed=pass_rate >= 0.85),
            MetricScore(name="toxicity", score=0.02, threshold=0.05, passed=True),
        ]
        return EvalResult(
            project_name="test", use_case_type="rag",
            scores=scores, total_samples=10, total_latency_ms=100, model="test",
        )

    def test_start_and_complete_run(self):
        run = self.scheduler.start_run(triggered_by="manual")
        assert run.status == "running"
        result = self._make_result(0.90)
        self.scheduler.complete_run(run.run_id, result)
        completed = self.scheduler.get_run(run.run_id)
        assert completed.status == "completed"
        assert completed.passed

    def test_fail_run(self):
        run = self.scheduler.start_run()
        self.scheduler.fail_run(run.run_id, "Connection timeout")
        failed = self.scheduler.get_run(run.run_id)
        assert failed.status == "failed"
        assert failed.error == "Connection timeout"

    def test_get_history(self):
        for _ in range(5):
            run = self.scheduler.start_run()
            self.scheduler.complete_run(run.run_id, self._make_result(0.90))
        history = self.scheduler.get_history(last_n=3)
        assert len(history) == 3

    def test_get_trend(self):
        for score in [0.85, 0.87, 0.90, 0.88, 0.86]:
            run = self.scheduler.start_run()
            result = EvalResult(
                project_name="test", use_case_type="rag",
                scores=[MetricScore(name="faithfulness", score=score, threshold=0.85, passed=score >= 0.85)],
                total_samples=10, total_latency_ms=100, model="test",
            )
            self.scheduler.complete_run(run.run_id, result)

        trend = self.scheduler.get_trend("faithfulness", last_n=5)
        assert len(trend) == 5
        assert all("value" in t for t in trend)

    def test_detect_regression_declining(self):
        for rate in [0.95, 0.90, 0.85, 0.80, 0.75]:
            run = self.scheduler.start_run()
            self.scheduler.complete_run(run.run_id, self._make_result(rate))
        regression = self.scheduler.detect_regression(last_n=5)
        assert regression is not None
        assert "Declining" in regression

    def test_no_regression_stable(self):
        for _ in range(5):
            run = self.scheduler.start_run()
            self.scheduler.complete_run(run.run_id, self._make_result(0.90))
        assert self.scheduler.detect_regression() is None

    def test_success_rate(self):
        for rate in [0.90, 0.90, 0.70, 0.90]:
            run = self.scheduler.start_run()
            self.scheduler.complete_run(run.run_id, self._make_result(rate))
        # 3 out of 4 pass (faithfulness >= 0.85)
        assert self.scheduler.success_rate == 0.75

    def test_summary(self):
        run = self.scheduler.start_run()
        self.scheduler.complete_run(run.run_id, self._make_result(0.90))
        summary = self.scheduler.summary()
        assert "Total runs" in summary
        assert "Success rate" in summary

    def test_pass_rate_trend(self):
        for rate in [0.90, 0.85, 0.80]:
            run = self.scheduler.start_run()
            self.scheduler.complete_run(run.run_id, self._make_result(rate))
        trend = self.scheduler.get_pass_rate_trend()
        assert len(trend) == 3
