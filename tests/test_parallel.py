"""Tests for parallel metric execution."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from evalforge.execution.parallel import (
    ExecutionConfig,
    ExecutionReport,
    MetricResult,
    MetricTask,
    ParallelExecutor,
)


# --- Helpers ---


def make_metric_fn(score: float = 0.85, passed: bool = True, delay: float = 0.0):
    """Create a simple metric function that returns a score."""

    def fn(samples=None, context=None, **kwargs):
        if delay > 0:
            time.sleep(delay)
        return MetricResult(
            name="test",
            score=score,
            passed=passed,
            details={"sample_count": len(samples) if samples else 0},
        )

    return fn


def make_failing_metric_fn(error_cls=RuntimeError, message="Metric failed"):
    """Create a metric function that raises an exception."""

    def fn(samples=None, context=None, **kwargs):
        raise error_cls(message)

    return fn


def make_dependency_aware_fn(name: str, base_score: float = 0.8):
    """Create a metric that uses dependency results."""

    def fn(samples=None, context=None, dependencies=None, **kwargs):
        dep_scores = []
        if dependencies:
            dep_scores = [r.score for r in dependencies.values() if r.success]

        # Composite score based on dependencies
        if dep_scores:
            score = (base_score + sum(dep_scores)) / (1 + len(dep_scores))
        else:
            score = base_score

        return MetricResult(name=name, score=score, passed=score >= 0.7)

    return fn


# --- Tests ---


class TestParallelExecutorBasic:
    """Basic parallel executor functionality."""

    def test_empty_executor_returns_empty_report(self):
        """An executor with no tasks should return an empty report."""
        executor = ParallelExecutor()
        report = executor.execute(samples=[])

        assert report.metrics_executed == 0
        assert report.metrics_succeeded == 0
        assert report.total_latency_ms >= 0

    def test_single_metric_execution(self):
        """A single metric should execute and return results."""
        executor = ParallelExecutor()
        executor.add_task(MetricTask(
            name="relevance",
            fn=make_metric_fn(score=0.92, passed=True),
        ))

        report = executor.execute(samples=["sample1", "sample2"])

        assert report.metrics_executed == 1
        assert report.metrics_succeeded == 1
        assert report.get_score("relevance") == pytest.approx(0.92)
        assert report.all_passed is True

    def test_multiple_independent_metrics_run_in_parallel(self):
        """Independent metrics should run concurrently."""
        executor = ParallelExecutor(config=ExecutionConfig(max_workers=4))

        # Each metric takes 0.1s; sequential would be 0.4s
        for i in range(4):
            executor.add_task(MetricTask(
                name=f"metric-{i}",
                fn=make_metric_fn(score=0.8 + i * 0.05, delay=0.1),
            ))

        start = time.perf_counter()
        report = executor.execute(samples=["test"])
        elapsed = time.perf_counter() - start

        assert report.metrics_executed == 4
        assert report.metrics_succeeded == 4
        # Should complete in ~0.1s (parallel), not ~0.4s (sequential)
        assert elapsed < 0.35

    def test_add_and_remove_tasks(self):
        """Should support adding and removing tasks."""
        executor = ParallelExecutor()
        executor.add_task(MetricTask(name="a", fn=make_metric_fn()))
        executor.add_task(MetricTask(name="b", fn=make_metric_fn()))

        assert executor.task_count == 2

        executor.remove_task("a")
        assert executor.task_count == 1

    def test_duplicate_task_name_raises(self):
        """Adding a task with a duplicate name should raise ValueError."""
        executor = ParallelExecutor()
        executor.add_task(MetricTask(name="dup", fn=make_metric_fn()))

        with pytest.raises(ValueError, match="already registered"):
            executor.add_task(MetricTask(name="dup", fn=make_metric_fn()))


class TestParallelExecutorDependencies:
    """Dependency-aware scheduling tests."""

    def test_dependent_metrics_execute_in_order(self):
        """Metrics with dependencies should wait for prerequisites."""
        executor = ParallelExecutor()

        executor.add_task(MetricTask(
            name="base",
            fn=make_metric_fn(score=0.9),
        ))
        executor.add_task(MetricTask(
            name="composite",
            fn=make_dependency_aware_fn("composite", base_score=0.8),
            depends_on=["base"],
        ))

        report = executor.execute(samples=["test"])

        assert report.metrics_executed == 2
        assert report.metrics_succeeded == 2
        # Composite should have access to base results
        assert report.get_score("composite") is not None

    def test_diamond_dependency_graph(self):
        """Diamond DAG: A -> B, A -> C, B -> D, C -> D."""
        executor = ParallelExecutor()

        executor.add_task(MetricTask(name="A", fn=make_metric_fn(score=0.9)))
        executor.add_task(MetricTask(
            name="B", fn=make_metric_fn(score=0.85), depends_on=["A"]
        ))
        executor.add_task(MetricTask(
            name="C", fn=make_metric_fn(score=0.88), depends_on=["A"]
        ))
        executor.add_task(MetricTask(
            name="D",
            fn=make_dependency_aware_fn("D"),
            depends_on=["B", "C"],
        ))

        report = executor.execute(samples=["test"])
        assert report.metrics_executed == 4
        assert report.execution_layers == 3  # [A], [B, C], [D]

    def test_circular_dependency_raises(self):
        """Circular dependencies should raise ValueError."""
        executor = ParallelExecutor()
        executor.add_task(MetricTask(name="A", fn=make_metric_fn(), depends_on=["B"]))
        executor.add_task(MetricTask(name="B", fn=make_metric_fn(), depends_on=["A"]))

        with pytest.raises(ValueError, match="Circular dependency"):
            executor.execute(samples=[])

    def test_validate_dependencies_missing_dep(self):
        """validate_dependencies should catch missing dependencies."""
        executor = ParallelExecutor()
        executor.add_task(MetricTask(
            name="child", fn=make_metric_fn(), depends_on=["nonexistent"]
        ))

        errors = executor.validate_dependencies()
        assert len(errors) >= 1
        assert "nonexistent" in errors[0]


class TestParallelExecutorErrorHandling:
    """Error handling and timeout tests."""

    def test_failed_metric_with_skip_policy(self):
        """Failed metrics with 'skip' policy should not crash execution."""
        config = ExecutionConfig(on_metric_error="skip")
        executor = ParallelExecutor(config=config)
        executor.add_task(MetricTask(name="bad", fn=make_failing_metric_fn()))
        executor.add_task(MetricTask(name="good", fn=make_metric_fn(score=0.9)))

        report = executor.execute(samples=["test"])

        assert report.metrics_executed == 2
        assert report.metrics_succeeded == 1
        assert report.metrics_failed == 1
        assert report.get_score("good") == pytest.approx(0.9)

    def test_failed_metric_with_fail_policy_raises(self):
        """Failed metrics with 'fail' policy should raise RuntimeError."""
        config = ExecutionConfig(on_metric_error="fail")
        executor = ParallelExecutor(config=config)
        executor.add_task(MetricTask(name="bad", fn=make_failing_metric_fn()))

        with pytest.raises(RuntimeError, match="failed"):
            executor.execute(samples=["test"])

    def test_metric_timeout(self):
        """Metrics exceeding timeout should be marked as failed."""
        config = ExecutionConfig(default_timeout=0.1)
        executor = ParallelExecutor(config=config)
        executor.add_task(MetricTask(
            name="slow",
            fn=make_metric_fn(delay=5.0),
            timeout_seconds=0.1,
        ))

        report = executor.execute(samples=["test"])
        assert report.metrics_failed == 1
        assert "timed out" in report.results["slow"].error

    def test_retry_on_failure(self):
        """Metrics should be retried according to retry_count config."""
        call_count = {"n": 0}

        def flaky_fn(samples=None, context=None, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise RuntimeError("Transient error")
            return MetricResult(name="flaky", score=0.9, passed=True)

        config = ExecutionConfig(retry_count=2, retry_delay_seconds=0.01)
        executor = ParallelExecutor(config=config)
        executor.add_task(MetricTask(name="flaky", fn=flaky_fn))

        report = executor.execute(samples=["test"])
        assert report.metrics_succeeded == 1
        assert call_count["n"] == 3

    def test_fail_fast_stops_on_first_failure(self):
        """fail_fast should stop processing after first layer failure."""
        config = ExecutionConfig(fail_fast=True, on_metric_error="skip")
        executor = ParallelExecutor(config=config)

        executor.add_task(MetricTask(name="bad", fn=make_failing_metric_fn()))
        executor.add_task(MetricTask(
            name="dependent",
            fn=make_metric_fn(),
            depends_on=["bad"],
        ))

        report = executor.execute(samples=["test"])
        assert report.metrics_failed >= 1


class TestExecutionReport:
    """ExecutionReport functionality tests."""

    def test_success_rate_calculation(self):
        """success_rate should correctly compute the ratio."""
        report = ExecutionReport(
            results={
                "a": MetricResult(name="a", score=0.9, passed=True),
                "b": MetricResult(name="b", score=0.5, passed=False, error="failed"),
            },
            metrics_executed=2,
            metrics_succeeded=1,
            metrics_failed=1,
        )
        assert report.success_rate == pytest.approx(0.5)

    def test_get_failed_metrics(self):
        """get_failed_metrics should return only failed results."""
        report = ExecutionReport(
            results={
                "good": MetricResult(name="good", score=0.9, passed=True),
                "bad": MetricResult(name="bad", score=0.0, passed=False, error="err"),
            },
        )
        failed = report.get_failed_metrics()
        assert len(failed) == 1
        assert failed[0].name == "bad"

    def test_get_below_threshold(self):
        """get_below_threshold should return metrics that ran but didn't pass."""
        report = ExecutionReport(
            results={
                "high": MetricResult(name="high", score=0.9, passed=True),
                "low": MetricResult(name="low", score=0.4, passed=False),
            },
        )
        below = report.get_below_threshold()
        assert len(below) == 1
        assert below[0].name == "low"

    def test_execution_history_tracked(self):
        """Executor should maintain execution history."""
        executor = ParallelExecutor()
        executor.add_task(MetricTask(name="m", fn=make_metric_fn()))

        executor.execute(samples=[])
        executor.execute(samples=[])

        assert len(executor.execution_history) == 2
