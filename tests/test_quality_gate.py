"""Tests for CI/CD quality gate with configurable thresholds."""

import json
import tempfile
from pathlib import Path

import pytest

from evalforge.gate.quality_gate import (
    GateConfig,
    GateResult,
    GateStatus,
    MetricThreshold,
    QualityGate,
    ThresholdOperator,
    ThresholdResult,
    generate_github_actions_example,
)


@pytest.fixture
def basic_thresholds() -> list[MetricThreshold]:
    """Create basic metric thresholds."""
    return [
        MetricThreshold(
            metric_name="accuracy",
            operator=ThresholdOperator.GREATER_THAN_OR_EQUAL,
            value=0.9,
            warning_value=0.95,
            description="Model accuracy must be >= 90%",
        ),
        MetricThreshold(
            metric_name="latency_p99",
            operator=ThresholdOperator.LESS_THAN,
            value=2.0,
            description="P99 latency must be < 2 seconds",
        ),
        MetricThreshold(
            metric_name="hallucination_rate",
            operator=ThresholdOperator.LESS_THAN_OR_EQUAL,
            value=0.05,
            description="Hallucination rate must be <= 5%",
        ),
    ]


@pytest.fixture
def gate_config(basic_thresholds: list[MetricThreshold]) -> GateConfig:
    """Create a gate configuration."""
    return GateConfig(thresholds=basic_thresholds)


@pytest.fixture
def gate(gate_config: GateConfig) -> QualityGate:
    """Create a quality gate instance."""
    return QualityGate(gate_config)


class TestMetricThreshold:
    """Test individual metric threshold evaluation."""

    def test_gte_passes_when_above(self) -> None:
        """Should pass when value >= threshold."""
        threshold = MetricThreshold(
            metric_name="accuracy",
            operator=ThresholdOperator.GREATER_THAN_OR_EQUAL,
            value=0.9,
        )
        result = threshold.evaluate(0.95)
        assert result.passed is True

    def test_gte_fails_when_below(self) -> None:
        """Should fail when value < threshold."""
        threshold = MetricThreshold(
            metric_name="accuracy",
            operator=ThresholdOperator.GREATER_THAN_OR_EQUAL,
            value=0.9,
        )
        result = threshold.evaluate(0.85)
        assert result.passed is False

    def test_lt_passes_when_below(self) -> None:
        """Should pass when value < threshold."""
        threshold = MetricThreshold(
            metric_name="latency",
            operator=ThresholdOperator.LESS_THAN,
            value=2.0,
        )
        result = threshold.evaluate(1.5)
        assert result.passed is True

    def test_warning_value_triggers_warning(self) -> None:
        """Should set warning when between threshold and warning value."""
        threshold = MetricThreshold(
            metric_name="accuracy",
            operator=ThresholdOperator.GREATER_THAN_OR_EQUAL,
            value=0.9,
            warning_value=0.95,
        )
        result = threshold.evaluate(0.92)  # Passes threshold but below warning
        assert result.passed is True
        assert result.warning is True

    def test_non_required_metric_passes_as_warning(self) -> None:
        """Should pass with warning when non-required metric fails."""
        threshold = MetricThreshold(
            metric_name="optional_metric",
            operator=ThresholdOperator.GREATER_THAN_OR_EQUAL,
            value=0.9,
            required=False,
        )
        result = threshold.evaluate(0.5)
        assert result.passed is True
        assert result.warning is True


class TestQualityGate:
    """Test the quality gate evaluation."""

    def test_all_metrics_pass(self, gate: QualityGate) -> None:
        """Should return PASSED when all metrics meet thresholds."""
        metrics = {
            "accuracy": 0.96,
            "latency_p99": 1.0,
            "hallucination_rate": 0.02,
        }
        result = gate.evaluate(metrics)
        assert result.status == GateStatus.PASSED
        assert result.exit_code == 0
        assert result.passed_count == 3
        assert result.failed_count == 0

    def test_metric_failure_returns_exit_code_1(self, gate: QualityGate) -> None:
        """Should return exit code 1 when a required metric fails."""
        metrics = {
            "accuracy": 0.80,  # Below 0.9 threshold
            "latency_p99": 1.0,
            "hallucination_rate": 0.02,
        }
        result = gate.evaluate(metrics)
        assert result.status == GateStatus.FAILED
        assert result.exit_code == 1
        assert result.failed_count == 1

    def test_missing_metric_fails_by_default(self, gate: QualityGate) -> None:
        """Should fail when a required metric is missing."""
        metrics = {
            "accuracy": 0.96,
            # latency_p99 missing
            "hallucination_rate": 0.02,
        }
        result = gate.evaluate(metrics)
        assert result.status == GateStatus.ERROR
        assert result.exit_code == 1

    def test_warning_does_not_fail_by_default(self, gate: QualityGate) -> None:
        """Should not fail on warnings unless configured."""
        metrics = {
            "accuracy": 0.92,  # Between 0.9 and 0.95 (warning zone)
            "latency_p99": 1.0,
            "hallucination_rate": 0.02,
        }
        result = gate.evaluate(metrics)
        assert result.status == GateStatus.WARNING
        assert result.exit_code == 0

    def test_fail_on_warning_config(self, basic_thresholds: list[MetricThreshold]) -> None:
        """Should fail on warnings when configured."""
        config = GateConfig(thresholds=basic_thresholds, fail_on_warning=True)
        gate = QualityGate(config)

        metrics = {
            "accuracy": 0.92,  # Warning zone
            "latency_p99": 1.0,
            "hallucination_rate": 0.02,
        }
        result = gate.evaluate(metrics)
        assert result.exit_code == 1


class TestGateResult:
    """Test gate result formatting."""

    def test_github_output_format(self, gate: QualityGate) -> None:
        """Should generate GitHub Actions output format."""
        metrics = {"accuracy": 0.96, "latency_p99": 1.0, "hallucination_rate": 0.02}
        result = gate.evaluate(metrics)

        output = result.to_github_output()
        assert "gate_status=passed" in output
        assert "gate_exit_code=0" in output
        assert "metrics_passed=3" in output

    def test_markdown_summary(self, gate: QualityGate) -> None:
        """Should generate markdown summary."""
        metrics = {"accuracy": 0.96, "latency_p99": 1.0, "hallucination_rate": 0.02}
        result = gate.evaluate(metrics)

        md = result.to_markdown_summary()
        assert "Quality Gate" in md
        assert "accuracy" in md
        assert "PASSED" in md


class TestGateConfig:
    """Test gate configuration loading."""

    def test_from_dict(self) -> None:
        """Should create config from dictionary."""
        data = {
            "thresholds": [
                {
                    "metric_name": "accuracy",
                    "operator": "gte",
                    "value": 0.9,
                }
            ],
            "fail_on_warning": True,
        }
        config = GateConfig.from_dict(data)
        assert len(config.thresholds) == 1
        assert config.fail_on_warning is True

    def test_from_file(self) -> None:
        """Should load config from JSON file."""
        data = {
            "thresholds": [
                {"metric_name": "score", "operator": "gte", "value": 0.8}
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            config = GateConfig.from_file(Path(f.name))

        assert len(config.thresholds) == 1
        assert config.thresholds[0].metric_name == "score"


class TestGitHubActionsIntegration:
    """Test GitHub Actions integration helpers."""

    def test_generate_workflow_example(self) -> None:
        """Should generate a valid workflow example."""
        example = generate_github_actions_example()
        assert "evalforge gate" in example
        assert "quality-gate" in example
        assert "actions/checkout" in example
