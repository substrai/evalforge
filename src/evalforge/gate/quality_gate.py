"""CI/CD quality gate with configurable thresholds.

Provides an `evalforge gate` command concept that evaluates metrics
against configurable thresholds and returns exit code 0 (pass) or 1 (fail).
Supports per-metric thresholds and GitHub Actions integration.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ThresholdOperator(str, Enum):
    """Comparison operators for threshold evaluation."""

    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    EQUAL = "eq"


class GateStatus(str, Enum):
    """Overall gate status."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class MetricThreshold:
    """Threshold configuration for a single metric."""

    metric_name: str
    operator: ThresholdOperator
    value: float
    warning_value: Optional[float] = None
    required: bool = True
    description: str = ""

    def evaluate(self, actual_value: float) -> ThresholdResult:
        """Evaluate an actual value against this threshold.

        Args:
            actual_value: The measured metric value.

        Returns:
            ThresholdResult indicating pass/fail/warning.
        """
        passed = self._compare(actual_value, self.value)
        warning = False

        if not passed and not self.required:
            warning = True
            passed = True

        if passed and self.warning_value is not None:
            warning = not self._compare(actual_value, self.warning_value)

        return ThresholdResult(
            metric_name=self.metric_name,
            actual_value=actual_value,
            threshold_value=self.value,
            operator=self.operator,
            passed=passed,
            warning=warning,
            required=self.required,
        )

    def _compare(self, actual: float, threshold: float) -> bool:
        """Compare actual value against threshold using operator."""
        if self.operator == ThresholdOperator.GREATER_THAN:
            return actual > threshold
        elif self.operator == ThresholdOperator.GREATER_THAN_OR_EQUAL:
            return actual >= threshold
        elif self.operator == ThresholdOperator.LESS_THAN:
            return actual < threshold
        elif self.operator == ThresholdOperator.LESS_THAN_OR_EQUAL:
            return actual <= threshold
        elif self.operator == ThresholdOperator.EQUAL:
            return abs(actual - threshold) < 1e-9
        return False


@dataclass
class ThresholdResult:
    """Result of evaluating a single metric threshold."""

    metric_name: str
    actual_value: float
    threshold_value: float
    operator: ThresholdOperator
    passed: bool
    warning: bool = False
    required: bool = True


@dataclass
class GateResult:
    """Complete result of a quality gate evaluation."""

    status: GateStatus
    exit_code: int
    results: list[ThresholdResult] = field(default_factory=list)
    summary: str = ""
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed_count(self) -> int:
        """Number of metrics that passed."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        """Number of metrics that failed."""
        return sum(1 for r in self.results if not r.passed)

    @property
    def warning_count(self) -> int:
        """Number of metrics with warnings."""
        return sum(1 for r in self.results if r.warning)

    def to_github_output(self) -> str:
        """Format result for GitHub Actions output."""
        lines = [
            f"gate_status={self.status.value}",
            f"gate_exit_code={self.exit_code}",
            f"metrics_passed={self.passed_count}",
            f"metrics_failed={self.failed_count}",
            f"metrics_warnings={self.warning_count}",
        ]
        return "\n".join(lines)

    def to_markdown_summary(self) -> str:
        """Generate a markdown summary for GitHub Actions step summary."""
        status_emoji = {
            GateStatus.PASSED: ":white_check_mark:",
            GateStatus.FAILED: ":x:",
            GateStatus.WARNING: ":warning:",
            GateStatus.ERROR: ":rotating_light:",
        }
        emoji = status_emoji.get(self.status, ":question:")

        lines = [
            f"## {emoji} Quality Gate: {self.status.value.upper()}",
            "",
            f"| Metric | Value | Threshold | Status |",
            f"|--------|-------|-----------|--------|",
        ]

        for r in self.results:
            status = ":white_check_mark:" if r.passed else ":x:"
            if r.warning:
                status = ":warning:"
            lines.append(
                f"| {r.metric_name} | {r.actual_value:.4f} | "
                f"{r.operator.value} {r.threshold_value:.4f} | {status} |"
            )

        lines.extend([
            "",
            f"**Summary:** {self.passed_count} passed, "
            f"{self.failed_count} failed, {self.warning_count} warnings",
        ])

        return "\n".join(lines)


@dataclass
class GateConfig:
    """Configuration for the quality gate."""

    thresholds: list[MetricThreshold] = field(default_factory=list)
    fail_on_warning: bool = False
    fail_on_missing_metric: bool = True
    output_format: str = "text"  # text, json, markdown
    github_actions: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GateConfig:
        """Create config from a dictionary."""
        thresholds = []
        for t in data.get("thresholds", []):
            thresholds.append(MetricThreshold(
                metric_name=t["metric_name"],
                operator=ThresholdOperator(t["operator"]),
                value=t["value"],
                warning_value=t.get("warning_value"),
                required=t.get("required", True),
                description=t.get("description", ""),
            ))

        return cls(
            thresholds=thresholds,
            fail_on_warning=data.get("fail_on_warning", False),
            fail_on_missing_metric=data.get("fail_on_missing_metric", True),
            output_format=data.get("output_format", "text"),
            github_actions=data.get("github_actions", False),
        )

    @classmethod
    def from_file(cls, path: Path) -> GateConfig:
        """Load config from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


class QualityGate:
    """CI/CD quality gate that evaluates metrics against thresholds.

    Designed to be used as an `evalforge gate` CLI command that returns
    exit code 0 (all metrics pass) or 1 (any required metric fails).

    Example:
        >>> gate = QualityGate(config)
        >>> result = gate.evaluate({"accuracy": 0.95, "latency_p99": 1.2})
        >>> sys.exit(result.exit_code)

    GitHub Actions integration:
        ```yaml
        - name: Quality Gate
          run: evalforge gate --config gate.json --metrics results.json
        ```
    """

    def __init__(self, config: GateConfig) -> None:
        self._config = config

    @property
    def config(self) -> GateConfig:
        """Gate configuration."""
        return self._config

    def evaluate(self, metrics: dict[str, float]) -> GateResult:
        """Evaluate metrics against configured thresholds.

        Args:
            metrics: Dictionary of metric_name -> measured_value.

        Returns:
            GateResult with overall status and per-metric results.
        """
        start_time = time.time()
        results: list[ThresholdResult] = []
        has_failure = False
        has_warning = False
        has_error = False

        for threshold in self._config.thresholds:
            if threshold.metric_name not in metrics:
                if self._config.fail_on_missing_metric and threshold.required:
                    results.append(ThresholdResult(
                        metric_name=threshold.metric_name,
                        actual_value=0.0,
                        threshold_value=threshold.value,
                        operator=threshold.operator,
                        passed=False,
                        required=threshold.required,
                    ))
                    has_error = True
                continue

            actual_value = metrics[threshold.metric_name]
            result = threshold.evaluate(actual_value)
            results.append(result)

            if not result.passed:
                has_failure = True
            if result.warning:
                has_warning = True

        # Determine overall status
        if has_error:
            status = GateStatus.ERROR
        elif has_failure:
            status = GateStatus.FAILED
        elif has_warning and self._config.fail_on_warning:
            status = GateStatus.FAILED
        elif has_warning:
            status = GateStatus.WARNING
        else:
            status = GateStatus.PASSED

        exit_code = 0 if status in (GateStatus.PASSED, GateStatus.WARNING) else 1
        if self._config.fail_on_warning and status == GateStatus.WARNING:
            exit_code = 1

        duration_ms = (time.time() - start_time) * 1000

        gate_result = GateResult(
            status=status,
            exit_code=exit_code,
            results=results,
            summary=self._build_summary(results),
            duration_ms=duration_ms,
        )

        return gate_result

    def evaluate_from_file(self, metrics_path: Path) -> GateResult:
        """Evaluate metrics loaded from a JSON file.

        Args:
            metrics_path: Path to JSON file with metric values.

        Returns:
            GateResult with evaluation results.
        """
        with open(metrics_path) as f:
            metrics = json.load(f)
        return self.evaluate(metrics)

    def _build_summary(self, results: list[ThresholdResult]) -> str:
        """Build a human-readable summary."""
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        total = len(results)
        return f"{passed}/{total} metrics passed, {failed} failed"


def generate_github_actions_example() -> str:
    """Generate an example GitHub Actions workflow snippet."""
    return """# .github/workflows/quality-gate.yml
name: EvalForge Quality Gate

on:
  pull_request:
    branches: [main]

jobs:
  quality-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Evaluations
        run: evalforge run --config eval.json --output results.json

      - name: Quality Gate Check
        id: gate
        run: |
          evalforge gate --config gate.json --metrics results.json --format github
          echo "status=$?" >> $GITHUB_OUTPUT

      - name: Comment PR
        if: always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const summary = fs.readFileSync('gate-summary.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: summary
            });
"""
