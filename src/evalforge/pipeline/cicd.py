"""CI/CD integration - quality gates for deployment pipelines."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from evalforge.core.result import EvalResult


class GateDecision(Enum):
    """CI/CD gate decision."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass
class CICDResult:
    """Result of a CI/CD quality gate check."""

    decision: GateDecision
    eval_result: EvalResult
    message: str
    failing_metrics: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    exit_code: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def should_block_deploy(self) -> bool:
        return self.decision == GateDecision.FAIL

    def summary(self) -> str:
        icon = {"pass": "✓", "fail": "✗", "warn": "⚠"}[self.decision.value]
        lines = [
            f"{icon} CI/CD Quality Gate: {self.decision.value.upper()}",
            f"  {self.message}",
            f"  Overall score: {self.eval_result.overall_score:.4f}",
            f"  Pass rate: {self.eval_result.pass_rate:.0%}",
        ]
        if self.failing_metrics:
            lines.append(f"  Failing: {', '.join(self.failing_metrics)}")
        if self.warnings:
            lines.append(f"  Warnings: {', '.join(self.warnings)}")
        return "\n".join(lines)

    def to_github_output(self) -> str:
        """Format for GitHub Actions output."""
        lines = [
            f"::{'error' if self.should_block_deploy else 'notice'}::EvalForge Quality Gate: {self.decision.value.upper()}",
        ]
        for metric in self.failing_metrics:
            lines.append(f"::error::Metric failed: {metric}")
        for warning in self.warnings:
            lines.append(f"::warning::{warning}")
        return "\n".join(lines)


class CICDIntegration:
    """Integrates EvalForge with CI/CD pipelines.

    Runs evaluation and returns pass/fail decision for deployment gates.

    Usage:
        cicd = CICDIntegration(min_pass_rate=0.95)
        result = cicd.check(eval_result)
        if result.should_block_deploy:
            sys.exit(1)
    """

    def __init__(
        self,
        min_pass_rate: float = 0.95,
        required_metrics: Optional[List[str]] = None,
        warn_on_degradation: bool = True,
        previous_result: Optional[EvalResult] = None,
    ):
        """Initialize CI/CD integration.

        Args:
            min_pass_rate: Minimum metric pass rate to allow deploy
            required_metrics: Metrics that MUST pass (fail if any fail)
            warn_on_degradation: Warn if scores dropped from previous run
            previous_result: Previous evaluation result for comparison
        """
        self.min_pass_rate = min_pass_rate
        self.required_metrics = required_metrics or []
        self.warn_on_degradation = warn_on_degradation
        self.previous_result = previous_result

    def check(self, eval_result: EvalResult) -> CICDResult:
        """Run quality gate check.

        Args:
            eval_result: Current evaluation result

        Returns:
            CICDResult with pass/fail decision
        """
        failing_metrics = []
        warnings = []

        # Check required metrics
        for metric_name in self.required_metrics:
            score = eval_result.get_score(metric_name)
            if score and not score.passed:
                failing_metrics.append(f"{metric_name} ({score.score:.4f} < {score.threshold})")

        # Check overall pass rate
        if eval_result.pass_rate < self.min_pass_rate:
            failing_metrics.append(
                f"pass_rate ({eval_result.pass_rate:.0%} < {self.min_pass_rate:.0%})"
            )

        # Check for degradation vs previous
        if self.warn_on_degradation and self.previous_result:
            for score in eval_result.scores:
                prev_score = self.previous_result.get_score(score.name)
                if prev_score and score.score < prev_score.score - 0.05:
                    warnings.append(
                        f"{score.name} degraded: {prev_score.score:.4f} → {score.score:.4f}"
                    )

        # Determine decision
        if failing_metrics:
            decision = GateDecision.FAIL
            message = f"Quality gate FAILED: {len(failing_metrics)} issue(s)"
            exit_code = 1
        elif warnings:
            decision = GateDecision.WARN
            message = f"Quality gate PASSED with {len(warnings)} warning(s)"
            exit_code = 0
        else:
            decision = GateDecision.PASS
            message = "Quality gate PASSED — all metrics within thresholds"
            exit_code = 0

        return CICDResult(
            decision=decision,
            eval_result=eval_result,
            message=message,
            failing_metrics=failing_metrics,
            warnings=warnings,
            exit_code=exit_code,
        )

    def check_and_exit(self, eval_result: EvalResult) -> None:
        """Check and exit with appropriate code (for CLI usage)."""
        import sys
        result = self.check(eval_result)
        print(result.summary())
        sys.exit(result.exit_code)
