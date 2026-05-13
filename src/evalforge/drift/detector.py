"""Drift detection engine.

Detects statistically significant quality degradation by comparing
recent evaluation results against established baselines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from evalforge.drift.baseline import Baseline, BaselineManager


class DriftSeverity(Enum):
    """Severity of detected drift."""

    NONE = "none"
    LOW = "low"          # within 1-2 std devs
    MEDIUM = "medium"    # 2-3 std devs
    HIGH = "high"        # >3 std devs
    CRITICAL = "critical"  # sustained degradation


@dataclass
class DriftResult:
    """Result of a drift detection check."""

    metric_name: str
    severity: DriftSeverity
    baseline_mean: float
    current_value: float
    deviation_sigma: float  # standard deviations from mean
    drift_percent: float
    is_degradation: bool
    timestamp: float = field(default_factory=time.time)
    message: str = ""

    @property
    def needs_alert(self) -> bool:
        return self.severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL)

    @property
    def needs_action(self) -> bool:
        return self.severity == DriftSeverity.CRITICAL

    def summary(self) -> str:
        if self.severity == DriftSeverity.NONE:
            return f"  ✓ {self.metric_name}: {self.current_value:.4f} (baseline: {self.baseline_mean:.4f}) — stable"
        icon = {"low": "⚠", "medium": "⚠️", "high": "🔴", "critical": "🚨"}
        direction = "↓" if self.drift_percent < 0 else "↑"
        return (
            f"  {icon.get(self.severity.value, '•')} {self.metric_name}: "
            f"{self.current_value:.4f} {direction}{abs(self.drift_percent):.1f}% "
            f"(baseline: {self.baseline_mean:.4f}, {self.deviation_sigma:.1f}σ) — {self.severity.value}"
        )


class DriftDetector:
    """Detects quality drift by comparing against baselines.

    Usage:
        detector = DriftDetector(baseline_manager)
        results = detector.check({"faithfulness": 0.72, "toxicity": 0.08})
        for r in results:
            if r.needs_alert:
                send_alert(r)
    """

    def __init__(
        self,
        baseline_manager: Optional[BaselineManager] = None,
        sensitivity: str = "medium",  # low, medium, high
    ):
        self.baseline = baseline_manager or BaselineManager()
        self.sensitivity = sensitivity
        self._alert_history: List[DriftResult] = []

        # Sigma thresholds per sensitivity
        self._thresholds = {
            "low": {"low": 2.0, "medium": 3.0, "high": 4.0, "critical": 5.0},
            "medium": {"low": 1.5, "medium": 2.0, "high": 3.0, "critical": 4.0},
            "high": {"low": 1.0, "medium": 1.5, "high": 2.0, "critical": 3.0},
        }

    def check(
        self,
        current_scores: Dict[str, float],
        safety_metrics: Optional[List[str]] = None,
    ) -> List[DriftResult]:
        """Check current scores against baselines for drift.

        Args:
            current_scores: Dict of metric_name -> current value
            safety_metrics: Metrics where higher = worse (e.g., toxicity)

        Returns:
            List of DriftResults for each metric
        """
        safety = safety_metrics or ["toxicity", "bias_detection"]
        results = []

        for metric_name, current_value in current_scores.items():
            baseline = self.baseline.get_baseline(metric_name)
            if not baseline:
                results.append(DriftResult(
                    metric_name=metric_name,
                    severity=DriftSeverity.NONE,
                    baseline_mean=0.0,
                    current_value=current_value,
                    deviation_sigma=0.0,
                    drift_percent=0.0,
                    is_degradation=False,
                    message="Insufficient baseline data",
                ))
                continue

            # Calculate deviation
            sigma = baseline.deviation_from_mean(current_value)
            drift_pct = ((current_value - baseline.mean) / baseline.mean * 100) if baseline.mean != 0 else 0

            # Determine if this is degradation
            is_safety = metric_name in safety
            if is_safety:
                # For safety metrics, increase = degradation
                is_degradation = current_value > baseline.mean
                sigma_for_severity = sigma  # positive sigma = worse
            else:
                # For quality metrics, decrease = degradation
                is_degradation = current_value < baseline.mean
                sigma_for_severity = -sigma  # negative sigma = worse

            # Determine severity
            severity = self._classify_severity(abs(sigma_for_severity) if is_degradation else 0)

            result = DriftResult(
                metric_name=metric_name,
                severity=severity,
                baseline_mean=baseline.mean,
                current_value=current_value,
                deviation_sigma=round(sigma, 2),
                drift_percent=round(drift_pct, 2),
                is_degradation=is_degradation,
                message=f"{metric_name} {'degraded' if is_degradation else 'stable'}",
            )
            results.append(result)

            if result.needs_alert:
                self._alert_history.append(result)

        return results

    def check_and_record(
        self,
        current_scores: Dict[str, float],
        safety_metrics: Optional[List[str]] = None,
    ) -> List[DriftResult]:
        """Check for drift AND record scores to baseline."""
        results = self.check(current_scores, safety_metrics)
        self.baseline.record_batch(current_scores)
        return results

    def get_alert_history(self) -> List[DriftResult]:
        """Get history of drift alerts."""
        return self._alert_history

    def get_degraded_metrics(self, current_scores: Dict[str, float]) -> List[str]:
        """Get list of metrics currently degraded."""
        results = self.check(current_scores)
        return [r.metric_name for r in results if r.is_degradation and r.severity != DriftSeverity.NONE]

    def should_rollback(self, current_scores: Dict[str, float]) -> bool:
        """Determine if quality is bad enough to trigger rollback."""
        results = self.check(current_scores)
        critical = [r for r in results if r.severity == DriftSeverity.CRITICAL]
        return len(critical) > 0

    def summary(self, current_scores: Dict[str, float]) -> str:
        """Generate drift detection summary."""
        results = self.check(current_scores)
        lines = ["Drift Detection Report:", ""]

        degraded = [r for r in results if r.is_degradation and r.severity != DriftSeverity.NONE]
        stable = [r for r in results if not r.is_degradation or r.severity == DriftSeverity.NONE]

        if degraded:
            lines.append(f"⚠️  {len(degraded)} metric(s) degraded:")
            for r in degraded:
                lines.append(r.summary())
        if stable:
            lines.append(f"\n✓ {len(stable)} metric(s) stable:")
            for r in stable:
                lines.append(r.summary())

        return "\n".join(lines)

    def _classify_severity(self, abs_sigma: float) -> DriftSeverity:
        """Classify severity based on sigma deviation."""
        thresholds = self._thresholds.get(self.sensitivity, self._thresholds["medium"])
        if abs_sigma >= thresholds["critical"]:
            return DriftSeverity.CRITICAL
        elif abs_sigma >= thresholds["high"]:
            return DriftSeverity.HIGH
        elif abs_sigma >= thresholds["medium"]:
            return DriftSeverity.MEDIUM
        elif abs_sigma >= thresholds["low"]:
            return DriftSeverity.LOW
        return DriftSeverity.NONE
