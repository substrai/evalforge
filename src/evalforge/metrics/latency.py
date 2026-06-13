"""Response latency metric with percentile tracking.

Measure and score model response times, flag slow responses.
Percentile tracking (p50/p90/p95/p99), SLA compliance scoring.

Features:
- Response time measurement and scoring
- Percentile calculation (p50, p90, p95, p99)
- SLA compliance scoring against configurable thresholds
- Slow response flagging with severity classification
- Historical latency tracking with sliding window
- Latency budget allocation for multi-step pipelines
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class LatencySeverity(Enum):
    """Severity classification for latency violations."""

    ACCEPTABLE = "acceptable"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    TIMEOUT = "timeout"


@dataclass
class LatencySample:
    """A single latency measurement sample.

    Attributes:
        duration_ms: Response time in milliseconds.
        timestamp: When the measurement was taken (unix timestamp).
        model_id: Which model produced this response.
        step_name: Which pipeline step this belongs to.
        metadata: Additional measurement metadata.
    """

    duration_ms: float
    timestamp: float = 0.0
    model_id: str = ""
    step_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class PercentileResult:
    """Percentile calculation results.

    Attributes:
        p50: 50th percentile (median) in milliseconds.
        p90: 90th percentile in milliseconds.
        p95: 95th percentile in milliseconds.
        p99: 99th percentile in milliseconds.
        min_ms: Minimum observed latency.
        max_ms: Maximum observed latency.
        mean_ms: Mean latency.
        std_dev_ms: Standard deviation of latencies.
        sample_count: Number of samples used.
    """

    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    std_dev_ms: float = 0.0
    sample_count: int = 0


@dataclass
class SLAThreshold:
    """SLA threshold configuration for latency compliance.

    Attributes:
        target_ms: Target latency in milliseconds.
        warning_ms: Warning threshold (above target but below critical).
        critical_ms: Critical threshold (service degradation).
        timeout_ms: Timeout threshold (request failure).
        target_percentile: Which percentile must meet the target (e.g., 0.95 = p95).
        compliance_target: Required compliance percentage (0.0 to 1.0).
    """

    target_ms: float = 500.0
    warning_ms: float = 1000.0
    critical_ms: float = 3000.0
    timeout_ms: float = 10000.0
    target_percentile: float = 0.95
    compliance_target: float = 0.99


@dataclass
class SLAComplianceResult:
    """SLA compliance scoring result.

    Attributes:
        compliant: Whether overall SLA is met.
        compliance_percentage: Percentage of requests within target.
        score: Compliance score from 0.0 to 1.0.
        violations_count: Number of SLA violations.
        severity: Overall severity classification.
        details: Detailed breakdown of compliance.
    """

    compliant: bool = True
    compliance_percentage: float = 100.0
    score: float = 1.0
    violations_count: int = 0
    severity: LatencySeverity = LatencySeverity.ACCEPTABLE
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyMetricResult:
    """Complete latency metric evaluation result.

    Attributes:
        score: Overall latency score (0.0 to 1.0).
        passed: Whether the metric evaluation passed.
        percentiles: Percentile statistics.
        sla_compliance: SLA compliance result.
        slow_responses: List of flagged slow responses.
        severity: Overall severity classification.
        recommendation: Improvement recommendation.
    """

    score: float = 1.0
    passed: bool = True
    percentiles: PercentileResult = field(default_factory=PercentileResult)
    sla_compliance: SLAComplianceResult = field(default_factory=SLAComplianceResult)
    slow_responses: List[LatencySample] = field(default_factory=list)
    severity: LatencySeverity = LatencySeverity.ACCEPTABLE
    recommendation: str = ""


class LatencyMetric:
    """Response latency metric with percentile tracking and SLA scoring.

    Tracks response latencies, computes percentiles, and scores
    against configurable SLA thresholds.

    Args:
        sla: SLA threshold configuration.
        window_size: Maximum number of samples to keep in the sliding window.

    Example:
        metric = LatencyMetric(
            sla=SLAThreshold(target_ms=500, critical_ms=2000)
        )
        metric.record(LatencySample(duration_ms=150))
        metric.record(LatencySample(duration_ms=320))
        result = metric.evaluate()
        print(f"Score: {result.score}, P95: {result.percentiles.p95}ms")
    """

    def __init__(
        self,
        sla: Optional[SLAThreshold] = None,
        window_size: int = 1000,
    ):
        self.sla = sla or SLAThreshold()
        self.window_size = window_size
        self._samples: List[LatencySample] = []

    @property
    def sample_count(self) -> int:
        """Number of recorded samples."""
        return len(self._samples)

    def record(self, sample: LatencySample) -> None:
        """Record a latency sample.

        Args:
            sample: The latency measurement to record.
        """
        self._samples.append(sample)
        # Maintain sliding window
        if len(self._samples) > self.window_size:
            self._samples = self._samples[-self.window_size:]

    def record_duration(
        self,
        duration_ms: float,
        model_id: str = "",
        step_name: str = "",
    ) -> LatencySample:
        """Convenience method to record a duration directly.

        Args:
            duration_ms: Response time in milliseconds.
            model_id: Optional model identifier.
            step_name: Optional step name.

        Returns:
            The created LatencySample.
        """
        sample = LatencySample(
            duration_ms=duration_ms,
            model_id=model_id,
            step_name=step_name,
        )
        self.record(sample)
        return sample

    def clear(self) -> None:
        """Clear all recorded samples."""
        self._samples.clear()

    def get_percentiles(self) -> PercentileResult:
        """Calculate percentile statistics from recorded samples.

        Returns:
            PercentileResult with p50/p90/p95/p99 and other stats.
        """
        if not self._samples:
            return PercentileResult()

        durations = sorted(s.duration_ms for s in self._samples)
        n = len(durations)

        return PercentileResult(
            p50=self._percentile(durations, 0.50),
            p90=self._percentile(durations, 0.90),
            p95=self._percentile(durations, 0.95),
            p99=self._percentile(durations, 0.99),
            min_ms=durations[0],
            max_ms=durations[-1],
            mean_ms=sum(durations) / n,
            std_dev_ms=self._std_dev(durations),
            sample_count=n,
        )

    def check_sla_compliance(self) -> SLAComplianceResult:
        """Check SLA compliance against configured thresholds.

        Returns:
            SLAComplianceResult with compliance percentage and score.
        """
        if not self._samples:
            return SLAComplianceResult()

        durations = [s.duration_ms for s in self._samples]
        n = len(durations)

        # Count violations at each severity level
        within_target = sum(1 for d in durations if d <= self.sla.target_ms)
        within_warning = sum(1 for d in durations if d <= self.sla.warning_ms)
        within_critical = sum(1 for d in durations if d <= self.sla.critical_ms)
        timeouts = sum(1 for d in durations if d > self.sla.timeout_ms)

        compliance_pct = within_target / n
        violations = n - within_target

        # Determine severity
        if compliance_pct >= self.sla.compliance_target:
            severity = LatencySeverity.ACCEPTABLE
        elif compliance_pct >= 0.95:
            severity = LatencySeverity.WARNING
        elif compliance_pct >= 0.90:
            severity = LatencySeverity.DEGRADED
        elif timeouts > 0:
            severity = LatencySeverity.TIMEOUT
        else:
            severity = LatencySeverity.CRITICAL

        # Calculate score
        score = min(1.0, compliance_pct / self.sla.compliance_target)

        return SLAComplianceResult(
            compliant=compliance_pct >= self.sla.compliance_target,
            compliance_percentage=compliance_pct * 100,
            score=score,
            violations_count=violations,
            severity=severity,
            details={
                "within_target": within_target,
                "within_warning": within_warning,
                "within_critical": within_critical,
                "timeouts": timeouts,
                "total_samples": n,
            },
        )

    def get_slow_responses(self) -> List[LatencySample]:
        """Get all responses that exceeded the warning threshold.

        Returns:
            List of slow LatencySamples.
        """
        return [
            s for s in self._samples
            if s.duration_ms > self.sla.warning_ms
        ]

    def evaluate(self) -> LatencyMetricResult:
        """Run full latency evaluation.

        Computes percentiles, checks SLA compliance, flags slow responses,
        and produces an overall score.

        Returns:
            LatencyMetricResult with comprehensive evaluation.
        """
        if not self._samples:
            return LatencyMetricResult(
                score=1.0,
                passed=True,
                recommendation="No samples recorded yet.",
            )

        percentiles = self.get_percentiles()
        sla_compliance = self.check_sla_compliance()
        slow_responses = self.get_slow_responses()

        # Overall score combines SLA compliance and percentile performance
        target_percentile_value = self._percentile(
            sorted(s.duration_ms for s in self._samples),
            self.sla.target_percentile,
        )

        # Score based on how target percentile compares to SLA target
        if target_percentile_value <= self.sla.target_ms:
            percentile_score = 1.0
        elif target_percentile_value <= self.sla.warning_ms:
            percentile_score = 0.8
        elif target_percentile_value <= self.sla.critical_ms:
            percentile_score = 0.5
        else:
            percentile_score = 0.2

        overall_score = (sla_compliance.score * 0.6) + (percentile_score * 0.4)
        passed = overall_score >= 0.8 and sla_compliance.compliant

        # Generate recommendation
        recommendation = self._generate_recommendation(
            percentiles, sla_compliance, slow_responses
        )

        return LatencyMetricResult(
            score=overall_score,
            passed=passed,
            percentiles=percentiles,
            sla_compliance=sla_compliance,
            slow_responses=slow_responses,
            severity=sla_compliance.severity,
            recommendation=recommendation,
        )

    def classify_severity(self, duration_ms: float) -> LatencySeverity:
        """Classify a single duration into severity levels.

        Args:
            duration_ms: Response duration in milliseconds.

        Returns:
            LatencySeverity classification.
        """
        if duration_ms <= self.sla.target_ms:
            return LatencySeverity.ACCEPTABLE
        elif duration_ms <= self.sla.warning_ms:
            return LatencySeverity.WARNING
        elif duration_ms <= self.sla.critical_ms:
            return LatencySeverity.DEGRADED
        elif duration_ms <= self.sla.timeout_ms:
            return LatencySeverity.CRITICAL
        else:
            return LatencySeverity.TIMEOUT

    def _percentile(self, sorted_values: List[float], p: float) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        n = len(sorted_values)
        if n == 1:
            return sorted_values[0]
        idx = p * (n - 1)
        lower = int(math.floor(idx))
        upper = int(math.ceil(idx))
        if lower == upper:
            return sorted_values[lower]
        fraction = idx - lower
        return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction

    def _std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def _generate_recommendation(
        self,
        percentiles: PercentileResult,
        sla: SLAComplianceResult,
        slow_responses: List[LatencySample],
    ) -> str:
        """Generate improvement recommendation based on results."""
        if sla.compliant and not slow_responses:
            return "Latency performance is within acceptable bounds."

        recommendations = []

        if not sla.compliant:
            recommendations.append(
                f"SLA compliance is at {sla.compliance_percentage:.1f}% "
                f"(target: {self.sla.compliance_target * 100:.0f}%). "
                f"Consider optimizing slow paths or scaling infrastructure."
            )

        if slow_responses:
            slow_models = set(s.model_id for s in slow_responses if s.model_id)
            if slow_models:
                recommendations.append(
                    f"Models with slow responses: {', '.join(slow_models)}. "
                    f"Consider model optimization or switching to faster alternatives."
                )

        if percentiles.p99 > self.sla.critical_ms:
            recommendations.append(
                f"P99 latency ({percentiles.p99:.0f}ms) exceeds critical threshold "
                f"({self.sla.critical_ms:.0f}ms). Investigate outlier causes."
            )

        return " ".join(recommendations) if recommendations else "Performance within bounds."
