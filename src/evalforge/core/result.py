"""Evaluation result models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MetricScore:
    """Score for a single metric."""

    name: str
    score: float
    threshold: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    samples_evaluated: int = 0
    latency_ms: float = 0.0

    @property
    def margin(self) -> float:
        """How far above/below threshold."""
        return self.score - self.threshold

    def summary_line(self) -> str:
        icon = "✓" if self.passed else "✗"
        return f"  {icon} {self.name}: {self.score:.4f} (threshold: {self.threshold})"


@dataclass
class EvalResult:
    """Complete evaluation pipeline result."""

    project_name: str
    use_case_type: str
    timestamp: float = field(default_factory=time.time)
    scores: List[MetricScore] = field(default_factory=list)
    total_samples: int = 0
    total_latency_ms: float = 0.0
    model: str = ""
    environment: str = "dev"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_passing(self) -> bool:
        """Check if all metrics pass their thresholds."""
        return all(s.passed for s in self.scores)

    @property
    def pass_count(self) -> int:
        return sum(1 for s in self.scores if s.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for s in self.scores if not s.passed)

    @property
    def overall_score(self) -> float:
        """Average score across all metrics."""
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)

    @property
    def pass_rate(self) -> float:
        """Percentage of metrics passing."""
        if not self.scores:
            return 0.0
        return self.pass_count / len(self.scores)

    def get_score(self, metric_name: str) -> Optional[MetricScore]:
        """Get score for a specific metric."""
        for s in self.scores:
            if s.name == metric_name:
                return s
        return None

    def failing_metrics(self) -> List[MetricScore]:
        """Get metrics that failed their thresholds."""
        return [s for s in self.scores if not s.passed]

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "PASS" if self.all_passing else "FAIL"
        lines = [
            f"EvalForge Results: {self.project_name}",
            f"Status: {status} ({self.pass_count}/{len(self.scores)} metrics passing)",
            f"Use case: {self.use_case_type}",
            f"Model: {self.model}",
            f"Samples: {self.total_samples}",
            f"Duration: {self.total_latency_ms:.0f}ms",
            "",
            "Metrics:",
        ]
        for score in self.scores:
            lines.append(score.summary_line())

        if not self.all_passing:
            lines.append("")
            lines.append("Failing metrics:")
            for s in self.failing_metrics():
                lines.append(f"  • {s.name}: {s.score:.4f} < {s.threshold} (gap: {s.margin:.4f})")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "project_name": self.project_name,
            "use_case_type": self.use_case_type,
            "timestamp": self.timestamp,
            "all_passing": self.all_passing,
            "overall_score": self.overall_score,
            "pass_rate": self.pass_rate,
            "total_samples": self.total_samples,
            "total_latency_ms": self.total_latency_ms,
            "model": self.model,
            "environment": self.environment,
            "scores": [
                {"name": s.name, "score": s.score, "threshold": s.threshold, "passed": s.passed}
                for s in self.scores
            ],
        }
