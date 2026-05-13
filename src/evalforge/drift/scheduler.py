"""Evaluation scheduler for continuous monitoring.

Manages scheduled evaluation runs with history tracking
and trend analysis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from evalforge.core.result import EvalResult


@dataclass
class ScheduleRun:
    """Record of a scheduled evaluation run."""

    run_id: str
    timestamp: float
    result: Optional[EvalResult] = None
    status: str = "pending"  # pending, running, completed, failed
    duration_ms: float = 0.0
    error: Optional[str] = None
    triggered_by: str = "schedule"  # schedule, manual, ci_cd, drift_alert

    @property
    def passed(self) -> bool:
        return self.result.all_passing if self.result else False

    def summary_line(self) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.timestamp))
        status_icon = {"completed": "✓", "failed": "✗", "running": "⟳", "pending": "○"}
        icon = status_icon.get(self.status, "?")
        pass_info = ""
        if self.result:
            pass_info = f" — {'PASS' if self.passed else 'FAIL'} ({self.result.pass_count}/{len(self.result.scores)})"
        return f"  {icon} [{ts}] {self.run_id} ({self.status}){pass_info}"


class EvalScheduler:
    """Manages scheduled evaluation runs.

    Tracks run history, detects trends, and supports
    different trigger types (schedule, CI/CD, manual).

    Usage:
        scheduler = EvalScheduler()
        run = scheduler.start_run(triggered_by="schedule")
        # ... execute pipeline ...
        scheduler.complete_run(run.run_id, result)
        trend = scheduler.get_trend("faithfulness", last_n=10)
    """

    def __init__(self, max_history: int = 1000):
        self._runs: List[ScheduleRun] = []
        self._max_history = max_history
        self._run_counter = 0

    def start_run(self, triggered_by: str = "schedule") -> ScheduleRun:
        """Start a new evaluation run."""
        self._run_counter += 1
        run = ScheduleRun(
            run_id=f"run-{self._run_counter:04d}",
            timestamp=time.time(),
            status="running",
            triggered_by=triggered_by,
        )
        self._runs.append(run)
        return run

    def complete_run(self, run_id: str, result: EvalResult) -> None:
        """Mark a run as completed with results."""
        for run in self._runs:
            if run.run_id == run_id:
                run.status = "completed"
                run.result = result
                run.duration_ms = result.total_latency_ms
                break

    def fail_run(self, run_id: str, error: str) -> None:
        """Mark a run as failed."""
        for run in self._runs:
            if run.run_id == run_id:
                run.status = "failed"
                run.error = error
                break

    def get_run(self, run_id: str) -> Optional[ScheduleRun]:
        """Get a specific run by ID."""
        for run in self._runs:
            if run.run_id == run_id:
                return run
        return None

    def get_history(self, last_n: Optional[int] = None, status: Optional[str] = None) -> List[ScheduleRun]:
        """Get run history with optional filters."""
        runs = self._runs
        if status:
            runs = [r for r in runs if r.status == status]
        if last_n:
            runs = runs[-last_n:]
        return runs

    def get_trend(self, metric_name: str, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get metric trend over recent runs.

        Returns list of {timestamp, value, passed} dicts.
        """
        completed = [r for r in self._runs if r.status == "completed" and r.result]
        recent = completed[-last_n:]

        trend = []
        for run in recent:
            score = run.result.get_score(metric_name)
            if score:
                trend.append({
                    "timestamp": run.timestamp,
                    "value": score.score,
                    "threshold": score.threshold,
                    "passed": score.passed,
                    "run_id": run.run_id,
                })
        return trend

    def get_pass_rate_trend(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get overall pass rate trend."""
        completed = [r for r in self._runs if r.status == "completed" and r.result]
        recent = completed[-last_n:]

        return [
            {
                "timestamp": run.timestamp,
                "pass_rate": run.result.pass_rate,
                "all_passing": run.result.all_passing,
                "run_id": run.run_id,
            }
            for run in recent
        ]

    def detect_regression(self, last_n: int = 5) -> Optional[str]:
        """Detect if recent runs show a regression pattern.

        Returns description of regression, or None if stable.
        """
        completed = [r for r in self._runs if r.status == "completed" and r.result]
        if len(completed) < last_n:
            return None

        recent = completed[-last_n:]
        pass_rates = [r.result.pass_rate for r in recent]

        # Check for declining trend
        if len(pass_rates) >= 3:
            declining = all(pass_rates[i] <= pass_rates[i-1] for i in range(1, len(pass_rates)))
            if declining and pass_rates[-1] < pass_rates[0] - 0.1:
                return f"Declining pass rate: {pass_rates[0]:.0%} → {pass_rates[-1]:.0%} over last {last_n} runs"

        # Check for consecutive failures
        recent_failures = sum(1 for r in recent if not r.result.all_passing)
        if recent_failures >= 3:
            return f"{recent_failures}/{last_n} recent runs failed"

        return None

    @property
    def total_runs(self) -> int:
        return len(self._runs)

    @property
    def success_rate(self) -> float:
        completed = [r for r in self._runs if r.status == "completed" and r.result]
        if not completed:
            return 0.0
        passed = sum(1 for r in completed if r.result.all_passing)
        return passed / len(completed)

    def summary(self) -> str:
        """Generate scheduler summary."""
        completed = [r for r in self._runs if r.status == "completed"]
        failed = [r for r in self._runs if r.status == "failed"]

        lines = [
            "Evaluation Schedule Summary:",
            f"  Total runs: {self.total_runs}",
            f"  Completed: {len(completed)}",
            f"  Failed: {len(failed)}",
            f"  Success rate: {self.success_rate:.0%}",
        ]

        regression = self.detect_regression()
        if regression:
            lines.append(f"  ⚠️  Regression detected: {regression}")

        lines.append("\nRecent runs:")
        for run in self._runs[-5:]:
            lines.append(run.summary_line())

        return "\n".join(lines)
