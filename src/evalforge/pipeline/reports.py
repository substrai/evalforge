"""Report generation for compliance and stakeholder communication."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evalforge.core.result import EvalResult


@dataclass
class ComplianceReport:
    """A compliance/quality report for stakeholders."""

    title: str
    project_name: str
    period: str
    generated_at: float = field(default_factory=time.time)
    results: List[EvalResult] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    @property
    def total_evaluations(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.all_passing) / len(self.results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "project_name": self.project_name,
            "period": self.period,
            "generated_at": self.generated_at,
            "total_evaluations": self.total_evaluations,
            "pass_rate": self.pass_rate,
            "summary_stats": self.summary_stats,
            "recommendations": self.recommendations,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# {self.title}",
            "",
            f"**Project:** {self.project_name}",
            f"**Period:** {self.period}",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M', time.localtime(self.generated_at))}",
            "",
            "## Summary",
            "",
            f"- Total evaluations: {self.total_evaluations}",
            f"- Pass rate: {self.pass_rate:.0%}",
        ]

        if self.summary_stats:
            lines.append("")
            lines.append("## Metrics Overview")
            lines.append("")
            lines.append("| Metric | Avg Score | Pass Rate |")
            lines.append("|--------|-----------|-----------|")
            for metric, stats in self.summary_stats.items():
                lines.append(f"| {metric} | {stats.get('avg', 0):.4f} | {stats.get('pass_rate', 0):.0%} |")

        if self.recommendations:
            lines.append("")
            lines.append("## Recommendations")
            lines.append("")
            for rec in self.recommendations:
                lines.append(f"- {rec}")

        return "\n".join(lines)


class ReportGenerator:
    """Generates evaluation reports for different audiences.

    Usage:
        gen = ReportGenerator()
        report = gen.generate(results, period="7d")
        print(report.to_markdown())
    """

    def generate(
        self,
        results: List[EvalResult],
        project_name: str = "evaluation",
        period: str = "7d",
        title: Optional[str] = None,
    ) -> ComplianceReport:
        """Generate a compliance report from evaluation results.

        Args:
            results: List of evaluation results
            project_name: Project name
            period: Report period
            title: Optional custom title

        Returns:
            ComplianceReport
        """
        title = title or f"Evaluation Quality Report — {project_name}"

        # Calculate summary stats per metric
        summary_stats = self._calculate_stats(results)

        # Generate recommendations
        recommendations = self._generate_recommendations(results, summary_stats)

        return ComplianceReport(
            title=title,
            project_name=project_name,
            period=period,
            results=results,
            summary_stats=summary_stats,
            recommendations=recommendations,
        )

    def generate_executive_summary(self, results: List[EvalResult]) -> str:
        """Generate a brief executive summary."""
        if not results:
            return "No evaluation data available."

        total = len(results)
        passing = sum(1 for r in results if r.all_passing)
        avg_score = sum(r.overall_score for r in results) / total

        lines = [
            f"Executive Summary: {total} evaluations conducted",
            f"  Pass rate: {passing}/{total} ({passing/total:.0%})",
            f"  Average quality score: {avg_score:.2f}",
        ]

        # Find worst metric
        all_scores: Dict[str, List[float]] = {}
        for r in results:
            for s in r.scores:
                if s.name not in all_scores:
                    all_scores[s.name] = []
                all_scores[s.name].append(s.score)

        if all_scores:
            worst_metric = min(all_scores.items(), key=lambda x: sum(x[1])/len(x[1]))
            avg_worst = sum(worst_metric[1]) / len(worst_metric[1])
            lines.append(f"  Lowest performing metric: {worst_metric[0]} ({avg_worst:.4f})")

        return "\n".join(lines)

    def _calculate_stats(self, results: List[EvalResult]) -> Dict[str, Dict[str, float]]:
        """Calculate per-metric statistics."""
        metric_scores: Dict[str, List[tuple]] = {}

        for result in results:
            for score in result.scores:
                if score.name not in metric_scores:
                    metric_scores[score.name] = []
                metric_scores[score.name].append((score.score, score.passed))

        stats = {}
        for metric, scores in metric_scores.items():
            values = [s[0] for s in scores]
            passed = [s[1] for s in scores]
            stats[metric] = {
                "avg": round(sum(values) / len(values), 4) if values else 0,
                "min": round(min(values), 4) if values else 0,
                "max": round(max(values), 4) if values else 0,
                "pass_rate": sum(passed) / len(passed) if passed else 0,
                "samples": len(values),
            }

        return stats

    def _generate_recommendations(
        self, results: List[EvalResult], stats: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        for metric, metric_stats in stats.items():
            if metric_stats["pass_rate"] < 0.8:
                recommendations.append(
                    f"Improve {metric}: pass rate is {metric_stats['pass_rate']:.0%} "
                    f"(avg score: {metric_stats['avg']:.4f})"
                )

        # Check for declining trend
        if len(results) >= 3:
            recent_scores = [r.overall_score for r in results[-3:]]
            if all(recent_scores[i] < recent_scores[i-1] for i in range(1, len(recent_scores))):
                recommendations.append(
                    "Quality is declining over recent evaluations — investigate root cause"
                )

        if not recommendations:
            recommendations.append("All metrics within acceptable thresholds — maintain current approach")

        return recommendations
