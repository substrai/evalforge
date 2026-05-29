"""HTML report generation with inline SVG charts.

Generates standalone HTML reports with inline SVG bar charts, trend lines,
and pass/fail badges. Template-based generation with no external dependencies.
"""

from __future__ import annotations

import html
import statistics
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MetricResult:
    """A single metric evaluation result."""
    name: str
    score: float
    threshold: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether the metric meets its threshold."""
        return self.score >= self.threshold


@dataclass
class EvalRun:
    """A single evaluation run with multiple metrics."""
    run_id: str
    timestamp: str
    metrics: list[MetricResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pass_count(self) -> int:
        return sum(1 for m in self.metrics if m.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for m in self.metrics if not m.passed)

    @property
    def pass_rate(self) -> float:
        if not self.metrics:
            return 0.0
        return self.pass_count / len(self.metrics)


class SVGChartGenerator:
    """Generates inline SVG charts for HTML reports."""

    @staticmethod
    def bar_chart(
        labels: list[str],
        values: list[float],
        width: int = 500,
        height: int = 300,
        bar_color: str = "#4CAF50",
        threshold: Optional[float] = None,
    ) -> str:
        """Generate an SVG bar chart.

        Args:
            labels: X-axis labels for each bar.
            values: Numeric values for each bar.
            width: Chart width in pixels.
            height: Chart height in pixels.
            bar_color: Default bar color.
            threshold: Optional threshold line value.

        Returns:
            SVG markup string.
        """
        if not values:
            return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'

        max_val = max(values) if max(values) > 0 else 1.0
        margin = 60
        chart_width = width - 2 * margin
        chart_height = height - 2 * margin
        bar_width = chart_width / len(values) * 0.7
        bar_gap = chart_width / len(values) * 0.3

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="#fafafa" rx="4"/>',
        ]

        # Draw bars
        for i, (label, value) in enumerate(zip(labels, values)):
            bar_height = (value / max_val) * chart_height
            x = margin + i * (bar_width + bar_gap)
            y = margin + chart_height - bar_height

            color = bar_color
            if threshold is not None and value < threshold:
                color = "#F44336"

            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" fill="{color}" rx="2"/>'
            )
            # Label
            label_x = x + bar_width / 2
            label_y = height - margin / 3
            escaped_label = html.escape(label[:12])
            parts.append(
                f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="middle" '
                f'font-size="11" fill="#333">{escaped_label}</text>'
            )
            # Value on top
            parts.append(
                f'<text x="{label_x:.1f}" y="{y - 5:.1f}" text-anchor="middle" '
                f'font-size="10" fill="#666">{value:.2f}</text>'
            )

        # Threshold line
        if threshold is not None:
            threshold_y = margin + chart_height - (threshold / max_val) * chart_height
            parts.append(
                f'<line x1="{margin}" y1="{threshold_y:.1f}" x2="{width - margin}" '
                f'y2="{threshold_y:.1f}" stroke="#FF9800" stroke-width="2" '
                f'stroke-dasharray="5,3"/>'
            )

        parts.append("</svg>")
        return "\n".join(parts)

    @staticmethod
    def trend_line(
        values: list[float],
        width: int = 500,
        height: int = 200,
        line_color: str = "#2196F3",
    ) -> str:
        """Generate an SVG trend line chart.

        Args:
            values: Sequential numeric values to plot.
            width: Chart width in pixels.
            height: Chart height in pixels.
            line_color: Line color.

        Returns:
            SVG markup string.
        """
        if not values:
            return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'

        margin = 40
        chart_width = width - 2 * margin
        chart_height = height - 2 * margin
        max_val = max(values) if max(values) > 0 else 1.0
        min_val = min(values)
        val_range = max_val - min_val if max_val != min_val else 1.0

        points = []
        for i, value in enumerate(values):
            x = margin + (i / max(len(values) - 1, 1)) * chart_width
            y = margin + chart_height - ((value - min_val) / val_range) * chart_height
            points.append(f"{x:.1f},{y:.1f}")

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="#fafafa" rx="4"/>',
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{line_color}" stroke-width="2" stroke-linejoin="round"/>',
        ]

        # Data points
        for point in points:
            x, y = point.split(",")
            parts.append(
                f'<circle cx="{x}" cy="{y}" r="3" fill="{line_color}"/>'
            )

        parts.append("</svg>")
        return "\n".join(parts)

    @staticmethod
    def pass_fail_badge(passed: bool, label: str = "") -> str:
        """Generate a pass/fail badge SVG.

        Args:
            passed: Whether the result passed.
            label: Optional label text.

        Returns:
            SVG badge markup string.
        """
        color = "#4CAF50" if passed else "#F44336"
        text = label or ("PASS" if passed else "FAIL")
        text_width = len(text) * 8 + 16
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{text_width}" height="24">'
            f'<rect width="{text_width}" height="24" fill="{color}" rx="4"/>'
            f'<text x="{text_width // 2}" y="16" text-anchor="middle" '
            f'font-size="12" font-weight="bold" fill="white">{html.escape(text)}</text>'
            f'</svg>'
        )


class HTMLReportGenerator:
    """Generates standalone HTML reports from evaluation results.

    Creates self-contained HTML files with inline SVG charts,
    no external dependencies required.

    Args:
        title: Report title.
        description: Optional report description.
    """

    def __init__(self, title: str = "Evaluation Report", description: str = ""):
        self.title = title
        self.description = description
        self._chart_gen = SVGChartGenerator()

    def generate(self, runs: list[EvalRun]) -> str:
        """Generate a complete HTML report.

        Args:
            runs: List of evaluation runs to include in the report.

        Returns:
            Complete HTML document as a string.
        """
        sections = [
            self._generate_header(),
            self._generate_summary(runs),
            self._generate_metrics_chart(runs),
            self._generate_trend_section(runs),
            self._generate_details_table(runs),
            self._generate_footer(),
        ]
        return "\n".join(sections)

    def _generate_header(self) -> str:
        """Generate HTML header with embedded styles."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(self.title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
.card {{ background: white; border-radius: 8px; padding: 24px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
h1 {{ color: #1a1a1a; margin-top: 0; }}
h2 {{ color: #444; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
.stat {{ text-align: center; padding: 16px; background: #f8f9fa; border-radius: 8px; }}
.stat-value {{ font-size: 2em; font-weight: bold; color: #2196F3; }}
.stat-label {{ font-size: 0.9em; color: #666; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
th {{ background: #f8f9fa; font-weight: 600; }}
.chart-container {{ text-align: center; margin: 20px 0; }}
.badge {{ display: inline-block; margin: 2px; }}
</style>
</head>
<body>
<div class="container">
<div class="card">
<h1>{html.escape(self.title)}</h1>
{"<p>" + html.escape(self.description) + "</p>" if self.description else ""}
</div>"""

    def _generate_summary(self, runs: list[EvalRun]) -> str:
        """Generate summary statistics section."""
        if not runs:
            return '<div class="card"><p>No evaluation runs to display.</p></div>'

        total_metrics = sum(len(r.metrics) for r in runs)
        total_pass = sum(r.pass_count for r in runs)
        total_fail = sum(r.fail_count for r in runs)
        avg_pass_rate = statistics.mean(r.pass_rate for r in runs) if runs else 0

        return f"""
<div class="card">
<h2>Summary</h2>
<div class="summary">
<div class="stat"><div class="stat-value">{len(runs)}</div><div class="stat-label">Total Runs</div></div>
<div class="stat"><div class="stat-value">{total_metrics}</div><div class="stat-label">Total Metrics</div></div>
<div class="stat"><div class="stat-value">{total_pass}</div><div class="stat-label">Passed</div></div>
<div class="stat"><div class="stat-value">{total_fail}</div><div class="stat-label">Failed</div></div>
<div class="stat"><div class="stat-value">{avg_pass_rate:.0%}</div><div class="stat-label">Avg Pass Rate</div></div>
</div>
</div>"""

    def _generate_metrics_chart(self, runs: list[EvalRun]) -> str:
        """Generate bar chart of latest run metrics."""
        if not runs:
            return ""

        latest_run = runs[-1]
        if not latest_run.metrics:
            return ""

        labels = [m.name for m in latest_run.metrics]
        values = [m.score for m in latest_run.metrics]
        threshold = latest_run.metrics[0].threshold if latest_run.metrics else None

        chart_svg = self._chart_gen.bar_chart(
            labels=labels,
            values=values,
            threshold=threshold,
        )

        return f"""
<div class="card">
<h2>Latest Run Metrics</h2>
<div class="chart-container">{chart_svg}</div>
</div>"""

    def _generate_trend_section(self, runs: list[EvalRun]) -> str:
        """Generate trend line chart across runs."""
        if len(runs) < 2:
            return ""

        pass_rates = [r.pass_rate for r in runs]
        trend_svg = self._chart_gen.trend_line(pass_rates)

        return f"""
<div class="card">
<h2>Pass Rate Trend</h2>
<div class="chart-container">{trend_svg}</div>
</div>"""

    def _generate_details_table(self, runs: list[EvalRun]) -> str:
        """Generate detailed results table."""
        if not runs:
            return ""

        rows = []
        for run in runs:
            for metric in run.metrics:
                badge = self._chart_gen.pass_fail_badge(metric.passed)
                rows.append(
                    f"<tr><td>{html.escape(run.run_id)}</td>"
                    f"<td>{html.escape(run.timestamp)}</td>"
                    f"<td>{html.escape(metric.name)}</td>"
                    f"<td>{metric.score:.4f}</td>"
                    f"<td>{metric.threshold:.4f}</td>"
                    f'<td class="badge">{badge}</td></tr>'
                )

        return f"""
<div class="card">
<h2>Detailed Results</h2>
<table>
<thead><tr><th>Run ID</th><th>Timestamp</th><th>Metric</th><th>Score</th><th>Threshold</th><th>Status</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
</div>"""

    def _generate_footer(self) -> str:
        """Generate HTML footer."""
        return """
</div>
</body>
</html>"""
