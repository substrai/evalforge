"""Tests for HTML report generation with inline SVG charts."""

import pytest

from evalforge.reporting.html_report import (
    EvalRun,
    HTMLReportGenerator,
    MetricResult,
    SVGChartGenerator,
)


class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_passed_when_score_meets_threshold(self):
        """Test that metric passes when score >= threshold."""
        metric = MetricResult(name="accuracy", score=0.85, threshold=0.8)
        assert metric.passed is True

    def test_failed_when_score_below_threshold(self):
        """Test that metric fails when score < threshold."""
        metric = MetricResult(name="accuracy", score=0.6, threshold=0.8)
        assert metric.passed is False

    def test_passed_at_exact_threshold(self):
        """Test that metric passes at exact threshold."""
        metric = MetricResult(name="accuracy", score=0.8, threshold=0.8)
        assert metric.passed is True


class TestEvalRun:
    """Tests for EvalRun dataclass."""

    def test_pass_rate_calculation(self):
        """Test pass rate calculation across metrics."""
        run = EvalRun(
            run_id="run-1",
            timestamp="2024-01-01T00:00:00Z",
            metrics=[
                MetricResult(name="a", score=0.9, threshold=0.8),
                MetricResult(name="b", score=0.7, threshold=0.8),
                MetricResult(name="c", score=0.85, threshold=0.8),
            ],
        )
        assert run.pass_count == 2
        assert run.fail_count == 1
        assert run.pass_rate == pytest.approx(2 / 3)

    def test_empty_metrics_gives_zero_rate(self):
        """Test that empty metrics gives 0.0 pass rate."""
        run = EvalRun(run_id="empty", timestamp="2024-01-01", metrics=[])
        assert run.pass_rate == 0.0


class TestSVGChartGenerator:
    """Tests for SVG chart generation."""

    def test_bar_chart_generates_valid_svg(self):
        """Test that bar chart produces valid SVG markup."""
        svg = SVGChartGenerator.bar_chart(
            labels=["A", "B", "C"],
            values=[0.8, 0.6, 0.9],
        )
        assert '<svg xmlns="http://www.w3.org/2000/svg"' in svg
        assert "</svg>" in svg
        assert "rect" in svg

    def test_bar_chart_with_threshold_line(self):
        """Test that threshold line is rendered."""
        svg = SVGChartGenerator.bar_chart(
            labels=["A", "B"],
            values=[0.9, 0.4],
            threshold=0.7,
        )
        assert "stroke-dasharray" in svg
        assert "line" in svg

    def test_bar_chart_empty_values(self):
        """Test bar chart with empty values."""
        svg = SVGChartGenerator.bar_chart(labels=[], values=[])
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_trend_line_generates_valid_svg(self):
        """Test that trend line produces valid SVG markup."""
        svg = SVGChartGenerator.trend_line(values=[0.7, 0.75, 0.8, 0.85])
        assert '<svg xmlns="http://www.w3.org/2000/svg"' in svg
        assert "polyline" in svg
        assert "circle" in svg

    def test_trend_line_empty_values(self):
        """Test trend line with empty values."""
        svg = SVGChartGenerator.trend_line(values=[])
        assert "<svg" in svg

    def test_pass_fail_badge_pass(self):
        """Test pass badge generation."""
        svg = SVGChartGenerator.pass_fail_badge(passed=True)
        assert "PASS" in svg
        assert "#4CAF50" in svg

    def test_pass_fail_badge_fail(self):
        """Test fail badge generation."""
        svg = SVGChartGenerator.pass_fail_badge(passed=False)
        assert "FAIL" in svg
        assert "#F44336" in svg


class TestHTMLReportGenerator:
    """Tests for the full HTML report generator."""

    def test_generate_complete_report(self):
        """Test generating a complete HTML report."""
        runs = [
            EvalRun(
                run_id="run-1",
                timestamp="2024-01-01T00:00:00Z",
                metrics=[
                    MetricResult(name="accuracy", score=0.92, threshold=0.8),
                    MetricResult(name="latency", score=0.45, threshold=0.5),
                ],
            ),
        ]
        generator = HTMLReportGenerator(title="Test Report", description="A test")
        report = generator.generate(runs)

        assert "<!DOCTYPE html>" in report
        assert "Test Report" in report
        assert "A test" in report
        assert "accuracy" in report
        assert "</html>" in report

    def test_generate_report_with_trend(self):
        """Test report with multiple runs shows trend."""
        runs = [
            EvalRun(
                run_id=f"run-{i}",
                timestamp=f"2024-01-0{i}T00:00:00Z",
                metrics=[MetricResult(name="acc", score=0.7 + i * 0.05, threshold=0.8)],
            )
            for i in range(1, 4)
        ]
        generator = HTMLReportGenerator(title="Trend Report")
        report = generator.generate(runs)

        assert "Pass Rate Trend" in report
        assert "polyline" in report

    def test_generate_empty_report(self):
        """Test report generation with no runs."""
        generator = HTMLReportGenerator(title="Empty Report")
        report = generator.generate([])

        assert "<!DOCTYPE html>" in report
        assert "No evaluation runs" in report

    def test_report_contains_summary_stats(self):
        """Test that report contains summary statistics."""
        runs = [
            EvalRun(
                run_id="run-1",
                timestamp="2024-01-01",
                metrics=[
                    MetricResult(name="m1", score=0.9, threshold=0.8),
                    MetricResult(name="m2", score=0.6, threshold=0.8),
                ],
            ),
        ]
        generator = HTMLReportGenerator()
        report = generator.generate(runs)

        assert "Total Runs" in report
        assert "Passed" in report
        assert "Failed" in report

    def test_report_escapes_html_in_title(self):
        """Test that HTML special chars are escaped."""
        generator = HTMLReportGenerator(title="<script>alert('xss')</script>")
        report = generator.generate([])
        assert "<script>" not in report
        assert "&lt;script&gt;" in report
