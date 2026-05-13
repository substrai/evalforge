"""Tests for benchmark registry."""

import pytest
from evalforge.benchmarks.registry import BenchmarkRegistry, Benchmark, BenchmarkResult


class TestBenchmarkRegistry:
    def setup_method(self):
        self.registry = BenchmarkRegistry()

    def test_list_published_benchmarks(self):
        benchmarks = self.registry.list_benchmarks()
        assert len(benchmarks) >= 4
        names = [b.name for b in benchmarks]
        assert "RAG-Quality-Baseline" in names
        assert "Safety-Baseline" in names

    def test_list_by_category(self):
        rag = self.registry.list_benchmarks(category="rag")
        assert len(rag) >= 1
        assert all(b.category == "rag" for b in rag)

    def test_compare_above_benchmark(self):
        scores = {"faithfulness": 0.92, "answer_relevancy": 0.85, "context_precision": 0.80, "context_recall": 0.75}
        result = self.registry.compare("RAG-Quality-Baseline", scores)
        assert result.overall_percentile == 100.0  # all above
        assert len(result.metrics_above_benchmark) == 4

    def test_compare_below_benchmark(self):
        scores = {"faithfulness": 0.50, "answer_relevancy": 0.40}
        result = self.registry.compare("RAG-Quality-Baseline", scores)
        assert result.overall_percentile < 100.0
        assert "faithfulness" in result.metrics_below_benchmark

    def test_compare_safety(self):
        # For safety: lower system score = better
        scores = {"toxicity": 0.01, "injection_resistance": 0.95, "bias_detection": 0.02}
        result = self.registry.compare("Safety-Baseline", scores)
        assert result.overall_percentile == 100.0

    def test_register_custom_benchmark(self):
        custom = Benchmark(
            name="My-Internal-Benchmark",
            description="Internal quality standard",
            source="custom",
            category="rag",
            metrics={"faithfulness": 0.90, "answer_relevancy": 0.85},
        )
        self.registry.register(custom)
        assert self.registry.get("My-Internal-Benchmark") is not None

    def test_compare_custom(self):
        custom = Benchmark(
            name="Custom", description="Test", source="custom",
            metrics={"faithfulness": 0.90},
        )
        self.registry.register(custom)
        result = self.registry.compare("Custom", {"faithfulness": 0.95})
        assert result.metrics_above_benchmark == ["faithfulness"]

    def test_compare_all(self):
        scores = {"faithfulness": 0.90, "answer_relevancy": 0.85, "toxicity": 0.01}
        results = self.registry.compare_all(scores)
        assert len(results) >= 1

    def test_result_summary(self):
        scores = {"faithfulness": 0.90, "answer_relevancy": 0.85}
        result = self.registry.compare("RAG-Quality-Baseline", scores)
        summary = result.summary()
        assert "RAG-Quality-Baseline" in summary
        assert "faithfulness" in summary

    def test_unknown_benchmark(self):
        with pytest.raises(KeyError):
            self.registry.compare("Nonexistent", {"faithfulness": 0.9})

    def test_history(self):
        self.registry.compare("RAG-Quality-Baseline", {"faithfulness": 0.90})
        self.registry.compare("RAG-Quality-Baseline", {"faithfulness": 0.85})
        history = self.registry.get_history("RAG-Quality-Baseline")
        assert len(history) == 2

    def test_pass_rate(self):
        scores = {"faithfulness": 0.90, "answer_relevancy": 0.70}
        result = self.registry.compare("RAG-Quality-Baseline", scores)
        # faithfulness above (0.90 > 0.82), relevancy below (0.70 < 0.78)
        assert result.pass_rate == 0.5
