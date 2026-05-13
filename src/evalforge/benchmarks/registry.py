"""Benchmark registry - compare against published and custom benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Benchmark:
    """A benchmark definition."""

    name: str
    description: str
    source: str = "published"  # published | custom | community
    metrics: Dict[str, float] = field(default_factory=dict)  # metric -> reference score
    category: str = "general"  # general, rag, summarization, safety, etc.
    url: str = ""
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "metrics": self.metrics,
            "category": self.category,
        }


@dataclass
class BenchmarkResult:
    """Result of comparing against a benchmark."""

    benchmark_name: str
    system_scores: Dict[str, float]
    benchmark_scores: Dict[str, float]
    comparisons: Dict[str, Dict[str, float]] = field(default_factory=dict)
    overall_percentile: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def metrics_above_benchmark(self) -> List[str]:
        return [m for m, c in self.comparisons.items() if c.get("difference", 0) >= 0]

    @property
    def metrics_below_benchmark(self) -> List[str]:
        return [m for m, c in self.comparisons.items() if c.get("difference", 0) < 0]

    @property
    def pass_rate(self) -> float:
        if not self.comparisons:
            return 0.0
        return len(self.metrics_above_benchmark) / len(self.comparisons)

    def summary(self) -> str:
        lines = [
            f"Benchmark Comparison: {self.benchmark_name}",
            f"Overall percentile: {self.overall_percentile:.0f}th",
            f"Metrics above benchmark: {len(self.metrics_above_benchmark)}/{len(self.comparisons)}",
            "",
        ]
        for metric, comp in self.comparisons.items():
            diff = comp.get("difference", 0)
            icon = "✓" if diff >= 0 else "✗"
            lines.append(
                f"  {icon} {metric}: {comp.get('system', 0):.4f} vs {comp.get('benchmark', 0):.4f} "
                f"({'+'if diff >= 0 else ''}{diff:.4f})"
            )
        return "\n".join(lines)


# Built-in published benchmarks
PUBLISHED_BENCHMARKS = [
    Benchmark(
        name="RAG-Quality-Baseline",
        description="Baseline quality scores for RAG systems (industry average)",
        source="published",
        category="rag",
        metrics={"faithfulness": 0.82, "answer_relevancy": 0.78, "context_precision": 0.70, "context_recall": 0.68},
    ),
    Benchmark(
        name="Safety-Baseline",
        description="Baseline safety scores for production LLM systems",
        source="published",
        category="safety",
        metrics={"toxicity": 0.03, "injection_resistance": 0.92, "bias_detection": 0.05},
    ),
    Benchmark(
        name="Summarization-Baseline",
        description="Baseline scores for text summarization",
        source="published",
        category="summarization",
        metrics={"rouge_l": 0.65, "bleu": 0.55, "coherence": 0.78, "conciseness": 0.80},
    ),
    Benchmark(
        name="Classification-Baseline",
        description="Baseline classification performance",
        source="published",
        category="classification",
        metrics={"accuracy": 0.88, "precision": 0.85, "recall": 0.83, "f1_score": 0.84},
    ),
]


class BenchmarkRegistry:
    """Registry of benchmarks for comparison.

    Usage:
        registry = BenchmarkRegistry()
        result = registry.compare("RAG-Quality-Baseline", my_scores)
        print(result.summary())
    """

    def __init__(self):
        self._benchmarks: Dict[str, Benchmark] = {}
        self._results: List[BenchmarkResult] = []
        # Load published benchmarks
        for bm in PUBLISHED_BENCHMARKS:
            self._benchmarks[bm.name] = bm

    def register(self, benchmark: Benchmark) -> None:
        """Register a custom benchmark."""
        self._benchmarks[benchmark.name] = benchmark

    def get(self, name: str) -> Optional[Benchmark]:
        """Get a benchmark by name."""
        return self._benchmarks.get(name)

    def list_benchmarks(self, category: Optional[str] = None) -> List[Benchmark]:
        """List available benchmarks."""
        benchmarks = list(self._benchmarks.values())
        if category:
            benchmarks = [b for b in benchmarks if b.category == category]
        return benchmarks

    def compare(self, benchmark_name: str, system_scores: Dict[str, float]) -> BenchmarkResult:
        """Compare system scores against a benchmark.

        Args:
            benchmark_name: Name of the benchmark
            system_scores: Dict of metric -> system score

        Returns:
            BenchmarkResult with detailed comparison
        """
        benchmark = self._benchmarks.get(benchmark_name)
        if not benchmark:
            raise KeyError(f"Benchmark '{benchmark_name}' not found. Available: {list(self._benchmarks.keys())}")

        comparisons = {}
        for metric, bench_score in benchmark.metrics.items():
            system_score = system_scores.get(metric)
            if system_score is not None:
                # For safety metrics (lower = better), invert comparison
                is_safety = metric in ("toxicity", "bias_detection")
                if is_safety:
                    diff = bench_score - system_score  # lower system = better
                else:
                    diff = system_score - bench_score  # higher system = better

                comparisons[metric] = {
                    "system": system_score,
                    "benchmark": bench_score,
                    "difference": round(diff, 4),
                    "above_benchmark": diff >= 0,
                }

        # Calculate percentile (simplified: based on % of metrics above benchmark)
        if comparisons:
            above = sum(1 for c in comparisons.values() if c["above_benchmark"])
            percentile = (above / len(comparisons)) * 100
        else:
            percentile = 0.0

        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            system_scores=system_scores,
            benchmark_scores=benchmark.metrics,
            comparisons=comparisons,
            overall_percentile=percentile,
        )
        self._results.append(result)
        return result

    def compare_all(self, system_scores: Dict[str, float], category: Optional[str] = None) -> List[BenchmarkResult]:
        """Compare against all relevant benchmarks."""
        benchmarks = self.list_benchmarks(category)
        results = []
        for bm in benchmarks:
            # Only compare if we have overlapping metrics
            overlap = set(bm.metrics.keys()) & set(system_scores.keys())
            if overlap:
                results.append(self.compare(bm.name, system_scores))
        return results

    def get_history(self, benchmark_name: Optional[str] = None) -> List[BenchmarkResult]:
        """Get comparison history."""
        if benchmark_name:
            return [r for r in self._results if r.benchmark_name == benchmark_name]
        return self._results
