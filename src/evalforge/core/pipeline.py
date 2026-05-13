"""Evaluation pipeline - orchestrates metric execution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalforge.core.config import EvalConfig
from evalforge.core.result import EvalResult, MetricScore
from evalforge.metrics.base import BaseMetric, MetricInput
from evalforge.metrics.registry import MetricRegistry


@dataclass
class TestSample:
    """A single test sample for evaluation."""

    query: str
    response: str
    context: str = ""
    reference: str = ""
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvalPipeline:
    """Orchestrates the evaluation pipeline.

    Usage:
        pipeline = EvalPipeline.from_config("evalforge.yaml")
        results = pipeline.run(samples=my_test_data)
        print(results.summary())
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self._registry = MetricRegistry()
        self._metrics: List[BaseMetric] = []
        self._load_metrics()

    @classmethod
    def from_config(cls, path: str | Path) -> "EvalPipeline":
        """Create pipeline from a config file."""
        config = EvalConfig.from_file(path)
        return cls(config)

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "EvalPipeline":
        """Create pipeline from YAML content."""
        config = EvalConfig.from_yaml(yaml_content)
        return cls(config)

    @classmethod
    def for_use_case(cls, use_case: str, project_name: str = "evaluation") -> "EvalPipeline":
        """Quick-start pipeline for a use case type."""
        config = EvalConfig.for_use_case(use_case, project_name)
        return cls(config)

    def _load_metrics(self) -> None:
        """Load metrics based on config."""
        for metric_name in self.config.metrics:
            try:
                metric = self._registry.get(metric_name)
                self._metrics.append(metric)
            except KeyError:
                pass

    def run(
        self,
        samples: Optional[List[TestSample]] = None,
        metrics: Optional[List[str]] = None,
    ) -> EvalResult:
        """Run the evaluation pipeline.

        Args:
            samples: Test samples to evaluate (uses defaults if None)
            metrics: Specific metrics to run (all if None)

        Returns:
            EvalResult with all metric scores
        """
        start_time = time.time()

        if samples is None:
            samples = self._generate_default_samples()

        metrics_to_run = self._metrics
        if metrics:
            metrics_to_run = [m for m in self._metrics if m.name in metrics]

        scores: List[MetricScore] = []
        for metric in metrics_to_run:
            metric_start = time.time()
            threshold = self.config.get_threshold(metric.name)

            outputs = []
            for sample in samples:
                metric_input = MetricInput(
                    query=sample.query,
                    response=sample.response,
                    context=sample.context,
                    reference=sample.reference,
                    metadata=sample.metadata,
                )
                output = metric.evaluate(metric_input, threshold)
                outputs.append(output)

            if outputs:
                avg_score = sum(o.score for o in outputs) / len(outputs)
                if metric.category == "safety" and metric.name in ("toxicity", "bias_detection"):
                    passed = avg_score <= threshold
                else:
                    passed = avg_score >= threshold
            else:
                avg_score = 0.0
                passed = False

            metric_latency = (time.time() - metric_start) * 1000

            scores.append(MetricScore(
                name=metric.name,
                score=round(avg_score, 4),
                threshold=threshold,
                passed=passed,
                samples_evaluated=len(outputs),
                latency_ms=round(metric_latency, 2),
                details={"category": metric.category},
            ))

        total_latency = (time.time() - start_time) * 1000

        return EvalResult(
            project_name=self.config.project_name,
            use_case_type=self.config.use_case_type.value,
            scores=scores,
            total_samples=len(samples),
            total_latency_ms=round(total_latency, 2),
            model=f"{self.config.model.provider}/{self.config.model.model_id}",
        )

    def run_single(self, sample: TestSample) -> Dict[str, float]:
        """Run all metrics on a single sample."""
        results = {}
        for metric in self._metrics:
            threshold = self.config.get_threshold(metric.name)
            metric_input = MetricInput(
                query=sample.query,
                response=sample.response,
                context=sample.context,
                reference=sample.reference,
            )
            output = metric.evaluate(metric_input, threshold)
            results[metric.name] = output.score
        return results

    def list_metrics(self) -> List[str]:
        """List metrics that will be run."""
        return [m.name for m in self._metrics]

    def _generate_default_samples(self) -> List[TestSample]:
        """Generate minimal default samples for testing."""
        return [
            TestSample(
                query="What is the return policy?",
                response="Our return policy allows returns within 30 days of purchase with a valid receipt.",
                context="Return Policy: Customers may return items within 30 days of purchase. A valid receipt is required. Items must be in original condition.",
                reference="Returns are accepted within 30 days with receipt.",
                category="simple",
            ),
            TestSample(
                query="How do I reset my password?",
                response="To reset your password, go to Settings > Security > Reset Password and follow the prompts.",
                context="Password Reset: Navigate to Settings, then Security, then click Reset Password. You will receive an email with a reset link.",
                reference="Go to Settings > Security > Reset Password.",
                category="simple",
            ),
            TestSample(
                query="Compare premium vs basic plans",
                response="The premium plan includes unlimited storage, priority support, and advanced analytics. The basic plan offers 10GB storage and email support.",
                context="Basic Plan: 10GB storage, email support, $9/month. Premium Plan: Unlimited storage, priority support, advanced analytics, $29/month.",
                reference="Premium has unlimited storage and priority support; basic has 10GB and email support.",
                category="complex",
            ),
        ]
