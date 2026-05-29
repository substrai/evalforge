"""
EvalForge - Automated LLM Evaluation Pipeline Generator

Describe your GenAI use case in a config file, and EvalForge generates
the complete evaluation infrastructure: metrics selection, test data,
scheduled pipelines, drift detection, and human-in-the-loop review.

Usage:
    from evalforge import EvalPipeline, EvalConfig, EvalResult

    pipeline = EvalPipeline.from_config("evalforge.yaml")
    results = pipeline.run()
    print(results.summary())
"""

__version__ = "0.8.0"

from evalforge.core.config import EvalConfig, UseCaseType
from evalforge.core.pipeline import EvalPipeline
from evalforge.core.result import EvalResult, MetricScore
from evalforge.metrics.registry import MetricRegistry, get_metrics_for_use_case

__all__ = [
    "EvalConfig",
    "EvalPipeline",
    "EvalResult",
    "MetricScore",
    "UseCaseType",
    "MetricRegistry",
    "get_metrics_for_use_case",
]
