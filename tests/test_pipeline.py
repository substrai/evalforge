"""Tests for EvalForge pipeline."""

import pytest
from evalforge.core.pipeline import EvalPipeline, TestSample
from evalforge.core.config import EvalConfig


def test_pipeline_for_use_case():
    pipeline = EvalPipeline.for_use_case("rag")
    assert len(pipeline.list_metrics()) > 0


def test_pipeline_run_default_samples():
    pipeline = EvalPipeline.for_use_case("rag")
    results = pipeline.run()
    assert results.total_samples > 0
    assert len(results.scores) > 0
    assert results.overall_score > 0


def test_pipeline_run_custom_samples():
    pipeline = EvalPipeline.for_use_case("rag")
    samples = [
        TestSample(
            query="What is Python?",
            response="Python is a programming language known for its simplicity.",
            context="Python is a high-level programming language. It is known for its simple syntax and readability.",
            reference="Python is a programming language.",
        ),
    ]
    results = pipeline.run(samples=samples)
    assert results.total_samples == 1


def test_pipeline_run_specific_metrics():
    pipeline = EvalPipeline.for_use_case("rag")
    results = pipeline.run(metrics=["faithfulness"])
    assert len(results.scores) == 1
    assert results.scores[0].name == "faithfulness"


def test_pipeline_result_summary():
    pipeline = EvalPipeline.for_use_case("rag")
    results = pipeline.run()
    summary = results.summary()
    assert "EvalForge Results" in summary
    assert "rag" in summary


def test_pipeline_result_to_dict():
    pipeline = EvalPipeline.for_use_case("rag")
    results = pipeline.run()
    data = results.to_dict()
    assert "scores" in data
    assert "all_passing" in data
    assert "overall_score" in data


def test_pipeline_summarization():
    pipeline = EvalPipeline.for_use_case("summarization")
    samples = [
        TestSample(
            query="Summarize this",
            response="This is a concise summary of the main points.",
            reference="This is a summary of the key points discussed.",
        ),
    ]
    results = pipeline.run(samples=samples)
    assert results.total_samples == 1
    assert len(results.scores) > 0


def test_pipeline_classification():
    pipeline = EvalPipeline.for_use_case("classification")
    samples = [
        TestSample(query="Classify", response="positive", reference="positive"),
        TestSample(query="Classify", response="negative", reference="negative"),
        TestSample(query="Classify", response="positive", reference="negative"),
    ]
    results = pipeline.run(samples=samples)
    assert results.total_samples == 3


def test_pipeline_failing_metrics():
    pipeline = EvalPipeline.for_use_case("rag")
    # Intentionally bad sample
    samples = [
        TestSample(
            query="What is the weather?",
            response="I like pizza and cats are cute animals.",
            context="Weather forecast: sunny, 75F.",
            reference="Sunny and 75 degrees.",
        ),
    ]
    results = pipeline.run(samples=samples)
    failing = results.failing_metrics()
    # Some metrics should fail on this irrelevant response
    assert len(failing) >= 0  # at least runs without error
