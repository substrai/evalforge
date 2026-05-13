"""Tests for EvalForge configuration."""

import pytest
from evalforge.core.config import EvalConfig, UseCaseType, DEFAULT_METRICS, DEFAULT_THRESHOLDS


SAMPLE_CONFIG = """
project:
  name: "test-eval"
  version: "1.0.0"

use_case:
  type: rag
  description: "Test RAG evaluation"
  model:
    provider: bedrock
    model_id: anthropic.claude-3-haiku-20240307-v1:0

evaluation:
  metrics: auto
  thresholds:
    faithfulness: 0.90

test_data:
  source: synthetic
  count: 50

schedule:
  frequency: daily
"""


def test_parse_config():
    config = EvalConfig.from_yaml(SAMPLE_CONFIG)
    assert config.project_name == "test-eval"
    assert config.use_case_type == UseCaseType.RAG
    assert "faithfulness" in config.metrics


def test_auto_metrics_selection():
    config = EvalConfig.from_yaml(SAMPLE_CONFIG)
    assert config.metrics == DEFAULT_METRICS["rag"]


def test_threshold_override():
    config = EvalConfig.from_yaml(SAMPLE_CONFIG)
    assert config.get_threshold("faithfulness") == 0.90  # overridden
    assert config.get_threshold("answer_relevancy") == 0.80  # default


def test_for_use_case_shortcut():
    config = EvalConfig.for_use_case("summarization", "my-project")
    assert config.use_case_type == UseCaseType.SUMMARIZATION
    assert "rouge_l" in config.metrics
    assert "bleu" in config.metrics


def test_all_use_case_types():
    for uc_type in UseCaseType:
        config = EvalConfig.for_use_case(uc_type.value)
        assert len(config.metrics) > 0
        assert len(config.thresholds) > 0


def test_invalid_use_case():
    with pytest.raises(ValueError):
        UseCaseType.from_string("invalid")


def test_empty_config():
    with pytest.raises(ValueError):
        EvalConfig.from_yaml("")


def test_explicit_metrics():
    yaml_content = """
project:
  name: test
use_case:
  type: rag
evaluation:
  metrics: [faithfulness, toxicity]
"""
    config = EvalConfig.from_yaml(yaml_content)
    assert config.metrics == ["faithfulness", "toxicity"]


def test_config_summary():
    config = EvalConfig.from_yaml(SAMPLE_CONFIG)
    summary = config.summary()
    assert "test-eval" in summary
    assert "rag" in summary
    assert "faithfulness" in summary
