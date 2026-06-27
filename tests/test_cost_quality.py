"""Tests for cost-per-quality metric (cost efficiency scoring)."""

from __future__ import annotations

import pytest

from evalforge.metrics.cost_quality import (
    CostQualityMetric,
    CostQualityResult,
    EfficiencyTier,
    ModelComparison,
    MODEL_PRICING,
)


class TestCostQualityMetricInit:
    """Test metric initialization."""

    def test_default_weights(self):
        metric = CostQualityMetric()
        assert metric.quality_weight == 0.7
        assert metric.cost_weight == 0.3

    def test_custom_weights(self):
        metric = CostQualityMetric(quality_weight=0.5, cost_weight=0.5)
        assert metric.quality_weight == 0.5
        assert metric.cost_weight == 0.5

    def test_invalid_weights_sum(self):
        with pytest.raises(ValueError, match="must equal 1.0"):
            CostQualityMetric(quality_weight=0.5, cost_weight=0.3)

    def test_invalid_max_cost(self):
        with pytest.raises(ValueError, match="must be positive"):
            CostQualityMetric(max_expected_cost=0)

    def test_empty_history(self):
        metric = CostQualityMetric()
        assert metric.history == []


class TestCostQualityEvaluation:
    """Test single evaluation."""

    def test_high_quality_low_cost(self):
        metric = CostQualityMetric()
        result = metric.evaluate(quality_score=0.95, cost_usd=0.001, model="claude-3-haiku")
        assert result.efficiency_score > 0.8
        assert result.tier in (EfficiencyTier.EXCELLENT, EfficiencyTier.GOOD)

    def test_low_quality_high_cost(self):
        metric = CostQualityMetric()
        result = metric.evaluate(quality_score=0.2, cost_usd=0.08, model="gpt-4-turbo")
        assert result.efficiency_score < 0.4
        assert result.tier in (EfficiencyTier.POOR, EfficiencyTier.WASTEFUL)

    def test_cost_per_quality_point(self):
        metric = CostQualityMetric()
        result = metric.evaluate(quality_score=0.80, cost_usd=0.004)
        expected = 0.004 / 0.80
        assert abs(result.cost_per_quality_point - expected) < 0.0001

    def test_quality_per_dollar(self):
        metric = CostQualityMetric()
        result = metric.evaluate(quality_score=0.90, cost_usd=0.002)
        expected = 0.90 / 0.002
        assert abs(result.quality_per_dollar - expected) < 1.0

    def test_zero_quality_infinite_cost_per_point(self):
        metric = CostQualityMetric()
        result = metric.evaluate(quality_score=0.0, cost_usd=0.01)
        assert result.cost_per_quality_point == float("inf")

    def test_zero_cost_infinite_quality_per_dollar(self):
        metric = CostQualityMetric()
        result = metric.evaluate(quality_score=0.9, cost_usd=0.0)
        assert result.quality_per_dollar == float("inf")

    def test_result_stored_in_history(self):
        metric = CostQualityMetric()
        metric.evaluate(quality_score=0.9, cost_usd=0.01)
        metric.evaluate(quality_score=0.8, cost_usd=0.005)
        assert len(metric.history) == 2

    def test_efficiency_score_bounded(self):
        metric = CostQualityMetric()
        result = metric.evaluate(quality_score=1.0, cost_usd=0.0)
        assert 0.0 <= result.efficiency_score <= 1.0

        result = metric.evaluate(quality_score=0.0, cost_usd=1.0)
        assert 0.0 <= result.efficiency_score <= 1.0


class TestEfficiencyTiers:
    """Test tier classification."""

    def test_excellent_tier(self):
        metric = CostQualityMetric()
        result = metric.evaluate(quality_score=0.99, cost_usd=0.0001)
        assert result.tier == EfficiencyTier.EXCELLENT

    def test_good_tier(self):
        metric = CostQualityMetric()
        result = metric.evaluate(quality_score=0.85, cost_usd=0.02)
        assert result.tier == EfficiencyTier.GOOD

    def test_acceptable_tier(self):
        metric = CostQualityMetric()
        result = metric.evaluate(quality_score=0.70, cost_usd=0.04)
        assert result.tier == EfficiencyTier.ACCEPTABLE

    def test_poor_tier(self):
        metric = CostQualityMetric()
        result = metric.evaluate(quality_score=0.40, cost_usd=0.06)
        assert result.tier == EfficiencyTier.POOR

    def test_wasteful_tier(self):
        metric = CostQualityMetric()
        result = metric.evaluate(quality_score=0.1, cost_usd=0.09)
        assert result.tier == EfficiencyTier.WASTEFUL


class TestModelComparison:
    """Test multi-model comparison."""

    def test_compare_three_models(self):
        metric = CostQualityMetric()
        evaluations = [
            ("claude-3-haiku", 0.85, 0.001),
            ("claude-3-sonnet", 0.92, 0.009),
            ("claude-3-opus", 0.97, 0.045),
        ]
        comparison = metric.compare_models(evaluations)
        assert comparison.best_quality == "claude-3-opus"
        assert comparison.cheapest == "claude-3-haiku"
        assert len(comparison.results) == 3

    def test_best_value_is_haiku(self):
        metric = CostQualityMetric()
        evaluations = [
            ("claude-3-haiku", 0.82, 0.0005),
            ("claude-3-sonnet", 0.90, 0.008),
        ]
        comparison = metric.compare_models(evaluations)
        # Haiku offers nearly the same quality at 16x lower cost
        assert comparison.best_value == "claude-3-haiku"

    def test_recommendation_generated(self):
        metric = CostQualityMetric()
        evaluations = [
            ("model-a", 0.9, 0.01),
            ("model-b", 0.8, 0.002),
        ]
        comparison = metric.compare_models(evaluations)
        assert len(comparison.recommendation) > 0

    def test_same_model_best_in_all(self):
        metric = CostQualityMetric()
        evaluations = [
            ("super-model", 0.99, 0.0001),
            ("bad-model", 0.3, 0.05),
        ]
        comparison = metric.compare_models(evaluations)
        assert comparison.best_value == "super-model"
        assert comparison.best_quality == "super-model"
        assert comparison.cheapest == "super-model"


class TestCostEstimation:
    """Test cost estimation from model and tokens."""

    def test_estimate_haiku_cost(self):
        metric = CostQualityMetric()
        cost = metric.estimate_cost("claude-3-haiku", tokens=1000)
        assert abs(cost - 0.000625) < 0.0001

    def test_estimate_opus_cost(self):
        metric = CostQualityMetric()
        cost = metric.estimate_cost("claude-3-opus", tokens=2000)
        expected = 2.0 * 0.045  # 2K tokens
        assert abs(cost - expected) < 0.001

    def test_estimate_unknown_model(self):
        metric = CostQualityMetric()
        cost = metric.estimate_cost("unknown-model", tokens=1000)
        # Falls back to default 0.001 per 1K tokens
        assert cost == pytest.approx(0.001, abs=0.0001)


class TestRecommendations:
    """Test recommendation generation."""

    def test_excellent_recommendation(self):
        metric = CostQualityMetric()
        result = metric.evaluate(quality_score=0.95, cost_usd=0.0005, model="haiku")
        assert "excellent" in result.recommendation.lower()

    def test_wasteful_recommendation(self):
        metric = CostQualityMetric()
        result = metric.evaluate(quality_score=0.1, cost_usd=0.09, model="opus")
        assert "wasteful" in result.recommendation.lower() or "switch" in result.recommendation.lower()

    def test_poor_quality_recommendation(self):
        metric = CostQualityMetric()
        result = metric.evaluate(quality_score=0.3, cost_usd=0.05, model="bad-model")
        assert "quality" in result.recommendation.lower() or "capable" in result.recommendation.lower()
