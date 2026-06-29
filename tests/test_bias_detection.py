"""Tests for bias detection metric across demographic dimensions."""

from __future__ import annotations

import pytest

from evalforge.metrics.bias_detection import (
    BiasDetectionMetric,
    BiasReport,
    BiasType,
    DemographicDimension,
    DimensionResult,
)


class TestBiasDetectionInit:
    """Test metric initialization."""

    def test_default_dimensions(self):
        metric = BiasDetectionMetric()
        assert "gender" in metric.dimensions
        assert "age" in metric.dimensions
        assert "ethnicity" in metric.dimensions

    def test_custom_dimensions(self):
        metric = BiasDetectionMetric(dimensions=["gender", "religion"])
        assert metric.dimensions == ["gender", "religion"]

    def test_default_threshold(self):
        metric = BiasDetectionMetric()
        assert metric.threshold == 0.10

    def test_custom_threshold(self):
        metric = BiasDetectionMetric(threshold=0.05)
        assert metric.threshold == 0.05

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            BiasDetectionMetric(threshold=0.0)
        with pytest.raises(ValueError):
            BiasDetectionMetric(threshold=1.5)


class TestBiasDetection:
    """Test bias detection on pre-collected responses."""

    def test_no_bias_equal_scores(self):
        metric = BiasDetectionMetric(threshold=0.10)
        responses = {
            "gender": {
                "male": [0.90, 0.88, 0.91],
                "female": [0.89, 0.90, 0.88],
            }
        }
        report = metric.evaluate(responses)
        assert report.bias_detected is False
        assert report.passing is True

    def test_clear_bias_detected(self):
        metric = BiasDetectionMetric(threshold=0.10)
        responses = {
            "gender": {
                "male": [0.95, 0.93, 0.94],
                "female": [0.70, 0.72, 0.68],
            }
        }
        report = metric.evaluate(responses)
        assert report.bias_detected is True
        assert report.passing is False
        assert report.max_disparity > 0.10

    def test_multiple_dimensions(self):
        metric = BiasDetectionMetric(dimensions=["gender", "age"])
        responses = {
            "gender": {
                "male": [0.90, 0.88],
                "female": [0.89, 0.90],
            },
            "age": {
                "young": [0.92, 0.91],
                "senior": [0.70, 0.72],
            },
        }
        report = metric.evaluate(responses)
        assert report.dimensions_tested == 2
        assert report.dimensions_with_bias >= 1  # Age dimension is biased

    def test_identifies_worst_group(self):
        metric = BiasDetectionMetric(threshold=0.05)
        responses = {
            "ethnicity": {
                "group_a": [0.90, 0.91, 0.89],
                "group_b": [0.75, 0.73, 0.74],
                "group_c": [0.88, 0.87, 0.89],
            }
        }
        report = metric.evaluate(responses)
        biased_dim = report.dimension_results[0]
        assert biased_dim.worst_group == "group_b"
        assert biased_dim.best_group == "group_a"

    def test_empty_responses(self):
        metric = BiasDetectionMetric()
        report = metric.evaluate({})
        assert report.bias_detected is False
        assert report.dimensions_tested == 0
        assert report.overall_score == 1.0

    def test_single_group_no_bias(self):
        metric = BiasDetectionMetric()
        responses = {
            "gender": {
                "male": [0.90, 0.88],
            }
        }
        report = metric.evaluate(responses)
        assert report.bias_detected is False


class TestDimensionAnalysis:
    """Test single dimension analysis."""

    def test_disparity_calculation(self):
        metric = BiasDetectionMetric()
        result = metric.evaluate_single_dimension(
            "gender",
            {"male": [0.90], "female": [0.80]},
        )
        assert abs(result.disparity - 0.10) < 0.001

    def test_disparity_ratio(self):
        metric = BiasDetectionMetric()
        result = metric.evaluate_single_dimension(
            "age",
            {"young": [0.90], "senior": [0.45]},
        )
        assert abs(result.disparity_ratio - 2.0) < 0.01

    def test_sample_sizes_tracked(self):
        metric = BiasDetectionMetric()
        result = metric.evaluate_single_dimension(
            "gender",
            {"male": [0.9, 0.8, 0.85], "female": [0.7, 0.75]},
        )
        assert result.sample_sizes["male"] == 3
        assert result.sample_sizes["female"] == 2

    def test_bias_type_classified(self):
        metric = BiasDetectionMetric(threshold=0.05)
        result = metric.evaluate_single_dimension(
            "gender",
            {"male": [0.95, 0.93], "female": [0.70, 0.72]},
        )
        assert result.is_biased is True
        assert result.bias_type == BiasType.QUALITY_DISPARITY


class TestOverallScore:
    """Test overall bias score calculation."""

    def test_perfect_score_no_bias(self):
        metric = BiasDetectionMetric(threshold=0.10)
        responses = {
            "gender": {
                "male": [0.90, 0.90],
                "female": [0.90, 0.90],
            }
        }
        report = metric.evaluate(responses)
        assert report.overall_score == 1.0

    def test_low_score_high_bias(self):
        metric = BiasDetectionMetric(threshold=0.10)
        responses = {
            "gender": {
                "male": [0.95, 0.95],
                "female": [0.50, 0.50],
            }
        }
        report = metric.evaluate(responses)
        assert report.overall_score < 0.5


class TestRecommendations:
    """Test recommendation generation."""

    def test_recommendations_for_biased_dimension(self):
        metric = BiasDetectionMetric(threshold=0.05)
        responses = {
            "gender": {
                "male": [0.95, 0.93],
                "female": [0.70, 0.72],
            }
        }
        report = metric.evaluate(responses)
        assert len(report.recommendations) > 0
        assert "gender" in report.recommendations[0].lower()
        assert "debiasing" in report.recommendations[0].lower()

    def test_no_bias_recommendation(self):
        metric = BiasDetectionMetric()
        responses = {
            "gender": {
                "male": [0.90, 0.89],
                "female": [0.89, 0.90],
            }
        }
        report = metric.evaluate(responses)
        assert "no significant bias" in report.recommendations[0].lower()
