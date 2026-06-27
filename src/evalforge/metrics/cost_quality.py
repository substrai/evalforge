"""Cost-per-quality metric — cost efficiency scoring.

Evaluates the ratio of quality score to cost, identifying the best
value model/prompt combinations. Helps teams optimize their LLM spending
by quantifying the efficiency of different configurations.

Usage:
    from evalforge.metrics.cost_quality import CostQualityMetric

    metric = CostQualityMetric(quality_weight=0.7, cost_weight=0.3)
    result = metric.evaluate(
        quality_score=0.92,
        cost_usd=0.003,
        model="claude-3-haiku",
    )
    print(f"Efficiency: {result.efficiency_score:.2f}")
    print(f"Cost per quality point: ${result.cost_per_quality_point:.6f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class EfficiencyTier(str, Enum):
    """Efficiency classification tiers."""

    EXCELLENT = "excellent"  # Top 10% efficiency
    GOOD = "good"  # 60-90th percentile
    ACCEPTABLE = "acceptable"  # 30-60th percentile
    POOR = "poor"  # Bottom 30%
    WASTEFUL = "wasteful"  # Extremely poor value


@dataclass
class CostQualityResult:
    """Result of a cost-quality evaluation."""

    quality_score: float  # 0.0 to 1.0
    cost_usd: float
    efficiency_score: float  # 0.0 to 1.0 (higher = better value)
    cost_per_quality_point: float  # USD per quality point (lower = better)
    tier: EfficiencyTier
    model: str
    tokens_used: int
    quality_per_dollar: float  # Quality points per dollar
    recommendation: str


@dataclass
class ModelComparison:
    """Comparison of cost-quality across multiple models."""

    results: List[CostQualityResult]
    best_value: str  # Model name with best efficiency
    best_quality: str  # Model name with best quality regardless of cost
    cheapest: str  # Model name with lowest cost
    recommendation: str


# Known model pricing (input + output average per 1K tokens)
MODEL_PRICING: Dict[str, float] = {
    "claude-3-opus": 0.045,
    "claude-3.5-sonnet": 0.009,
    "claude-3-sonnet": 0.009,
    "claude-3-haiku": 0.000625,
    "claude-3.5-haiku": 0.002,
    "gpt-4o": 0.0075,
    "gpt-4o-mini": 0.000375,
    "gpt-4-turbo": 0.02,
    "amazon-titan-text-lite": 0.0002,
    "amazon-titan-text-express": 0.0008,
    "llama-3-70b": 0.00265,
    "llama-3-8b": 0.0003,
    "mistral-large": 0.008,
    "mistral-small": 0.001,
}


class CostQualityMetric:
    """Evaluates cost efficiency relative to quality output.

    Computes a normalized efficiency score that balances quality achieved
    against cost incurred, enabling comparison across models and prompts.

    The efficiency score is calculated as:
        efficiency = quality_normalized * quality_weight + cost_normalized * cost_weight

    Where:
        quality_normalized = quality_score (already 0-1)
        cost_normalized = 1.0 - (cost / max_expected_cost)

    Args:
        quality_weight: Weight for quality in the score (0-1).
        cost_weight: Weight for cost in the score (0-1).
        max_expected_cost: Maximum expected cost per call for normalization.
    """

    def __init__(
        self,
        quality_weight: float = 0.7,
        cost_weight: float = 0.3,
        max_expected_cost: float = 0.10,
    ):
        if abs(quality_weight + cost_weight - 1.0) > 0.001:
            raise ValueError("quality_weight + cost_weight must equal 1.0")
        if max_expected_cost <= 0:
            raise ValueError("max_expected_cost must be positive")

        self._quality_weight = quality_weight
        self._cost_weight = cost_weight
        self._max_expected_cost = max_expected_cost
        self._history: List[CostQualityResult] = []

    @property
    def quality_weight(self) -> float:
        """The weight assigned to quality in the efficiency score."""
        return self._quality_weight

    @property
    def cost_weight(self) -> float:
        """The weight assigned to cost in the efficiency score."""
        return self._cost_weight

    @property
    def history(self) -> List[CostQualityResult]:
        """Historical evaluation results."""
        return self._history

    def evaluate(
        self,
        quality_score: float,
        cost_usd: float,
        model: str = "unknown",
        tokens_used: int = 0,
    ) -> CostQualityResult:
        """Evaluate the cost-quality efficiency of a single call.

        Args:
            quality_score: Quality metric result (0.0 to 1.0).
            cost_usd: Actual cost in USD.
            model: Model identifier.
            tokens_used: Total tokens consumed (input + output).

        Returns:
            CostQualityResult with efficiency analysis.
        """
        # Normalize cost (0 = max cost, 1 = free)
        cost_normalized = max(0.0, 1.0 - (cost_usd / self._max_expected_cost))

        # Compute weighted efficiency score
        efficiency_score = (
            quality_score * self._quality_weight
            + cost_normalized * self._cost_weight
        )
        efficiency_score = max(0.0, min(1.0, efficiency_score))

        # Cost per quality point
        cost_per_quality = cost_usd / quality_score if quality_score > 0 else float("inf")

        # Quality per dollar
        quality_per_dollar = quality_score / cost_usd if cost_usd > 0 else float("inf")

        # Determine tier
        tier = self._classify_tier(efficiency_score)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            quality_score, cost_usd, efficiency_score, model, tier
        )

        result = CostQualityResult(
            quality_score=quality_score,
            cost_usd=cost_usd,
            efficiency_score=efficiency_score,
            cost_per_quality_point=cost_per_quality,
            tier=tier,
            model=model,
            tokens_used=tokens_used,
            quality_per_dollar=quality_per_dollar,
            recommendation=recommendation,
        )

        self._history.append(result)
        return result

    def compare_models(
        self,
        evaluations: List[Tuple[str, float, float]],
    ) -> ModelComparison:
        """Compare cost-quality across multiple model evaluations.

        Args:
            evaluations: List of (model_name, quality_score, cost_usd) tuples.

        Returns:
            ModelComparison with rankings and recommendation.
        """
        results: List[CostQualityResult] = []
        for model, quality, cost in evaluations:
            result = self.evaluate(quality_score=quality, cost_usd=cost, model=model)
            results.append(result)

        # Find best in each category
        best_value_result = max(results, key=lambda r: r.efficiency_score)
        best_quality_result = max(results, key=lambda r: r.quality_score)
        cheapest_result = min(results, key=lambda r: r.cost_usd)

        # Generate recommendation
        if best_value_result.model == best_quality_result.model:
            recommendation = (
                f"{best_value_result.model} offers both the best quality and value. "
                f"Use it as your primary model."
            )
        else:
            recommendation = (
                f"Best value: {best_value_result.model} "
                f"(efficiency={best_value_result.efficiency_score:.2f}). "
                f"Best quality: {best_quality_result.model} "
                f"(quality={best_quality_result.quality_score:.2f}). "
                f"Consider {best_value_result.model} for most requests, "
                f"reserve {best_quality_result.model} for complex tasks."
            )

        return ModelComparison(
            results=results,
            best_value=best_value_result.model,
            best_quality=best_quality_result.model,
            cheapest=cheapest_result.model,
            recommendation=recommendation,
        )

    def estimate_cost(self, model: str, tokens: int) -> float:
        """Estimate cost for a model given token count.

        Args:
            model: Model identifier.
            tokens: Total token count (input + output).

        Returns:
            Estimated cost in USD.
        """
        price_per_1k = MODEL_PRICING.get(model, 0.001)
        return (tokens / 1000.0) * price_per_1k

    def _classify_tier(self, efficiency_score: float) -> EfficiencyTier:
        """Classify efficiency into tiers."""
        if efficiency_score >= 0.9:
            return EfficiencyTier.EXCELLENT
        elif efficiency_score >= 0.7:
            return EfficiencyTier.GOOD
        elif efficiency_score >= 0.5:
            return EfficiencyTier.ACCEPTABLE
        elif efficiency_score >= 0.3:
            return EfficiencyTier.POOR
        else:
            return EfficiencyTier.WASTEFUL

    def _generate_recommendation(
        self,
        quality: float,
        cost: float,
        efficiency: float,
        model: str,
        tier: EfficiencyTier,
    ) -> str:
        """Generate a human-readable recommendation."""
        if tier == EfficiencyTier.EXCELLENT:
            return f"{model} provides excellent cost-quality efficiency."
        elif tier == EfficiencyTier.GOOD:
            return f"{model} offers good value. Quality: {quality:.2f}, Cost: ${cost:.6f}."
        elif tier == EfficiencyTier.ACCEPTABLE:
            return (
                f"{model} is acceptable but could be optimized. "
                f"Consider a cheaper model if quality threshold allows."
            )
        elif tier == EfficiencyTier.POOR:
            if quality < 0.5:
                return (
                    f"{model} has low quality ({quality:.2f}) — "
                    f"consider a more capable model."
                )
            else:
                return (
                    f"{model} is expensive for the quality delivered. "
                    f"Try a smaller model variant."
                )
        else:
            return (
                f"{model} is wasteful — high cost (${cost:.4f}) with "
                f"low quality ({quality:.2f}). Switch models immediately."
            )
