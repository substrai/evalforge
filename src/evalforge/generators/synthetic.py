"""Synthetic test data generator.

Orchestrates template-based and adversarial generation to produce
complete test datasets with configurable category distribution.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalforge.generators.templates import TemplateGenerator
from evalforge.generators.adversarial import AdversarialGenerator


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation."""

    use_case: str = "rag"
    total_count: int = 100
    category_distribution: Dict[str, float] = field(default_factory=lambda: {
        "simple": 0.30,
        "complex": 0.25,
        "adversarial": 0.25,
        "edge_case": 0.20,
    })
    seed: Optional[int] = None
    include_context: bool = True
    include_reference: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationConfig":
        return cls(
            use_case=data.get("use_case", "rag"),
            total_count=data.get("count", 100),
            category_distribution=data.get("category_distribution", cls.__dataclass_fields__["category_distribution"].default_factory()),
            seed=data.get("seed"),
        )

    @property
    def category_counts(self) -> Dict[str, int]:
        """Calculate exact counts per category."""
        counts = {}
        remaining = self.total_count
        categories = list(self.category_distribution.items())
        for i, (cat, pct) in enumerate(categories):
            if i == len(categories) - 1:
                counts[cat] = remaining
            else:
                count = int(self.total_count * pct)
                counts[cat] = count
                remaining -= count
        return counts


@dataclass
class GeneratedDataset:
    """A generated test dataset."""

    samples: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.samples)

    def filter_by_category(self, category: str) -> List[Dict[str, Any]]:
        return [s for s in self.samples if s.get("category") == category]

    def to_json(self) -> str:
        return json.dumps({"samples": self.samples, "metadata": self.metadata}, indent=2)

    def save(self, path: str | Path) -> None:
        """Save dataset to JSON file."""
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "GeneratedDataset":
        """Load dataset from JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(samples=data.get("samples", []), metadata=data.get("metadata", {}))

    def summary(self) -> str:
        categories = {}
        for s in self.samples:
            cat = s.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        lines = [
            f"Generated Dataset: {self.count} samples",
            "Categories:",
        ]
        for cat, count in sorted(categories.items()):
            lines.append(f"  • {cat}: {count}")
        return "\n".join(lines)


# Sample contexts for RAG test data
SAMPLE_CONTEXTS = [
    "Return Policy: Customers may return items within 30 days of purchase. A valid receipt is required. Items must be in original condition. Electronics have a 15-day return window.",
    "Password Reset: Navigate to Settings > Security > Reset Password. You will receive an email with a reset link valid for 24 hours. If you don't receive the email, check your spam folder.",
    "Pricing Plans: Basic ($9/month): 10GB storage, email support. Premium ($29/month): Unlimited storage, priority support, advanced analytics. Enterprise (custom): Dedicated account manager, SLA.",
    "Shipping: Standard shipping takes 5-7 business days. Express shipping (2-3 days) available for $12.99. Free shipping on orders over $50. International shipping available to 40+ countries.",
    "Account Deletion: To delete your account, go to Settings > Account > Delete Account. This action is irreversible. All data will be permanently removed within 30 days.",
]


class SyntheticGenerator:
    """Generates complete synthetic test datasets.

    Combines template-based generation, adversarial cases, and
    context/reference generation into a complete dataset.

    Usage:
        gen = SyntheticGenerator(use_case="rag")
        dataset = gen.generate(count=100)
        dataset.save("data/synthetic/test_v1.json")
    """

    def __init__(self, use_case: str = "rag", config: Optional[GenerationConfig] = None):
        self.use_case = use_case
        self.config = config or GenerationConfig(use_case=use_case)
        self._template_gen = TemplateGenerator(use_case=use_case)
        self._adversarial_gen = AdversarialGenerator()

        if self.config.seed is not None:
            random.seed(self.config.seed)

    def generate(self, count: Optional[int] = None) -> GeneratedDataset:
        """Generate a complete test dataset.

        Args:
            count: Override total count (uses config if None)

        Returns:
            GeneratedDataset with all samples
        """
        total = count or self.config.total_count
        config = GenerationConfig(
            use_case=self.use_case,
            total_count=total,
            category_distribution=self.config.category_distribution,
        )

        samples: List[Dict[str, Any]] = []
        category_counts = config.category_counts

        # Generate template-based questions (simple + complex)
        simple_count = category_counts.get("simple", 0)
        complex_count = category_counts.get("complex", 0)

        if simple_count > 0:
            simple_qs = self._template_gen.generate(count=simple_count, category="simple")
            for q in simple_qs:
                samples.append(self._enrich_sample(q))

        if complex_count > 0:
            complex_qs = self._template_gen.generate(count=complex_count, category="complex")
            for q in complex_qs:
                samples.append(self._enrich_sample(q))

        # Generate adversarial cases
        adversarial_count = category_counts.get("adversarial", 0)
        if adversarial_count > 0:
            adv_cases = self._adversarial_gen.generate(count=adversarial_count)
            for case in adv_cases:
                samples.append({
                    "query": case.query,
                    "response": "",
                    "context": "",
                    "reference": "",
                    "category": "adversarial",
                    "metadata": {
                        "attack_type": case.attack_type,
                        "severity": case.severity,
                    },
                })

        # Generate edge cases
        edge_count = category_counts.get("edge_case", 0)
        if edge_count > 0:
            edge_qs = self._template_gen.generate(count=edge_count, category="edge_case")
            for q in edge_qs:
                q["category"] = "edge_case"
                samples.append(self._enrich_sample(q))

        # Shuffle
        random.shuffle(samples)
        samples = samples[:total]

        metadata = {
            "use_case": self.use_case,
            "total_count": len(samples),
            "generated_at": time.time(),
            "category_distribution": {
                cat: len([s for s in samples if s.get("category") == cat])
                for cat in set(s.get("category", "") for s in samples)
            },
        }

        return GeneratedDataset(samples=samples, metadata=metadata)

    def generate_adversarial_only(self, count: int = 25) -> GeneratedDataset:
        """Generate only adversarial test cases."""
        cases = self._adversarial_gen.generate(count=count)
        samples = [
            {
                "query": case.query,
                "response": "",
                "context": "",
                "reference": "",
                "category": "adversarial",
                "metadata": {"attack_type": case.attack_type, "severity": case.severity},
            }
            for case in cases
        ]
        return GeneratedDataset(
            samples=samples,
            metadata={"type": "adversarial_only", "count": len(samples)},
        )

    def _enrich_sample(self, question: Dict[str, str]) -> Dict[str, Any]:
        """Add context and reference to a generated question."""
        context = random.choice(SAMPLE_CONTEXTS) if self.config.include_context else ""
        reference = ""
        if self.config.include_reference and context:
            # Use first sentence of context as reference
            sentences = context.split(". ")
            reference = sentences[0] + "." if sentences else ""

        return {
            "query": question.get("query", ""),
            "response": "",  # to be filled by system under test
            "context": context,
            "reference": reference,
            "category": question.get("category", "simple"),
            "metadata": {"difficulty": question.get("difficulty", "medium")},
        }
