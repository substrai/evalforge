"""Synthetic test data generation for EvalForge.

Generates high-quality test datasets tailored to the use case:
adversarial inputs, edge cases, domain-specific questions.
"""

from evalforge.generators.synthetic import SyntheticGenerator, GenerationConfig
from evalforge.generators.adversarial import AdversarialGenerator
from evalforge.generators.templates import TemplateGenerator, QuestionTemplate

__all__ = [
    "SyntheticGenerator",
    "GenerationConfig",
    "AdversarialGenerator",
    "TemplateGenerator",
    "QuestionTemplate",
]
