"""Tests for LLM-powered synthetic test data generator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evalforge.synthetic.data_generator import (
    CategoryTag,
    DataGenerator,
    DifficultyLevel,
    GeneratedTestCase,
    GenerationConfig,
    GenerationTemplate,
)


class TestGenerationTemplate:
    """Tests for the GenerationTemplate class."""

    def test_render_basic_template(self):
        """Test basic template rendering with variables."""
        template = GenerationTemplate(
            name="greeting",
            template="Hello {{name}}, welcome to {{place}}!",
            variables={
                "name": ["Alice", "Bob"],
                "place": ["Python", "Rust"],
            },
        )
        result = template.render(seed=42)
        assert "Hello" in result
        assert "welcome to" in result

    def test_render_deterministic_with_seed(self):
        """Test that rendering is deterministic with same seed."""
        template = GenerationTemplate(
            name="test",
            template="{{word}} is {{adj}}",
            variables={
                "word": ["cat", "dog", "fish"],
                "adj": ["big", "small", "fast"],
            },
        )
        r1 = template.render(seed=123)
        r2 = template.render(seed=123)
        assert r1 == r2

    def test_render_without_variables(self):
        """Test template with no variables renders as-is."""
        template = GenerationTemplate(
            name="static",
            template="This is a static prompt.",
        )
        assert template.render() == "This is a static prompt."


class TestGeneratedTestCase:
    """Tests for the GeneratedTestCase dataclass."""

    def test_auto_generates_fingerprint(self):
        """Test that fingerprint is generated automatically."""
        case = GeneratedTestCase(
            id="",
            input_text="test input",
            difficulty=DifficultyLevel.EASY,
        )
        assert case.fingerprint != ""
        assert len(case.fingerprint) == 12

    def test_auto_generates_id(self):
        """Test that ID is generated from fingerprint."""
        case = GeneratedTestCase(
            id="",
            input_text="test input",
            difficulty=DifficultyLevel.EASY,
        )
        assert case.id.startswith("tc-")

    def test_same_input_same_fingerprint(self):
        """Test that identical inputs produce identical fingerprints."""
        c1 = GeneratedTestCase(id="", input_text="hello", difficulty=DifficultyLevel.EASY)
        c2 = GeneratedTestCase(id="", input_text="hello", difficulty=DifficultyLevel.HARD)
        assert c1.fingerprint == c2.fingerprint


class TestDataGenerator:
    """Tests for the DataGenerator class."""

    def test_generate_adversarial(self):
        """Test adversarial input generation."""
        config = GenerationConfig(seed=42)
        gen = DataGenerator(config=config)
        cases = gen.generate_adversarial(count=5)

        assert len(cases) == 5
        for case in cases:
            assert case.difficulty == DifficultyLevel.ADVERSARIAL
            assert CategoryTag.ADVERSARIAL.value in case.categories

    def test_generate_edge_cases(self):
        """Test edge case generation."""
        config = GenerationConfig(seed=42, min_length=0)
        gen = DataGenerator(config=config)
        cases = gen.generate_edge_cases(count=5)

        assert len(cases) >= 1
        for case in cases:
            assert case.difficulty == DifficultyLevel.HARD
            assert CategoryTag.EDGE_CASE.value in case.categories

    def test_generate_boundary_values(self):
        """Test boundary value generation."""
        config = GenerationConfig(seed=42, min_length=0)
        gen = DataGenerator(config=config)
        cases = gen.generate_boundary_values(count=5)

        assert len(cases) >= 1
        for case in cases:
            assert CategoryTag.BOUNDARY.value in case.categories

    def test_generate_from_template(self):
        """Test generation from a custom template."""
        template = GenerationTemplate(
            name="question",
            template="What is the {{topic}} of {{subject}}?",
            variables={
                "topic": ["capital", "population", "area"],
                "subject": ["France", "Japan", "Brazil"],
            },
            category=CategoryTag.DOMAIN_SPECIFIC,
            difficulty=DifficultyLevel.EASY,
        )
        config = GenerationConfig(seed=42)
        gen = DataGenerator(config=config, templates=[template])
        cases = gen.generate_from_template(template, count=3)

        assert len(cases) == 3
        for case in cases:
            assert "What is the" in case.input_text
            assert case.template_name == "question"

    def test_deduplication(self):
        """Test that duplicate inputs are removed."""
        template = GenerationTemplate(
            name="static",
            template="Always the same input",
            category=CategoryTag.DOMAIN_SPECIFIC,
        )
        config = GenerationConfig(seed=42, deduplicate=True)
        gen = DataGenerator(config=config)
        cases = gen.generate_from_template(template, count=5)

        # Only one should pass dedup since all are identical
        assert len(cases) == 1

    def test_generate_batch(self):
        """Test batch generation with difficulty distribution."""
        template = GenerationTemplate(
            name="q",
            template="Question {{num}}",
            variables={"num": [str(i) for i in range(100)]},
        )
        config = GenerationConfig(seed=42, batch_size=10, min_length=0)
        gen = DataGenerator(config=config, templates=[template])
        cases = gen.generate_batch()

        assert len(cases) <= 10
        # Should have mix of difficulties
        difficulties = {c.difficulty for c in cases}
        assert len(difficulties) >= 1

    def test_generate_with_llm(self):
        """Test LLM-based generation with mock."""
        mock_llm = MagicMock(side_effect=lambda p, c: f"Generated text {c['index']}")
        config = GenerationConfig(seed=42)
        gen = DataGenerator(config=config, llm_generator=mock_llm)

        cases = gen.generate_with_llm(
            prompt_template="Generate test for {{domain}}",
            context={"domain": "healthcare"},
            count=3,
        )

        assert len(cases) == 3
        assert mock_llm.call_count == 3
        assert "Generated text 0" in cases[0].input_text

    def test_generate_with_llm_no_generator_raises(self):
        """Test that missing LLM generator raises RuntimeError."""
        gen = DataGenerator()

        with pytest.raises(RuntimeError, match="No LLM generator configured"):
            gen.generate_with_llm("prompt", {}, count=1)

    def test_reset_clears_state(self):
        """Test that reset clears fingerprints and RNG."""
        config = GenerationConfig(seed=42, min_length=0)
        gen = DataGenerator(config=config)
        gen.generate_edge_cases(count=3)

        stats_before = gen.get_stats()
        assert stats_before["total_generated"] > 0

        gen.reset()
        stats_after = gen.get_stats()
        assert stats_after["total_generated"] == 0

    def test_max_length_enforcement(self):
        """Test that generated inputs respect max_length."""
        template = GenerationTemplate(
            name="long",
            template="x" * 5000,
        )
        config = GenerationConfig(max_length=100, min_length=0)
        gen = DataGenerator(config=config)
        cases = gen.generate_from_template(template, count=1)

        assert len(cases) == 1
        assert len(cases[0].input_text) <= 100

    def test_add_template(self):
        """Test adding templates to the generator."""
        gen = DataGenerator()
        assert len(gen.templates) == 0

        template = GenerationTemplate(name="test", template="Hello")
        gen.add_template(template)
        assert len(gen.templates) == 1

    def test_get_stats(self):
        """Test generation statistics reporting."""
        gen = DataGenerator(
            config=GenerationConfig(seed=42),
            templates=[GenerationTemplate(name="t", template="x")],
        )
        stats = gen.get_stats()

        assert stats["total_generated"] == 0
        assert stats["templates_count"] == 1
        assert stats["seed"] == 42
        assert stats["has_llm_generator"] is False
