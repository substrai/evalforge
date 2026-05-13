"""Tests for synthetic data generators."""

import pytest
from evalforge.generators.synthetic import SyntheticGenerator, GenerationConfig, GeneratedDataset
from evalforge.generators.adversarial import AdversarialGenerator, AdversarialCase
from evalforge.generators.templates import TemplateGenerator, QuestionTemplate


class TestTemplateGenerator:
    def test_generate_rag_questions(self):
        gen = TemplateGenerator(use_case="rag")
        results = gen.generate(count=20)
        assert len(results) > 0
        assert len(results) <= 20
        assert all("query" in r for r in results)
        assert all("category" in r for r in results)

    def test_generate_by_category(self):
        gen = TemplateGenerator(use_case="rag")
        results = gen.generate(count=10, category="simple")
        assert len(results) > 0
        assert all(r["category"] == "simple" for r in results)

    def test_generate_adversarial(self):
        gen = TemplateGenerator(use_case="rag")
        results = gen.generate(count=5, category="adversarial")
        assert len(results) > 0
        assert all(r["category"] == "adversarial" for r in results)

    def test_custom_template(self):
        custom = [QuestionTemplate(
            template="What is {thing}?",
            category="simple",
            variables={"thing": ["Python", "Java", "Rust"]},
        )]
        gen = TemplateGenerator(use_case="rag", custom_templates=custom)
        results = gen.generate(count=5)
        assert len(results) > 0
        assert all("What is" in r["query"] for r in results)

    def test_generate_by_category_counts(self):
        gen = TemplateGenerator(use_case="rag")
        results = gen.generate_by_category({"simple": 5, "complex": 3})
        assert len(results) > 0


class TestAdversarialGenerator:
    def test_generate_all(self):
        gen = AdversarialGenerator()
        cases = gen.generate()
        assert len(cases) > 10

    def test_filter_by_attack_type(self):
        gen = AdversarialGenerator()
        cases = gen.generate(attack_type="prompt_injection")
        assert all(c.attack_type == "prompt_injection" for c in cases)
        assert len(cases) >= 3

    def test_filter_by_severity(self):
        gen = AdversarialGenerator()
        cases = gen.generate(severity="critical")
        assert all(c.severity == "critical" for c in cases)

    def test_generate_with_count(self):
        gen = AdversarialGenerator()
        cases = gen.generate(count=5)
        assert len(cases) == 5

    def test_generate_as_samples(self):
        gen = AdversarialGenerator()
        samples = gen.generate_as_samples(count=5)
        assert len(samples) == 5
        assert all("query" in s for s in samples)
        assert all(s["category"] == "adversarial" for s in samples)

    def test_get_attack_types(self):
        gen = AdversarialGenerator()
        types = gen.get_attack_types()
        assert "prompt_injection" in types
        assert "boundary" in types
        assert "confusion" in types

    def test_get_stats(self):
        gen = AdversarialGenerator()
        stats = gen.get_stats()
        assert sum(stats.values()) > 0

    def test_add_custom_case(self):
        gen = AdversarialGenerator()
        initial_count = len(gen.generate())
        gen.add_case(AdversarialCase(
            query="Custom attack",
            attack_type="custom",
            description="Test",
        ))
        assert len(gen.generate()) == initial_count + 1


class TestSyntheticGenerator:
    def test_generate_produces_samples(self):
        gen = SyntheticGenerator(use_case="rag")
        dataset = gen.generate(count=20)
        assert dataset.count > 0
        assert dataset.count <= 20

    def test_generate_with_config(self):
        config = GenerationConfig(
            use_case="rag",
            total_count=30,
            category_distribution={"simple": 0.5, "complex": 0.3, "adversarial": 0.2},
            seed=42,
        )
        gen = SyntheticGenerator(use_case="rag", config=config)
        dataset = gen.generate()
        assert dataset.count > 0
        assert dataset.count <= 30

    def test_has_multiple_categories(self):
        gen = SyntheticGenerator(use_case="rag")
        dataset = gen.generate(count=40)
        categories = set(s.get("category") for s in dataset.samples)
        assert len(categories) >= 2

    def test_adversarial_only(self):
        gen = SyntheticGenerator(use_case="rag")
        dataset = gen.generate_adversarial_only(count=10)
        assert dataset.count > 0
        assert all(s["category"] == "adversarial" for s in dataset.samples)

    def test_dataset_summary(self):
        gen = SyntheticGenerator(use_case="rag")
        dataset = gen.generate(count=20)
        summary = dataset.summary()
        assert "Generated Dataset" in summary

    def test_dataset_to_json(self):
        gen = SyntheticGenerator(use_case="rag")
        dataset = gen.generate(count=5)
        json_str = dataset.to_json()
        assert '"samples"' in json_str

    def test_dataset_save_and_load(self, tmp_path):
        gen = SyntheticGenerator(use_case="rag")
        dataset = gen.generate(count=10)
        path = tmp_path / "test_data.json"
        dataset.save(path)

        loaded = GeneratedDataset.load(path)
        assert loaded.count == dataset.count

    def test_filter_by_category(self):
        gen = SyntheticGenerator(use_case="rag")
        dataset = gen.generate(count=30)
        adversarial = dataset.filter_by_category("adversarial")
        assert all(s["category"] == "adversarial" for s in adversarial)

    def test_summarization_use_case(self):
        gen = SyntheticGenerator(use_case="summarization")
        dataset = gen.generate(count=10)
        assert dataset.count > 0
