"""Tests for factual consistency metric."""

import pytest

from evalforge.metrics.factual_consistency import (
    Claim,
    ClaimExtractor,
    ConsistencyResult,
    EntailmentLabel,
    EntailmentScore,
    EntailmentScorer,
    FactualConsistencyMetric,
)


@pytest.fixture
def extractor():
    return ClaimExtractor()


@pytest.fixture
def scorer():
    return EntailmentScorer()


@pytest.fixture
def metric():
    return FactualConsistencyMetric()


class TestClaimExtractor:
    def test_extracts_factual_sentences(self, extractor):
        text = "Python is a programming language. It was created by Guido van Rossum."
        claims = extractor.extract(text)
        assert len(claims) >= 1
        assert all(isinstance(c, Claim) for c in claims)

    def test_filters_questions(self, extractor):
        text = "What is Python? It is a programming language."
        claims = extractor.extract(text)
        # Question should be filtered out
        for claim in claims:
            assert not claim.text.endswith("?")

    def test_filters_opinions(self, extractor):
        text = "I think Python is great. Python is a programming language."
        claims = extractor.extract(text)
        for claim in claims:
            assert "I think" not in claim.text

    def test_respects_min_length(self):
        extractor = ClaimExtractor(min_claim_length=20)
        text = "Short. This is a longer sentence that should pass the filter."
        claims = extractor.extract(text)
        for claim in claims:
            assert len(claim.text) >= 20

    def test_confidence_scoring(self, extractor):
        text = "The population of Tokyo is 13.96 million people."
        claims = extractor.extract(text)
        assert len(claims) >= 1
        # Should have higher confidence due to numbers and proper nouns
        assert claims[0].confidence > 0.5

    def test_empty_text_returns_no_claims(self, extractor):
        claims = extractor.extract("")
        assert claims == []

    def test_claim_indices(self, extractor):
        text = "First sentence here. Second sentence follows. Third one arrives."
        claims = extractor.extract(text)
        indices = [c.index for c in claims]
        assert indices == sorted(indices)


class TestEntailmentScorer:
    def test_high_overlap_is_entailed(self, scorer):
        claim = Claim(text="Python is a programming language", source_sentence="", index=0)
        sources = ["Python is a popular programming language used worldwide"]
        result = scorer.score_claim(claim, sources)
        assert result.label == EntailmentLabel.ENTAILED
        assert result.score >= 0.7

    def test_no_overlap_is_contradicted(self, scorer):
        claim = Claim(text="Mars is the largest planet", source_sentence="", index=0)
        sources = ["The weather today is sunny and warm"]
        result = scorer.score_claim(claim, sources)
        assert result.label == EntailmentLabel.CONTRADICTED
        assert result.score <= 0.3

    def test_partial_overlap_is_neutral(self, scorer):
        claim = Claim(text="Python was created in 1991 by Guido", source_sentence="", index=0)
        sources = ["Python is a language. Java was created in 1995."]
        result = scorer.score_claim(claim, sources)
        assert result.label == EntailmentLabel.NEUTRAL

    def test_custom_scoring_function(self):
        def custom_fn(claim: str, source: str) -> float:
            return 1.0 if claim in source else 0.0

        scorer = EntailmentScorer(scoring_fn=custom_fn)
        claim = Claim(text="hello world", source_sentence="", index=0)
        sources = ["hello world is a common phrase"]
        result = scorer.score_claim(claim, sources)
        assert result.score == 1.0

    def test_supporting_evidence_returned(self, scorer):
        claim = Claim(text="Python is interpreted", source_sentence="", index=0)
        sources = ["Python is an interpreted language", "Java is compiled"]
        result = scorer.score_claim(claim, sources)
        assert len(result.supporting_evidence) >= 1

    def test_empty_sources(self, scorer):
        claim = Claim(text="Some factual claim here", source_sentence="", index=0)
        result = scorer.score_claim(claim, [])
        assert result.score == 0.0
        assert result.label == EntailmentLabel.CONTRADICTED


class TestFactualConsistencyMetric:
    def test_consistent_text_scores_high(self, metric):
        source = "Python is a programming language created by Guido van Rossum."
        generated = "Python is a programming language. It was created by Guido van Rossum."
        result = metric.evaluate(generated, source)
        assert result.overall_score > 0.5
        assert isinstance(result, ConsistencyResult)

    def test_inconsistent_text_scores_low(self, metric):
        source = "The Earth orbits the Sun. Water boils at 100 degrees Celsius."
        generated = "Cats enjoy swimming in lava. The moon is made of cheese."
        result = metric.evaluate(generated, source)
        assert result.overall_score < 0.5

    def test_empty_generated_text(self, metric):
        source = "Some source text here."
        generated = ""
        result = metric.evaluate(generated, source)
        assert result.overall_score == 1.0
        assert result.num_claims == 0

    def test_result_counts(self, metric):
        source = "Python is a programming language. It supports multiple paradigms."
        generated = "Python is a programming language. Python runs on Mars."
        result = metric.evaluate(generated, source)
        assert result.num_claims > 0
        assert result.num_entailed + result.num_contradicted + result.num_neutral == result.num_claims

    def test_faithfulness_ratio(self, metric):
        source = "Python is a programming language created by Guido van Rossum."
        generated = "Python is a programming language created by Guido van Rossum."
        result = metric.evaluate(generated, source)
        assert result.faithfulness_ratio >= 0.0
        assert result.faithfulness_ratio <= 1.0

    def test_evaluate_batch(self, metric):
        sources = ["Python is fast.", "Java is compiled."]
        generated = ["Python is fast.", "Java is compiled."]
        results = metric.evaluate_batch(generated, sources)
        assert len(results) == 2

    def test_batch_mismatched_lengths(self, metric):
        with pytest.raises(ValueError, match="Mismatched lengths"):
            metric.evaluate_batch(["a", "b"], ["c"])

    def test_aggregate_methods(self):
        source = "Python is a programming language. It was created in 1991."
        generated = "Python is a programming language. It was created in 1991."

        mean_metric = FactualConsistencyMetric(aggregate_method="mean")
        min_metric = FactualConsistencyMetric(aggregate_method="min")
        weighted_metric = FactualConsistencyMetric(aggregate_method="weighted")

        r1 = mean_metric.evaluate(generated, source)
        r2 = min_metric.evaluate(generated, source)
        r3 = weighted_metric.evaluate(generated, source)

        # All should produce valid scores
        assert 0.0 <= r1.overall_score <= 1.0
        assert 0.0 <= r2.overall_score <= 1.0
        assert 0.0 <= r3.overall_score <= 1.0
        # min should be <= mean
        assert r2.overall_score <= r1.overall_score + 0.01
