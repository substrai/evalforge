"""Factual consistency metric using NLI-based scoring for EvalForge.

Cross-references claims in generated text against source documents,
scoring faithfulness through sentence-level claim extraction and
entailment scoring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class EntailmentLabel(str, Enum):
    """NLI entailment classification labels."""

    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    NEUTRAL = "neutral"


@dataclass
class Claim:
    """A single factual claim extracted from text."""

    text: str
    source_sentence: str
    index: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntailmentScore:
    """Score for a single claim against source documents."""

    claim: Claim
    label: EntailmentLabel
    score: float
    supporting_evidence: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsistencyResult:
    """Aggregate factual consistency result."""

    overall_score: float
    claim_scores: List[EntailmentScore]
    num_claims: int
    num_entailed: int
    num_contradicted: int
    num_neutral: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def faithfulness_ratio(self) -> float:
        """Ratio of entailed claims to total claims."""
        if self.num_claims == 0:
            return 1.0
        return self.num_entailed / self.num_claims

    @property
    def contradiction_ratio(self) -> float:
        """Ratio of contradicted claims to total claims."""
        if self.num_claims == 0:
            return 0.0
        return self.num_contradicted / self.num_claims


class ClaimExtractor:
    """Extracts factual claims from text.

    Uses sentence segmentation and heuristic filtering to identify
    statements that make factual assertions.
    """

    def __init__(
        self,
        min_claim_length: int = 10,
        max_claim_length: int = 500,
        filter_opinions: bool = True,
    ):
        self.min_claim_length = min_claim_length
        self.max_claim_length = max_claim_length
        self.filter_opinions = filter_opinions

        # Patterns that indicate opinions rather than facts
        self._opinion_patterns = [
            r"\b(I think|I believe|in my opinion|probably|maybe|perhaps)\b",
            r"\b(might be|could be|seems like|appears to)\b",
        ]

        # Patterns that indicate factual claims
        self._fact_patterns = [
            r"\b(is|are|was|were|has|have|had)\b",
            r"\b\d+(\.\d+)?%?\b",  # Numbers/percentages
            r"\b(according to|research shows|studies indicate)\b",
        ]

    def extract(self, text: str) -> List[Claim]:
        """Extract factual claims from text.

        Args:
            text: The text to extract claims from.

        Returns:
            List of extracted claims.
        """
        sentences = self._split_sentences(text)
        claims = []

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not self._is_valid_claim(sentence):
                continue

            confidence = self._compute_confidence(sentence)
            claims.append(
                Claim(
                    text=sentence,
                    source_sentence=sentence,
                    index=i,
                    confidence=confidence,
                )
            )

        return claims

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common abbreviations
        text = re.sub(r'(?<=[.!?])\s+', '\n', text)
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        return sentences

    def _is_valid_claim(self, sentence: str) -> bool:
        """Check if a sentence is a valid factual claim."""
        if len(sentence) < self.min_claim_length:
            return False
        if len(sentence) > self.max_claim_length:
            return False
        if sentence.endswith("?"):
            return False

        if self.filter_opinions:
            for pattern in self._opinion_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    return False

        return True

    def _compute_confidence(self, sentence: str) -> float:
        """Compute confidence that a sentence is a factual claim."""
        score = 0.5  # Base confidence

        for pattern in self._fact_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                score += 0.15

        # Boost for sentences with specific entities or numbers
        if re.search(r'\b[A-Z][a-z]+\b', sentence):
            score += 0.1

        return min(score, 1.0)


class EntailmentScorer:
    """Scores claims against source documents using NLI-style comparison.

    Uses text overlap and semantic similarity heuristics as a lightweight
    alternative to full NLI model inference.
    """

    def __init__(
        self,
        entailment_threshold: float = 0.7,
        contradiction_threshold: float = 0.3,
        scoring_fn: Optional[Callable[[str, str], float]] = None,
    ):
        """Initialize the scorer.

        Args:
            entailment_threshold: Score above which a claim is entailed.
            contradiction_threshold: Score below which a claim is contradicted.
            scoring_fn: Custom scoring function(claim, evidence) -> float.
        """
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self._scoring_fn = scoring_fn or self._default_scoring

    def score_claim(
        self, claim: Claim, source_sentences: List[str]
    ) -> EntailmentScore:
        """Score a single claim against source sentences.

        Args:
            claim: The claim to score.
            source_sentences: Sentences from the source document.

        Returns:
            EntailmentScore with label and score.
        """
        best_score = 0.0
        best_evidence: List[str] = []

        for source in source_sentences:
            score = self._scoring_fn(claim.text, source)
            if score > best_score:
                best_score = score
                best_evidence = [source]
            elif score == best_score and score > 0:
                best_evidence.append(source)

        # Determine label based on thresholds
        if best_score >= self.entailment_threshold:
            label = EntailmentLabel.ENTAILED
        elif best_score <= self.contradiction_threshold:
            label = EntailmentLabel.CONTRADICTED
        else:
            label = EntailmentLabel.NEUTRAL

        return EntailmentScore(
            claim=claim,
            label=label,
            score=best_score,
            supporting_evidence=best_evidence[:3],
            details={"raw_score": best_score},
        )

    def _default_scoring(self, claim: str, source: str) -> float:
        """Default scoring using token overlap (Jaccard-like).

        This is a lightweight proxy; production systems should use
        a proper NLI model.
        """
        claim_tokens = set(self._tokenize(claim))
        source_tokens = set(self._tokenize(source))

        if not claim_tokens or not source_tokens:
            return 0.0

        intersection = claim_tokens & source_tokens
        union = claim_tokens | source_tokens

        # Weighted Jaccard: emphasize claim coverage
        claim_coverage = len(intersection) / len(claim_tokens) if claim_tokens else 0
        jaccard = len(intersection) / len(union) if union else 0

        # Blend coverage and jaccard
        return 0.6 * claim_coverage + 0.4 * jaccard

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for scoring."""
        # Lowercase, remove punctuation, split on whitespace
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = text.split()
        # Filter stopwords (minimal set)
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or", "but"}
        return [t for t in tokens if t not in stopwords and len(t) > 1]


class FactualConsistencyMetric:
    """Evaluates factual consistency between generated text and source documents.

    Combines claim extraction with entailment scoring to produce an
    aggregate faithfulness score.
    """

    def __init__(
        self,
        extractor: Optional[ClaimExtractor] = None,
        scorer: Optional[EntailmentScorer] = None,
        aggregate_method: str = "mean",
    ):
        """Initialize the metric.

        Args:
            extractor: ClaimExtractor instance. Uses default if None.
            scorer: EntailmentScorer instance. Uses default if None.
            aggregate_method: How to aggregate scores - 'mean', 'min', or 'weighted'.
        """
        self.extractor = extractor or ClaimExtractor()
        self.scorer = scorer or EntailmentScorer()
        self.aggregate_method = aggregate_method

    def evaluate(
        self, generated_text: str, source_text: str
    ) -> ConsistencyResult:
        """Evaluate factual consistency of generated text against source.

        Args:
            generated_text: The generated/output text to evaluate.
            source_text: The source/reference document.

        Returns:
            ConsistencyResult with scores and details.
        """
        # Extract claims from generated text
        claims = self.extractor.extract(generated_text)

        if not claims:
            return ConsistencyResult(
                overall_score=1.0,
                claim_scores=[],
                num_claims=0,
                num_entailed=0,
                num_contradicted=0,
                num_neutral=0,
                metadata={"note": "No claims extracted from generated text"},
            )

        # Split source into sentences for comparison
        source_sentences = self.extractor._split_sentences(source_text)

        # Score each claim
        claim_scores = []
        for claim in claims:
            score = self.scorer.score_claim(claim, source_sentences)
            claim_scores.append(score)

        # Compute aggregates
        num_entailed = sum(
            1 for s in claim_scores if s.label == EntailmentLabel.ENTAILED
        )
        num_contradicted = sum(
            1 for s in claim_scores if s.label == EntailmentLabel.CONTRADICTED
        )
        num_neutral = sum(
            1 for s in claim_scores if s.label == EntailmentLabel.NEUTRAL
        )

        overall_score = self._aggregate_scores(claim_scores)

        return ConsistencyResult(
            overall_score=overall_score,
            claim_scores=claim_scores,
            num_claims=len(claims),
            num_entailed=num_entailed,
            num_contradicted=num_contradicted,
            num_neutral=num_neutral,
            metadata={
                "aggregate_method": self.aggregate_method,
                "source_sentences": len(source_sentences),
            },
        )

    def evaluate_batch(
        self,
        generated_texts: List[str],
        source_texts: List[str],
    ) -> List[ConsistencyResult]:
        """Evaluate multiple generated texts against their sources.

        Args:
            generated_texts: List of generated texts.
            source_texts: List of corresponding source texts.

        Returns:
            List of ConsistencyResult objects.
        """
        if len(generated_texts) != len(source_texts):
            raise ValueError(
                f"Mismatched lengths: {len(generated_texts)} generated vs {len(source_texts)} sources"
            )
        return [
            self.evaluate(gen, src)
            for gen, src in zip(generated_texts, source_texts)
        ]

    def _aggregate_scores(self, claim_scores: List[EntailmentScore]) -> float:
        """Aggregate individual claim scores into an overall score."""
        if not claim_scores:
            return 1.0

        scores = [s.score for s in claim_scores]

        if self.aggregate_method == "min":
            return min(scores)
        elif self.aggregate_method == "weighted":
            weights = [s.claim.confidence for s in claim_scores]
            total_weight = sum(weights)
            if total_weight == 0:
                return 0.0
            return sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:  # mean
            return sum(scores) / len(scores)
