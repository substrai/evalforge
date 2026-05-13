"""RAG-specific metrics: faithfulness, answer relevancy, context precision/recall."""

from __future__ import annotations

import re
from typing import List, Set

from evalforge.metrics.base import BaseMetric, MetricInput, MetricOutput


class Faithfulness(BaseMetric):
    """Measures if the answer is grounded in the provided context.

    Score = proportion of claims in the answer that are supported by context.
    """

    name = "faithfulness"
    description = "Is the answer grounded in the retrieved context?"
    category = "rag"

    def evaluate(self, input: MetricInput, threshold: float = 0.85) -> MetricOutput:
        if not input.response or not input.context:
            return MetricOutput(score=0.0, passed=False, explanation="Missing response or context")

        # Extract key claims from response
        response_sentences = self._split_sentences(input.response)
        context_lower = input.context.lower()

        if not response_sentences:
            return MetricOutput(score=1.0, passed=True, explanation="No claims to verify")

        # Check how many response sentences are grounded in context
        grounded = 0
        for sentence in response_sentences:
            # Check word overlap as proxy for grounding
            sentence_words = set(self._tokenize(sentence))
            context_words = set(self._tokenize(input.context))
            if not sentence_words:
                grounded += 1
                continue
            overlap = len(sentence_words & context_words) / len(sentence_words)
            if overlap > 0.4:  # 40% word overlap = grounded
                grounded += 1

        score = grounded / len(response_sentences)
        return MetricOutput(
            score=round(score, 4),
            passed=score >= threshold,
            details={"grounded_claims": grounded, "total_claims": len(response_sentences)},
            explanation=f"{grounded}/{len(response_sentences)} claims grounded in context",
        )

    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.strip()) > 10]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w{3,}\b', text.lower())


class AnswerRelevancy(BaseMetric):
    """Measures if the answer addresses the original question."""

    name = "answer_relevancy"
    description = "Does the answer address the question?"
    category = "rag"

    def evaluate(self, input: MetricInput, threshold: float = 0.80) -> MetricOutput:
        if not input.response or not input.query:
            return MetricOutput(score=0.0, passed=False, explanation="Missing response or query")

        # Measure relevancy via keyword overlap between query and response
        query_words = set(self._tokenize(input.query))
        response_words = set(self._tokenize(input.response))

        if not query_words:
            return MetricOutput(score=1.0, passed=True, explanation="Empty query")

        # What fraction of query keywords appear in response
        overlap = len(query_words & response_words)
        relevancy = min(overlap / max(len(query_words) * 0.5, 1), 1.0)

        # Penalize very short or very long responses
        response_len = len(input.response.split())
        if response_len < 5:
            relevancy *= 0.5
        elif response_len > 500:
            relevancy *= 0.9

        score = round(relevancy, 4)
        return MetricOutput(
            score=score,
            passed=score >= threshold,
            details={"query_keywords": len(query_words), "overlap": overlap},
            explanation=f"Relevancy score: {score:.2f}",
        )

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w{3,}\b', text.lower())


class ContextPrecision(BaseMetric):
    """Measures if retrieved context is relevant to the query."""

    name = "context_precision"
    description = "Are the retrieved documents relevant?"
    category = "rag"

    def evaluate(self, input: MetricInput, threshold: float = 0.75) -> MetricOutput:
        if not input.context or not input.query:
            return MetricOutput(score=0.0, passed=False, explanation="Missing context or query")

        # Split context into chunks and check relevance of each
        chunks = self._split_chunks(input.context)
        if not chunks:
            return MetricOutput(score=0.0, passed=False, explanation="No context chunks")

        query_words = set(self._tokenize(input.query))
        relevant_chunks = 0

        for chunk in chunks:
            chunk_words = set(self._tokenize(chunk))
            if not chunk_words:
                continue
            overlap = len(query_words & chunk_words) / max(len(query_words), 1)
            if overlap > 0.2:
                relevant_chunks += 1

        score = round(relevant_chunks / len(chunks), 4)
        return MetricOutput(
            score=score,
            passed=score >= threshold,
            details={"relevant_chunks": relevant_chunks, "total_chunks": len(chunks)},
            explanation=f"{relevant_chunks}/{len(chunks)} context chunks are relevant",
        )

    def _split_chunks(self, text: str) -> List[str]:
        # Split by paragraphs or double newlines
        chunks = re.split(r'\n\n+', text)
        return [c.strip() for c in chunks if c.strip() and len(c.strip()) > 20]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w{3,}\b', text.lower())


class ContextRecall(BaseMetric):
    """Measures if all relevant information was retrieved."""

    name = "context_recall"
    description = "Were all relevant documents retrieved?"
    category = "rag"

    def evaluate(self, input: MetricInput, threshold: float = 0.75) -> MetricOutput:
        if not input.context or not input.reference:
            # Without reference, assume recall is adequate if context exists
            score = 0.8 if input.context else 0.0
            return MetricOutput(score=score, passed=score >= threshold, explanation="No reference for recall check")

        # Check how much of the reference answer is covered by context
        reference_words = set(self._tokenize(input.reference))
        context_words = set(self._tokenize(input.context))

        if not reference_words:
            return MetricOutput(score=1.0, passed=True, explanation="Empty reference")

        covered = len(reference_words & context_words)
        score = round(covered / len(reference_words), 4)

        return MetricOutput(
            score=score,
            passed=score >= threshold,
            details={"covered_terms": covered, "total_reference_terms": len(reference_words)},
            explanation=f"{covered}/{len(reference_words)} reference terms found in context",
        )

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w{3,}\b', text.lower())
