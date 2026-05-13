"""Text quality metrics: ROUGE, BLEU, coherence, conciseness, fluency."""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Set

from evalforge.metrics.base import BaseMetric, MetricInput, MetricOutput


class RougeL(BaseMetric):
    """ROUGE-L: Longest Common Subsequence based metric."""

    name = "rouge_l"
    description = "Longest common subsequence overlap with reference"
    category = "text"

    def evaluate(self, input: MetricInput, threshold: float = 0.70) -> MetricOutput:
        if not input.response or not input.reference:
            return MetricOutput(score=0.0, passed=False, explanation="Missing response or reference")

        response_tokens = self._tokenize(input.response)
        reference_tokens = self._tokenize(input.reference)

        lcs_length = self._lcs_length(reference_tokens, response_tokens)

        precision = lcs_length / len(response_tokens) if response_tokens else 0
        recall = lcs_length / len(reference_tokens) if reference_tokens else 0

        if precision + recall == 0:
            score = 0.0
        else:
            score = 2 * precision * recall / (precision + recall)

        return MetricOutput(
            score=round(score, 4),
            passed=score >= threshold,
            details={"precision": round(precision, 4), "recall": round(recall, 4), "lcs_length": lcs_length},
        )

    def _lcs_length(self, x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        # Optimize for long sequences
        if m > 500 or n > 500:
            # Use word set overlap as approximation
            return len(set(x) & set(y))
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())


class Bleu(BaseMetric):
    """BLEU score: n-gram precision with brevity penalty."""

    name = "bleu"
    description = "N-gram precision with brevity penalty"
    category = "text"

    def evaluate(self, input: MetricInput, threshold: float = 0.60) -> MetricOutput:
        if not input.response or not input.reference:
            return MetricOutput(score=0.0, passed=False, explanation="Missing response or reference")

        response_tokens = self._tokenize(input.response)
        reference_tokens = self._tokenize(input.reference)

        # Calculate n-gram precisions (1-4)
        precisions = []
        for n in range(1, 5):
            p = self._ngram_precision(response_tokens, reference_tokens, n)
            precisions.append(p)

        # Geometric mean of precisions
        import math
        non_zero = [p for p in precisions if p > 0]
        if not non_zero:
            score = 0.0
        else:
            log_avg = sum(math.log(p) for p in non_zero) / len(non_zero)
            score = math.exp(log_avg)

        # Brevity penalty
        bp = min(1.0, len(response_tokens) / max(len(reference_tokens), 1))
        score *= bp

        return MetricOutput(
            score=round(score, 4),
            passed=score >= threshold,
            details={"precisions": [round(p, 4) for p in precisions], "brevity_penalty": round(bp, 4)},
        )

    def _ngram_precision(self, candidate: List[str], reference: List[str], n: int) -> float:
        cand_ngrams = Counter(tuple(candidate[i:i+n]) for i in range(len(candidate) - n + 1))
        ref_ngrams = Counter(tuple(reference[i:i+n]) for i in range(len(reference) - n + 1))
        clipped = sum(min(count, ref_ngrams[ng]) for ng, count in cand_ngrams.items())
        total = sum(cand_ngrams.values())
        return clipped / total if total > 0 else 0.0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())


class Coherence(BaseMetric):
    """Measures logical coherence and flow of text."""

    name = "coherence"
    description = "Logical flow and consistency of the response"
    category = "text"

    def evaluate(self, input: MetricInput, threshold: float = 0.80) -> MetricOutput:
        if not input.response:
            return MetricOutput(score=0.0, passed=False, explanation="Missing response")

        sentences = [s.strip() for s in re.split(r'[.!?]+', input.response) if s.strip()]
        if len(sentences) <= 1:
            return MetricOutput(score=0.9, passed=True, explanation="Single sentence - coherent by default")

        # Check sentence-to-sentence word overlap (proxy for coherence)
        coherence_scores = []
        for i in range(1, len(sentences)):
            prev_words = set(re.findall(r'\b\w{3,}\b', sentences[i-1].lower()))
            curr_words = set(re.findall(r'\b\w{3,}\b', sentences[i].lower()))
            if prev_words and curr_words:
                overlap = len(prev_words & curr_words) / min(len(prev_words), len(curr_words))
                coherence_scores.append(min(overlap * 3, 1.0))  # scale up
            else:
                coherence_scores.append(0.5)

        score = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5
        score = round(min(score, 1.0), 4)

        return MetricOutput(score=score, passed=score >= threshold, details={"sentences": len(sentences)})


class Conciseness(BaseMetric):
    """Measures if the response is appropriately concise."""

    name = "conciseness"
    description = "Is the response appropriately concise without losing information?"
    category = "text"

    def evaluate(self, input: MetricInput, threshold: float = 0.85) -> MetricOutput:
        if not input.response:
            return MetricOutput(score=0.0, passed=False, explanation="Missing response")

        word_count = len(input.response.split())

        # Penalize very long responses
        if word_count > 500:
            score = max(0.3, 1.0 - (word_count - 500) / 1000)
        elif word_count > 200:
            score = max(0.6, 1.0 - (word_count - 200) / 600)
        elif word_count < 5:
            score = 0.4  # too short
        else:
            score = 0.95  # good length

        # Check for repetition (reduces conciseness)
        words = input.response.lower().split()
        unique_ratio = len(set(words)) / len(words) if words else 1.0
        if unique_ratio < 0.5:
            score *= 0.7  # heavy repetition penalty

        score = round(score, 4)
        return MetricOutput(
            score=score, passed=score >= threshold,
            details={"word_count": word_count, "unique_ratio": round(unique_ratio, 4)},
        )


class Fluency(BaseMetric):
    """Measures grammatical fluency and readability."""

    name = "fluency"
    description = "Grammatical correctness and readability"
    category = "text"

    def evaluate(self, input: MetricInput, threshold: float = 0.85) -> MetricOutput:
        if not input.response:
            return MetricOutput(score=0.0, passed=False, explanation="Missing response")

        text = input.response
        score = 1.0

        # Check for basic fluency indicators
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        # Penalize very short sentences (fragments)
        short_sentences = sum(1 for s in sentences if len(s.split()) < 3)
        if sentences and short_sentences / len(sentences) > 0.5:
            score -= 0.2

        # Penalize excessive capitalization
        upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if upper_ratio > 0.3:
            score -= 0.15

        # Penalize lack of punctuation
        if len(text) > 50 and not any(p in text for p in '.!?'):
            score -= 0.2

        # Check for repeated words (stuttering)
        words = text.split()
        repeated = sum(1 for i in range(1, len(words)) if words[i] == words[i-1])
        if words and repeated / len(words) > 0.1:
            score -= 0.15

        score = round(max(score, 0.0), 4)
        return MetricOutput(score=score, passed=score >= threshold, details={"sentences": len(sentences)})
