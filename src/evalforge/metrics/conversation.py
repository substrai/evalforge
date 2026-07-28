"""Multi-turn conversation evaluation metrics.

Measures coherence across turns, context retention, and topic drift
detection in multi-turn LLM conversations.

Usage:
    from evalforge.metrics.conversation import ConversationMetric

    metric = ConversationMetric()
    result = metric.evaluate(conversation=[
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "What are its main features?"},
        {"role": "assistant", "content": "Python features include easy syntax."},
    ])
    print(f"Coherence: {result.coherence_score:.2f}")
    print(f"Context retention: {result.context_retention_score:.2f}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class TurnScore:
    """Evaluation scores for a single conversation turn."""

    turn_index: int
    role: str
    coherence: float
    context_retention: float
    topic_consistency: float
    response_quality: float


@dataclass
class ConversationEvalResult:
    """Complete evaluation result for a multi-turn conversation."""

    coherence_score: float        # Average cross-turn coherence
    context_retention_score: float  # How well context is maintained
    topic_drift_score: float      # 1.0 = no drift, 0.0 = heavy drift
    turn_scores: List[TurnScore]
    total_turns: int
    assistant_turns: int
    topic_changes_detected: int
    passing: bool
    threshold: float
    details: str

    @property
    def overall_score(self) -> float:
        """Weighted average of all metrics."""
        return (
            self.coherence_score * 0.4
            + self.context_retention_score * 0.35
            + self.topic_drift_score * 0.25
        )


class ConversationMetric:
    """Evaluates quality metrics across a multi-turn conversation.

    Measures:
    - Coherence: Do responses logically follow from prior messages?
    - Context retention: Does the model remember earlier facts?
    - Topic drift: Does the conversation stay on topic?

    Args:
        threshold: Minimum score to pass (default 0.70).
        min_turns: Minimum turns required for evaluation.
    """

    def __init__(self, threshold: float = 0.70, min_turns: int = 2):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.0")
        self._threshold = threshold
        self._min_turns = min_turns

    @property
    def threshold(self) -> float:
        return self._threshold

    def evaluate(
        self,
        conversation: List[Dict[str, Any]],
    ) -> ConversationEvalResult:
        """Evaluate a multi-turn conversation.

        Args:
            conversation: List of messages with 'role' and 'content' keys.

        Returns:
            ConversationEvalResult with all metric scores.
        """
        if len(conversation) < self._min_turns:
            return self._empty_result(len(conversation))

        assistant_turns = [m for m in conversation if m.get("role") == "assistant"]
        turn_scores: List[TurnScore] = []

        for i, msg in enumerate(conversation):
            if msg.get("role") != "assistant":
                continue

            prior = conversation[:i]
            content = msg.get("content", "")

            coherence = self._score_coherence(content, prior)
            context_ret = self._score_context_retention(content, prior)
            topic_cons = self._score_topic_consistency(content, prior)
            quality = self._score_response_quality(content)

            turn_scores.append(TurnScore(
                turn_index=i,
                role="assistant",
                coherence=coherence,
                context_retention=context_ret,
                topic_consistency=topic_cons,
                response_quality=quality,
            ))

        if not turn_scores:
            return self._empty_result(len(conversation))

        avg_coherence = sum(t.coherence for t in turn_scores) / len(turn_scores)
        avg_context = sum(t.context_retention for t in turn_scores) / len(turn_scores)
        avg_topic = sum(t.topic_consistency for t in turn_scores) / len(turn_scores)

        topic_changes = self._count_topic_changes(conversation)

        result = ConversationEvalResult(
            coherence_score=avg_coherence,
            context_retention_score=avg_context,
            topic_drift_score=avg_topic,
            turn_scores=turn_scores,
            total_turns=len(conversation),
            assistant_turns=len(assistant_turns),
            topic_changes_detected=topic_changes,
            passing=False,
            threshold=self._threshold,
            details="",
        )

        result.passing = result.overall_score >= self._threshold
        result.details = self._format_details(result)
        return result

    def _score_coherence(
        self,
        response: str,
        prior_messages: List[Dict[str, Any]],
    ) -> float:
        """Score how coherently this response follows the prior messages."""
        if not prior_messages:
            return 1.0

        last_user = next(
            (m["content"] for m in reversed(prior_messages)
             if m.get("role") == "user" and isinstance(m.get("content"), str)),
            ""
        )
        if not last_user:
            return 0.8

        # Extract keywords from user message
        user_keywords = self._extract_keywords(last_user)
        response_words = set(response.lower().split())

        if not user_keywords:
            return 0.8

        # Check how many user keywords appear in response
        overlap = len(user_keywords & response_words)
        coverage = overlap / len(user_keywords)

        # Check response length is reasonable
        word_count = len(response.split())
        length_score = 1.0 if 5 <= word_count <= 500 else 0.5

        # Coherence combines keyword overlap and length
        return min(1.0, coverage * 0.6 + length_score * 0.4)

    def _score_context_retention(
        self,
        response: str,
        prior_messages: List[Dict[str, Any]],
    ) -> float:
        """Score whether prior context is retained in the response."""
        if len(prior_messages) < 2:
            return 1.0

        # Get all earlier assistant responses
        earlier_responses = [
            m["content"] for m in prior_messages[:-1]
            if m.get("role") == "assistant" and isinstance(m.get("content"), str)
        ]

        if not earlier_responses:
            return 1.0

        # Check for contradiction signals
        contradiction_patterns = [
            (r'\b(yes|correct|right)\b', r'\b(no|incorrect|wrong)\b'),
            (r'\bwill\b', r'\bwill not\b'),
        ]

        response_lower = response.lower()
        contradiction_score = 1.0

        for positive, negative in contradiction_patterns:
            for earlier in earlier_responses:
                earlier_lower = earlier.lower()
                if re.search(positive, earlier_lower) and re.search(negative, response_lower):
                    contradiction_score -= 0.2

        # Check that named entities from earlier are not confused
        earlier_entities = set()
        for resp in earlier_responses:
            words = re.findall(r'\b[A-Z][a-z]+\b', resp)
            earlier_entities.update(words)

        # Basic retention score
        retention = max(0.0, contradiction_score)

        # Bonus: if earlier entities appear in response (shows memory)
        if earlier_entities:
            response_entities = set(re.findall(r'\b[A-Z][a-z]+\b', response))
            retention_bonus = len(earlier_entities & response_entities) / len(earlier_entities)
            retention = min(1.0, retention + retention_bonus * 0.2)

        return max(0.0, min(1.0, retention))

    def _score_topic_consistency(
        self,
        response: str,
        prior_messages: List[Dict[str, Any]],
    ) -> float:
        """Score whether the response stays on topic."""
        if not prior_messages:
            return 1.0

        # Build topic vocabulary from all prior messages
        all_prior_text = " ".join(
            m.get("content", "") for m in prior_messages
            if isinstance(m.get("content"), str)
        )

        prior_keywords = self._extract_keywords(all_prior_text)
        response_keywords = self._extract_keywords(response)

        if not prior_keywords or not response_keywords:
            return 0.8

        # Jaccard similarity between response and prior topics
        intersection = prior_keywords & response_keywords
        union = prior_keywords | response_keywords
        jaccard = len(intersection) / len(union) if union else 0.0

        # Scale up since conversations naturally broaden
        return min(1.0, jaccard * 3.0 + 0.3)

    def _score_response_quality(self, response: str) -> float:
        """Score basic response quality (length, completeness)."""
        if not response.strip():
            return 0.0

        word_count = len(response.split())
        sentence_count = len(re.split(r'[.!?]+', response))

        # Length score
        if word_count < 3:
            length_score = 0.2
        elif word_count < 10:
            length_score = 0.6
        elif word_count < 500:
            length_score = 1.0
        else:
            length_score = 0.7  # Very long responses are penalized slightly

        # Sentence structure score
        sentence_score = min(1.0, sentence_count * 0.3)

        return min(1.0, length_score * 0.7 + sentence_score * 0.3)

    def _count_topic_changes(self, conversation: List[Dict[str, Any]]) -> int:
        """Count the number of topic shifts in the conversation."""
        user_messages = [
            m.get("content", "") for m in conversation
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        ]

        if len(user_messages) < 2:
            return 0

        changes = 0
        for i in range(1, len(user_messages)):
            kw_prev = self._extract_keywords(user_messages[i-1])
            kw_curr = self._extract_keywords(user_messages[i])
            if kw_prev and kw_curr:
                overlap = len(kw_prev & kw_curr) / max(len(kw_prev), len(kw_curr))
                if overlap < 0.2:
                    changes += 1

        return changes

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "this",
            "that", "these", "those", "i", "you", "he", "she", "it",
            "we", "they", "what", "which", "who", "how", "when", "where",
            "why", "and", "or", "but", "if", "then", "so", "yet", "for",
            "of", "in", "on", "at", "to", "from", "with", "about", "by",
        }
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return {w for w in words if w not in stop_words}

    def _empty_result(self, turn_count: int) -> ConversationEvalResult:
        return ConversationEvalResult(
            coherence_score=1.0,
            context_retention_score=1.0,
            topic_drift_score=1.0,
            turn_scores=[],
            total_turns=turn_count,
            assistant_turns=0,
            topic_changes_detected=0,
            passing=True,
            threshold=self._threshold,
            details=f"Conversation too short ({turn_count} turns) for evaluation.",
        )

    def _format_details(self, result: ConversationEvalResult) -> str:
        lines = [
            f"Overall: {result.overall_score:.2f} ({'PASS' if result.passing else 'FAIL'})",
            f"  Coherence:        {result.coherence_score:.2f}",
            f"  Context Retention: {result.context_retention_score:.2f}",
            f"  Topic Consistency: {result.topic_drift_score:.2f}",
            f"  Turns: {result.total_turns} total, {result.assistant_turns} assistant",
            f"  Topic changes: {result.topic_changes_detected}",
        ]
        return "\n".join(lines)
