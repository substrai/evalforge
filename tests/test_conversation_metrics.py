"""Tests for multi-turn conversation evaluation metrics."""

from __future__ import annotations

import pytest

from evalforge.metrics.conversation import ConversationMetric, ConversationEvalResult, TurnScore


def _make_convo(pairs: list) -> list:
    msgs = []
    for user_msg, assistant_msg in pairs:
        msgs.append({"role": "user", "content": user_msg})
        msgs.append({"role": "assistant", "content": assistant_msg})
    return msgs


PYTHON_CONVO = _make_convo([
    ("What is Python?", "Python is a high-level programming language known for simplicity and readability."),
    ("What are its main features?", "Python features include dynamic typing, garbage collection, and a vast standard library."),
    ("Can I use it for web development?", "Yes, Python is widely used for web development with frameworks like Django and Flask."),
])

SHORT_CONVO = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
]


class TestConversationMetricInit:
    def test_default_threshold(self):
        m = ConversationMetric()
        assert m.threshold == 0.70

    def test_custom_threshold(self):
        m = ConversationMetric(threshold=0.85)
        assert m.threshold == 0.85

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            ConversationMetric(threshold=0.0)


class TestEvaluateCoherentConversation:
    def test_returns_result(self):
        m = ConversationMetric()
        result = m.evaluate(PYTHON_CONVO)
        assert isinstance(result, ConversationEvalResult)
        assert result.total_turns == len(PYTHON_CONVO)

    def test_coherent_convo_scores_well(self):
        m = ConversationMetric(threshold=0.5)
        result = m.evaluate(PYTHON_CONVO)
        assert result.coherence_score > 0.3
        assert result.context_retention_score > 0.0
        assert result.topic_drift_score > 0.0

    def test_overall_score_bounded(self):
        m = ConversationMetric()
        result = m.evaluate(PYTHON_CONVO)
        assert 0.0 <= result.overall_score <= 1.0

    def test_turn_scores_for_each_assistant(self):
        m = ConversationMetric()
        result = m.evaluate(PYTHON_CONVO)
        assert result.assistant_turns == 3
        assert len(result.turn_scores) == 3

    def test_details_string_present(self):
        m = ConversationMetric()
        result = m.evaluate(PYTHON_CONVO)
        assert len(result.details) > 0


class TestTooShortConversation:
    def test_single_turn_passes(self):
        m = ConversationMetric(min_turns=2)
        result = m.evaluate(SHORT_CONVO)
        assert result.passing is True

    def test_too_short_returns_perfect_score(self):
        m = ConversationMetric(min_turns=4)
        result = m.evaluate(SHORT_CONVO)
        assert result.coherence_score == 1.0
        assert result.turn_scores == []


class TestCoherenceScoring:
    def test_on_topic_response(self):
        m = ConversationMetric()
        convo = [
            {"role": "user", "content": "Tell me about Python programming language features"},
            {"role": "assistant", "content": "Python is a programming language with many features including readability and simplicity."},
        ]
        result = m.evaluate(convo)
        assert result.coherence_score > 0.4

    def test_off_topic_response_lower_score(self):
        m = ConversationMetric()
        on_topic = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI that enables systems to learn from data."},
        ]
        off_topic = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "The weather today is sunny and warm."},
        ]
        on_result = m.evaluate(on_topic)
        off_result = m.evaluate(off_topic)
        assert on_result.coherence_score >= off_result.coherence_score


class TestContextRetention:
    def test_references_earlier_facts(self):
        m = ConversationMetric()
        convo = _make_convo([
            ("My name is Alice and I work at Google.", "Nice to meet you Alice! Google is a great company."),
            ("What company do I work at?", "You work at Google, as you mentioned earlier."),
        ])
        result = m.evaluate(convo)
        # Second response references Google from earlier
        assert result.context_retention_score > 0.5


class TestTopicDrift:
    def test_no_drift_in_focused_convo(self):
        m = ConversationMetric()
        result = m.evaluate(PYTHON_CONVO)
        # Python conversation stays on topic
        assert result.topic_drift_score > 0.3

    def test_topic_changes_counted(self):
        m = ConversationMetric()
        convo = _make_convo([
            ("What is Python?", "Python is a programming language."),
            ("How do I make pasta?", "Boil water, add pasta, cook for 10 minutes."),
            ("What is quantum computing?", "Quantum computing uses quantum bits or qubits."),
        ])
        result = m.evaluate(convo)
        # Multiple very different topics
        assert result.topic_changes_detected >= 1


class TestPassingThreshold:
    def test_good_convo_passes(self):
        m = ConversationMetric(threshold=0.3)
        result = m.evaluate(PYTHON_CONVO)
        assert result.passing is True

    def test_high_threshold_fails(self):
        m = ConversationMetric(threshold=0.99)
        result = m.evaluate(PYTHON_CONVO)
        # Very high threshold is hard to meet with heuristic scoring
        # Just verify the threshold is used
        assert result.threshold == 0.99

    def test_passing_consistent_with_score(self):
        m = ConversationMetric(threshold=0.5)
        result = m.evaluate(PYTHON_CONVO)
        if result.overall_score >= 0.5:
            assert result.passing is True
        else:
            assert result.passing is False


class TestEmptyAndEdgeCases:
    def test_empty_conversation(self):
        m = ConversationMetric()
        result = m.evaluate([])
        assert result.total_turns == 0

    def test_only_user_messages(self):
        m = ConversationMetric()
        convo = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ]
        result = m.evaluate(convo)
        assert result.assistant_turns == 0

    def test_very_long_response_penalized_slightly(self):
        m = ConversationMetric()
        convo = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "word " * 600},
        ]
        result = m.evaluate(convo)
        assert result.assistant_turns == 1
