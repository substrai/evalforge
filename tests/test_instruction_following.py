"""Tests for instruction following metric with rubric-based scoring."""

from __future__ import annotations

import pytest

from evalforge.metrics.instruction_following import (
    CriterionScore,
    InstructionFollowingMetric,
    InstructionFollowingResult,
    RubricCriterion,
)


class TestMetricInit:
    def test_default_threshold(self):
        metric = InstructionFollowingMetric()
        assert metric.threshold == 0.75

    def test_custom_threshold(self):
        metric = InstructionFollowingMetric(threshold=0.9)
        assert metric.threshold == 0.9

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            InstructionFollowingMetric(threshold=0.0)
        with pytest.raises(ValueError):
            InstructionFollowingMetric(threshold=1.5)


class TestJSONFormatInstruction:
    def test_valid_json_response(self):
        metric = InstructionFollowingMetric()
        result = metric.evaluate(
            instructions=["Respond in JSON format"],
            response='{"answer": "yes", "confidence": 0.9}',
        )
        # JSON instruction should score well
        assert result.score >= 0.8

    def test_invalid_json_response(self):
        metric = InstructionFollowingMetric()
        result = metric.evaluate(
            instructions=["Respond in JSON format"],
            response="This is a plain text response without JSON.",
        )
        assert result.score < 0.5

    def test_json_in_code_block(self):
        metric = InstructionFollowingMetric()
        result = metric.evaluate(
            instructions=["Return JSON format"],
            response='```json\n{"key": "value"}\n```',
        )
        assert result.score >= 0.5


class TestLengthInstruction:
    def test_concise_short_response(self):
        metric = InstructionFollowingMetric()
        result = metric.evaluate(
            instructions=["Be concise"],
            response="Short answer here.",
        )
        assert result.score >= 0.8

    def test_concise_long_response_fails(self):
        metric = InstructionFollowingMetric()
        result = metric.evaluate(
            instructions=["Be concise and brief"],
            response=" ".join(["word"] * 300),  # 300 words
        )
        assert result.score < 0.8

    def test_under_100_words_limit(self):
        metric = InstructionFollowingMetric()
        short = " ".join(["word"] * 50)
        result = metric.evaluate(
            instructions=["Respond in under 100 words"],
            response=short,
        )
        assert result.score >= 0.9

    def test_exceeds_word_limit(self):
        metric = InstructionFollowingMetric()
        long_text = " ".join(["word"] * 200)
        result = metric.evaluate(
            instructions=["Respond in under 50 words"],
            response=long_text,
        )
        assert result.score < 0.7


class TestStructureInstruction:
    def test_bullet_points_present(self):
        metric = InstructionFollowingMetric()
        result = metric.evaluate(
            instructions=["Use bullet points"],
            response="- Item one\n- Item two\n- Item three",
        )
        assert result.score >= 0.9

    def test_bullet_points_absent(self):
        metric = InstructionFollowingMetric()
        result = metric.evaluate(
            instructions=["Use bullet points"],
            response="Just a plain paragraph with no bullets.",
        )
        assert result.score < 0.5

    def test_numbered_list(self):
        metric = InstructionFollowingMetric()
        result = metric.evaluate(
            instructions=["Use a numbered list"],
            response="1. First item\n2. Second item\n3. Third item",
        )
        assert result.score >= 0.9

    def test_numbered_absent_fails(self):
        metric = InstructionFollowingMetric()
        result = metric.evaluate(
            instructions=["Use a numbered list"],
            response="Just some text with no numbers.",
        )
        assert result.score < 0.5


class TestFormalLanguageInstruction:
    def test_formal_response_passes(self):
        metric = InstructionFollowingMetric()
        result = metric.evaluate(
            instructions=["Use formal language"],
            response="The analysis indicates a significant improvement in performance metrics.",
        )
        assert result.score >= 0.7

    def test_informal_response_fails(self):
        metric = InstructionFollowingMetric()
        result = metric.evaluate(
            instructions=["Use formal language"],
            response="Yeah, gonna say it's kinda better sorta.",
        )
        assert result.score < 0.7


class TestMultipleInstructions:
    def test_all_instructions_followed(self):
        metric = InstructionFollowingMetric()
        result = metric.evaluate(
            instructions=[
                "Respond in JSON format",
                "Be concise",
            ],
            response='{"answer": "yes"}',
        )
        assert result.instructions_total == 2
        assert result.score > 0.7

    def test_some_instructions_followed(self):
        metric = InstructionFollowingMetric()
        result = metric.evaluate(
            instructions=[
                "Respond in JSON format",
                "Use bullet points",
            ],
            response='{"answer": "yes"}',
        )
        # JSON followed, bullets not — partial score
        assert 0.0 < result.score < 1.0

    def test_no_instructions(self):
        metric = InstructionFollowingMetric()
        result = metric.evaluate(instructions=[], response="any response")
        assert result.score == 1.0
        assert result.passing is True


class TestCustomRubric:
    def test_custom_rubric_applied(self):
        metric = InstructionFollowingMetric(threshold=0.5)
        rubric = [
            RubricCriterion(
                name="JSON format",
                instruction="Respond in JSON format",
                weight=2.0,
                instruction_type="format",
            ),
            RubricCriterion(
                name="Concise",
                instruction="Be concise",
                weight=1.0,
                instruction_type="length",
            ),
        ]
        result = metric.evaluate(
            instructions=["Respond in JSON format", "Be concise"],
            response='{"answer": "yes"}',
            rubric=rubric,
        )
        # Custom weights should be respected
        assert result.criterion_scores[0].weight == 2.0
        assert result.criterion_scores[1].weight == 1.0

    def test_higher_weight_criterion_dominates(self):
        metric = InstructionFollowingMetric(threshold=0.5)
        rubric = [
            RubricCriterion(
                name="JSON",
                instruction="Respond in JSON format",
                weight=10.0,
                instruction_type="format",
            ),
            RubricCriterion(
                name="Bullets",
                instruction="Use bullet points",
                weight=0.1,
                instruction_type="structure",
            ),
        ]
        result = metric.evaluate(
            instructions=["Respond in JSON format", "Use bullet points"],
            response='{"key": "value"}',
            rubric=rubric,
        )
        # JSON passes (weight 10), bullets fail (weight 0.1) → should pass overall
        assert result.passing is True


class TestBatchEvaluation:
    def test_batch_returns_all_results(self):
        metric = InstructionFollowingMetric()
        cases = [
            {"instructions": ["Be concise"], "response": "Short."},
            {"instructions": ["Use JSON format"], "response": '{"ok": true}'},
            {"instructions": ["Use bullet points"], "response": "No bullets here."},
        ]
        results = metric.evaluate_batch(cases)
        assert len(results) == 3

    def test_batch_results_independent(self):
        metric = InstructionFollowingMetric()
        cases = [
            {"instructions": ["Be concise"], "response": "Short."},
            {"instructions": ["Respond in JSON format"], "response": "not json"},
        ]
        results = metric.evaluate_batch(cases)
        assert results[0].score != results[1].score


class TestResultDetails:
    def test_details_string_present(self):
        metric = InstructionFollowingMetric()
        result = metric.evaluate(
            instructions=["Be concise"],
            response="Short answer.",
        )
        assert len(result.details) > 0
        assert "score" in result.details.lower() or "0." in result.details

    def test_passing_threshold(self):
        metric = InstructionFollowingMetric(threshold=0.5)
        result = metric.evaluate(
            instructions=["Respond in JSON format"],
            response='{"ok": true}',
        )
        assert result.passing is True
        assert result.threshold == 0.5
