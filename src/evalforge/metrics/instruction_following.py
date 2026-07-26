"""Instruction following metric with rubric-based scoring.

Scores how well a model follows specific instructions in a prompt,
using configurable rubrics with weighted criteria. Handles format
instructions, length constraints, style requirements, and content rules.

Usage:
    from evalforge.metrics.instruction_following import InstructionFollowingMetric

    metric = InstructionFollowingMetric()
    result = metric.evaluate(
        instructions=["Respond in JSON format", "Be concise (under 100 words)"],
        response='{"answer": "This is concise."}',
    )
    print(f"Score: {result.score:.2f}")
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class InstructionType(str):
    """Instruction types for classification."""
    FORMAT = "format"
    LENGTH = "length"
    STYLE = "style"
    CONTENT = "content"
    STRUCTURE = "structure"
    LANGUAGE = "language"


@dataclass
class RubricCriterion:
    """A single criterion in a scoring rubric."""

    name: str
    instruction: str
    weight: float = 1.0
    instruction_type: str = "content"


@dataclass
class CriterionScore:
    """Score for a single rubric criterion."""

    criterion_name: str
    score: float  # 0.0 to 1.0
    followed: bool
    reason: str
    weight: float


@dataclass
class InstructionFollowingResult:
    """Complete instruction following evaluation result."""

    score: float  # Weighted average 0.0-1.0
    criterion_scores: List[CriterionScore]
    instructions_followed: int
    instructions_total: int
    passing: bool
    threshold: float
    details: str


class InstructionFollowingMetric:
    """Scores how well a model response follows given instructions.

    Evaluates each instruction against a configurable rubric and
    computes a weighted score based on compliance.

    Args:
        threshold: Minimum score to pass (default 0.75).
        default_weight: Default weight for unspecified criteria.
    """

    def __init__(self, threshold: float = 0.75, default_weight: float = 1.0):
        if not 0 < threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.0")
        self._threshold = threshold
        self._default_weight = default_weight

    @property
    def threshold(self) -> float:
        return self._threshold

    def evaluate(
        self,
        instructions: List[str],
        response: str,
        rubric: Optional[List[RubricCriterion]] = None,
    ) -> InstructionFollowingResult:
        """Evaluate how well a response follows the given instructions.

        Args:
            instructions: List of instruction strings from the prompt.
            response: The model's response text.
            rubric: Optional custom rubric. If None, auto-detects criteria.

        Returns:
            InstructionFollowingResult with per-criterion scores.
        """
        if not instructions:
            return self._empty_result()

        # Build rubric from instructions if not provided
        criteria = rubric or self._build_rubric(instructions)

        # Score each criterion
        criterion_scores: List[CriterionScore] = []
        for criterion in criteria:
            score = self._score_criterion(criterion, response)
            criterion_scores.append(score)

        # Compute weighted average
        total_weight = sum(c.weight for c in criteria)
        if total_weight == 0:
            weighted_score = 0.0
        else:
            weighted_score = sum(
                cs.score * cs.weight for cs in criterion_scores
            ) / total_weight

        followed = sum(1 for cs in criterion_scores if cs.followed)

        # Generate details
        details = self._format_details(criterion_scores, weighted_score)

        return InstructionFollowingResult(
            score=weighted_score,
            criterion_scores=criterion_scores,
            instructions_followed=followed,
            instructions_total=len(criteria),
            passing=weighted_score >= self._threshold,
            threshold=self._threshold,
            details=details,
        )

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> List[InstructionFollowingResult]:
        """Evaluate multiple test cases.

        Args:
            test_cases: List of dicts with 'instructions' and 'response' keys.

        Returns:
            List of results in same order as input.
        """
        return [
            self.evaluate(
                instructions=case.get("instructions", []),
                response=case.get("response", ""),
            )
            for case in test_cases
        ]

    def _build_rubric(self, instructions: List[str]) -> List[RubricCriterion]:
        """Auto-detect instruction types and build a rubric."""
        criteria: List[RubricCriterion] = []

        for instruction in instructions:
            inst_lower = instruction.lower()

            # Format instructions
            if any(kw in inst_lower for kw in ["json", "xml", "yaml", "markdown", "format"]):
                weight = 2.0  # Format requirements are higher priority
                inst_type = InstructionType.FORMAT
            # Length instructions
            elif any(kw in inst_lower for kw in ["word", "sentence", "character", "brief", "concise", "short", "long"]):
                weight = 1.5
                inst_type = InstructionType.LENGTH
            # Structure instructions
            elif any(kw in inst_lower for kw in ["bullet", "numbered", "list", "paragraph", "section", "heading"]):
                weight = 1.5
                inst_type = InstructionType.STRUCTURE
            # Language instructions
            elif any(kw in inst_lower for kw in ["language", "english", "french", "formal", "informal", "tone"]):
                weight = 1.5
                inst_type = InstructionType.LANGUAGE
            else:
                weight = self._default_weight
                inst_type = InstructionType.CONTENT

            criteria.append(RubricCriterion(
                name=instruction[:50],
                instruction=instruction,
                weight=weight,
                instruction_type=inst_type,
            ))

        return criteria

    def _score_criterion(self, criterion: RubricCriterion, response: str) -> CriterionScore:
        """Score a single criterion against the response."""
        inst = criterion.instruction.lower()
        inst_type = criterion.instruction_type

        # Dispatch to specialized checkers
        if inst_type == InstructionType.FORMAT or any(kw in inst for kw in ["json", "xml", "yaml"]):
            score, reason = self._check_format(inst, response)
        elif inst_type == InstructionType.LENGTH or any(kw in inst for kw in ["word", "sentence", "character", "concise", "brief"]):
            score, reason = self._check_length(inst, response)
        elif inst_type == InstructionType.STRUCTURE or any(kw in inst for kw in ["bullet", "numbered", "list"]):
            score, reason = self._check_structure(inst, response)
        elif inst_type == InstructionType.LANGUAGE or any(kw in inst for kw in ["language", "formal", "informal"]):
            score, reason = self._check_language(inst, response)
        else:
            score, reason = self._check_content(inst, response)

        return CriterionScore(
            criterion_name=criterion.name,
            score=score,
            followed=score >= 0.5,
            reason=reason,
            weight=criterion.weight,
        )

    def _check_format(self, instruction: str, response: str) -> tuple:
        """Check format-related instructions."""
        if "json" in instruction:
            return self._check_json_format(response)
        if "markdown" in instruction:
            has_md = bool(re.search(r'(#{1,6} |\*{1,2}|`{1,3}|\[.*\]\()', response))
            if has_md:
                return 1.0, "Response uses markdown formatting"
            return 0.0, "Response does not use markdown formatting"
        if "bullet" in instruction or "list" in instruction:
            has_bullets = bool(re.search(r'^\s*[-*•]\s', response, re.MULTILINE))
            if has_bullets:
                return 1.0, "Response contains bullet points"
            return 0.3, "Response does not use bullet points"
        return 0.7, "Format check: format partially matches"

    def _check_json_format(self, response: str) -> tuple:
        """Check if response is valid JSON."""
        # Try to extract JSON from response
        stripped = response.strip()
        json_patterns = [
            stripped,
            re.sub(r'^```json\s*', '', stripped),
            re.sub(r'```\s*$', '', stripped),
        ]
        for candidate in json_patterns:
            candidate = candidate.strip().rstrip('`').strip()
            try:
                json.loads(candidate)
                return 1.0, "Response is valid JSON"
            except (json.JSONDecodeError, ValueError):
                pass
        # Check if it at least starts with {
        if stripped.startswith('{') or stripped.startswith('['):
            return 0.5, "Response starts with JSON delimiter but is not valid JSON"
        return 0.0, "Response is not in JSON format"

    def _check_length(self, instruction: str, response: str) -> tuple:
        """Check length-related instructions."""
        word_count = len(response.split())
        char_count = len(response)

        # Extract numeric limits
        numbers = re.findall(r'\d+', instruction)

        if numbers:
            limit = int(numbers[0])
            if "word" in instruction:
                if word_count <= limit:
                    return 1.0, f"Response has {word_count} words (limit: {limit})"
                ratio = min(limit / word_count, 1.0)
                return ratio * 0.5, f"Response has {word_count} words, exceeds limit of {limit}"
            elif "character" in instruction or "char" in instruction:
                if char_count <= limit:
                    return 1.0, f"Response has {char_count} chars (limit: {limit})"
                ratio = min(limit / char_count, 1.0)
                return ratio * 0.5, f"Response has {char_count} chars, exceeds limit of {limit}"

        # Qualitative length checks
        if any(kw in instruction for kw in ["concise", "brief", "short"]):
            if word_count <= 100:
                return 1.0, f"Response is concise ({word_count} words)"
            elif word_count <= 200:
                return 0.7, f"Response is moderately concise ({word_count} words)"
            return 0.3, f"Response is too long ({word_count} words) for concise requirement"

        if "comprehensive" in instruction or "detailed" in instruction or "thorough" in instruction:
            if word_count >= 150:
                return 1.0, f"Response is comprehensive ({word_count} words)"
            return 0.4, f"Response may be too brief ({word_count} words) for detailed requirement"

        return 0.7, f"Length check: {word_count} words"

    def _check_structure(self, instruction: str, response: str) -> tuple:
        """Check structural instructions."""
        if "numbered" in instruction or "1." in instruction:
            has_numbered = bool(re.search(r'^\s*\d+[\.\)]\s', response, re.MULTILINE))
            if has_numbered:
                return 1.0, "Response uses numbered list"
            return 0.0, "Response does not use numbered list"

        if "bullet" in instruction or "list" in instruction:
            has_bullets = bool(re.search(r'^\s*[-*•]\s', response, re.MULTILINE))
            if has_bullets:
                return 1.0, "Response uses bullet points"
            return 0.2, "Response does not use bullet points"

        if "paragraph" in instruction:
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
            if len(paragraphs) >= 2:
                return 1.0, f"Response has {len(paragraphs)} paragraphs"
            return 0.5, "Response could use better paragraph structure"

        return 0.7, "Structure partially matches"

    def _check_language(self, instruction: str, response: str) -> tuple:
        """Check language/tone instructions."""
        if "formal" in instruction:
            informal_markers = re.findall(r"\b(gonna|wanna|kinda|sorta|ya|yep|nope)\b", response.lower())
            if not informal_markers:
                return 1.0, "Response uses formal language"
            return 0.4, f"Response contains informal language: {informal_markers[:3]}"

        if "informal" in instruction or "casual" in instruction:
            return 0.8, "Informality check passed (heuristic)"

        return 0.7, "Language check: partially satisfied"

    def _check_content(self, instruction: str, response: str) -> tuple:
        """Check content-based instructions using keyword matching."""
        # Extract key terms from instruction
        instruction_words = set(re.findall(r'\b\w{4,}\b', instruction.lower()))
        response_lower = response.lower()

        # Remove common stop words
        stop_words = {"that", "this", "with", "from", "have", "will", "your", "should", "must", "provide", "include"}
        instruction_words -= stop_words

        if not instruction_words:
            return 0.7, "Content instruction acknowledged"

        matches = sum(1 for word in instruction_words if word in response_lower)
        coverage = matches / len(instruction_words)

        if coverage >= 0.6:
            return 1.0, f"Response addresses instruction keywords ({matches}/{len(instruction_words)})"
        elif coverage >= 0.3:
            return 0.6, f"Response partially addresses instruction ({matches}/{len(instruction_words)} keywords)"
        return 0.2, f"Response may not follow instruction (only {matches}/{len(instruction_words)} keywords matched)"

    def _empty_result(self) -> InstructionFollowingResult:
        """Return an empty result when no instructions given."""
        return InstructionFollowingResult(
            score=1.0,
            criterion_scores=[],
            instructions_followed=0,
            instructions_total=0,
            passing=True,
            threshold=self._threshold,
            details="No instructions to evaluate.",
        )

    def _format_details(self, scores: List[CriterionScore], final: float) -> str:
        """Format a human-readable detail string."""
        lines = [f"Overall score: {final:.2f} (threshold: {self._threshold})"]
        for cs in scores:
            icon = "✅" if cs.followed else "❌"
            lines.append(f"  {icon} [{cs.score:.2f}] {cs.criterion_name}: {cs.reason}")
        return "\n".join(lines)
