"""Use case configuration parser.

Reads evalforge.yaml and determines which metrics, thresholds,
and data generators to use based on the declared use case type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class UseCaseType(Enum):
    """Supported LLM use case types."""

    RAG = "rag"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    CHAT = "chat"
    CODE = "code"

    @classmethod
    def from_string(cls, value: str) -> "UseCaseType":
        try:
            return cls(value.lower())
        except ValueError:
            valid = [t.value for t in cls]
            raise ValueError(f"Unknown use case type: '{value}'. Must be one of: {valid}")


# Default metrics per use case type
DEFAULT_METRICS: Dict[str, List[str]] = {
    "rag": ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "toxicity", "latency"],
    "summarization": ["rouge_l", "bleu", "coherence", "conciseness", "factual_consistency"],
    "classification": ["accuracy", "precision", "recall", "f1_score", "confidence_calibration"],
    "generation": ["fluency", "coherence", "relevance", "creativity", "safety"],
    "chat": ["helpfulness", "harmlessness", "honesty", "engagement", "context_retention"],
    "code": ["correctness", "efficiency", "readability", "security", "test_pass_rate"],
}

# Default thresholds per use case type
DEFAULT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "rag": {"faithfulness": 0.85, "answer_relevancy": 0.80, "context_precision": 0.75, "context_recall": 0.75, "toxicity": 0.05, "latency": 3000},
    "summarization": {"rouge_l": 0.70, "bleu": 0.60, "coherence": 0.80, "conciseness": 0.85, "factual_consistency": 0.90},
    "classification": {"accuracy": 0.90, "precision": 0.85, "recall": 0.85, "f1_score": 0.85, "confidence_calibration": 0.80},
    "generation": {"fluency": 0.85, "coherence": 0.80, "relevance": 0.80, "creativity": 0.70, "safety": 0.95},
    "chat": {"helpfulness": 0.85, "harmlessness": 0.95, "honesty": 0.90, "engagement": 0.75, "context_retention": 0.80},
    "code": {"correctness": 0.90, "efficiency": 0.75, "readability": 0.80, "security": 0.95, "test_pass_rate": 0.85},
}


@dataclass
class ModelConfig:
    """Model configuration for evaluation."""

    provider: str = "bedrock"
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    region: str = "us-east-1"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(
            provider=data.get("provider", "bedrock"),
            model_id=data.get("model_id", "anthropic.claude-3-haiku-20240307-v1:0"),
            region=data.get("region", "us-east-1"),
        )


@dataclass
class TestDataConfig:
    """Test data configuration."""

    source: str = "synthetic"  # synthetic | file | api
    count: int = 100
    generator_model: str = "bedrock/claude-3-sonnet"
    categories: List[str] = field(default_factory=lambda: ["simple", "complex", "adversarial", "edge_cases"])
    golden_dataset: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestDataConfig":
        return cls(
            source=data.get("source", "synthetic"),
            count=data.get("count", 100),
            generator_model=data.get("generator_model", "bedrock/claude-3-sonnet"),
            categories=data.get("categories", ["simple", "complex", "adversarial", "edge_cases"]),
            golden_dataset=data.get("golden_dataset"),
        )


@dataclass
class ScheduleConfig:
    """Evaluation schedule configuration."""

    frequency: str = "daily"  # hourly | daily | weekly | on_demand
    time: str = "02:00"
    environments: List[str] = field(default_factory=lambda: ["prod"])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduleConfig":
        return cls(
            frequency=data.get("frequency", "daily"),
            time=data.get("time", "02:00"),
            environments=data.get("environments", ["prod"]),
        )


@dataclass
class CustomMetric:
    """Custom metric definition."""

    name: str
    type: str = "llm_judge"  # llm_judge | function | regex | threshold
    prompt: str = ""
    threshold: float = 0.8
    description: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomMetric":
        return cls(
            name=data.get("name", ""),
            type=data.get("type", "llm_judge"),
            prompt=data.get("prompt", ""),
            threshold=data.get("threshold", 0.8),
            description=data.get("description", ""),
        )


@dataclass
class EvalConfig:
    """Complete evaluation configuration.

    Parsed from evalforge.yaml, this determines the entire
    evaluation pipeline: metrics, thresholds, data, schedule.
    """

    project_name: str
    version: str
    use_case_type: UseCaseType
    use_case_description: str
    model: ModelConfig
    metrics: List[str]
    thresholds: Dict[str, float]
    custom_metrics: List[CustomMetric] = field(default_factory=list)
    test_data: TestDataConfig = field(default_factory=TestDataConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "EvalConfig":
        """Parse config from YAML content."""
        data = yaml.safe_load(yaml_content)
        if not data:
            raise ValueError("Empty configuration")

        project = data.get("project", {})
        use_case = data.get("use_case", {})
        evaluation = data.get("evaluation", {})

        # Determine use case type
        uc_type = UseCaseType.from_string(use_case.get("type", "rag"))

        # Determine metrics (auto or explicit)
        metrics_config = evaluation.get("metrics", "auto")
        if metrics_config == "auto" or metrics_config is None:
            metrics = DEFAULT_METRICS[uc_type.value]
        elif isinstance(metrics_config, list):
            metrics = metrics_config
        else:
            metrics = DEFAULT_METRICS[uc_type.value]

        # Determine thresholds
        default_thresh = DEFAULT_THRESHOLDS[uc_type.value]
        custom_thresh = evaluation.get("thresholds", {})
        thresholds = {**default_thresh, **custom_thresh}

        # Custom metrics
        custom_metrics = [
            CustomMetric.from_dict(cm) for cm in evaluation.get("custom_metrics", [])
        ]

        # Model config
        model = ModelConfig.from_dict(use_case.get("model", {}))

        # Test data
        test_data = TestDataConfig.from_dict(data.get("test_data", {}))

        # Schedule
        schedule = ScheduleConfig.from_dict(data.get("schedule", {}))

        return cls(
            project_name=project.get("name", "evaluation"),
            version=project.get("version", "1.0.0"),
            use_case_type=uc_type,
            use_case_description=use_case.get("description", ""),
            model=model,
            metrics=metrics,
            thresholds=thresholds,
            custom_metrics=custom_metrics,
            test_data=test_data,
            schedule=schedule,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "EvalConfig":
        """Load config from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return cls.from_yaml(path.read_text())

    @classmethod
    def for_use_case(cls, use_case_type: str, project_name: str = "evaluation") -> "EvalConfig":
        """Create a default config for a use case type (quick start)."""
        uc_type = UseCaseType.from_string(use_case_type)
        return cls(
            project_name=project_name,
            version="1.0.0",
            use_case_type=uc_type,
            use_case_description=f"Default {uc_type.value} evaluation",
            model=ModelConfig(),
            metrics=DEFAULT_METRICS[uc_type.value],
            thresholds=DEFAULT_THRESHOLDS[uc_type.value],
        )

    def get_threshold(self, metric: str) -> float:
        """Get threshold for a metric."""
        return self.thresholds.get(metric, 0.8)

    def summary(self) -> str:
        """Human-readable config summary."""
        lines = [
            f"EvalForge Config: {self.project_name} v{self.version}",
            f"  Use case: {self.use_case_type.value}",
            f"  Description: {self.use_case_description}",
            f"  Model: {self.model.provider}/{self.model.model_id}",
            f"  Metrics ({len(self.metrics)}):",
        ]
        for m in self.metrics:
            thresh = self.thresholds.get(m, "N/A")
            lines.append(f"    - {m} (threshold: {thresh})")
        if self.custom_metrics:
            lines.append(f"  Custom metrics ({len(self.custom_metrics)}):")
            for cm in self.custom_metrics:
                lines.append(f"    - {cm.name} ({cm.type})")
        lines.append(f"  Test data: {self.test_data.source} ({self.test_data.count} samples)")
        lines.append(f"  Schedule: {self.schedule.frequency} at {self.schedule.time}")
        return "\n".join(lines)
