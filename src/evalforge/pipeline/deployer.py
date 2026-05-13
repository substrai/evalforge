"""Pipeline deployer - generates infrastructure for evaluation pipelines.

Generates Step Functions state machine definitions, Lambda configs,
EventBridge schedules, and CloudWatch dashboards.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evalforge.core.config import EvalConfig


@dataclass
class LambdaConfig:
    """Configuration for a Lambda function in the pipeline."""

    function_name: str
    handler: str
    runtime: str = "python3.12"
    memory_mb: int = 512
    timeout_seconds: int = 300
    environment: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class DeploymentManifest:
    """Complete deployment manifest for the evaluation pipeline."""

    project_name: str
    region: str
    environment: str
    lambdas: List[LambdaConfig] = field(default_factory=list)
    state_machine_definition: Dict[str, Any] = field(default_factory=dict)
    schedule_expression: str = ""
    dashboard_widgets: List[Dict[str, Any]] = field(default_factory=list)
    iam_policies: List[Dict[str, Any]] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    def to_sam_template(self) -> Dict[str, Any]:
        """Generate SAM template."""
        resources = {}

        # Lambda functions
        for lam in self.lambdas:
            logical_id = lam.function_name.replace("-", "").title() + "Function"
            resources[logical_id] = {
                "Type": "AWS::Serverless::Function",
                "Properties": {
                    "FunctionName": f"{self.project_name}-{lam.function_name}",
                    "Handler": lam.handler,
                    "Runtime": lam.runtime,
                    "MemorySize": lam.memory_mb,
                    "Timeout": lam.timeout_seconds,
                    "Environment": {"Variables": lam.environment},
                    "Description": lam.description,
                },
            }

        # State Machine
        if self.state_machine_definition:
            resources["EvalStateMachine"] = {
                "Type": "AWS::Serverless::StateMachine",
                "Properties": {
                    "Name": f"{self.project_name}-eval-pipeline",
                    "Definition": self.state_machine_definition,
                },
            }

        # Schedule
        if self.schedule_expression:
            resources["EvalSchedule"] = {
                "Type": "AWS::Events::Rule",
                "Properties": {
                    "Name": f"{self.project_name}-eval-schedule",
                    "ScheduleExpression": self.schedule_expression,
                    "State": "ENABLED",
                },
            }

        return {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Transform": "AWS::Serverless-2016-10-31",
            "Description": f"EvalForge pipeline: {self.project_name}",
            "Resources": resources,
        }

    def to_json(self) -> str:
        """Serialize manifest to JSON."""
        return json.dumps(self.to_sam_template(), indent=2)

    def summary(self) -> str:
        lines = [
            f"Deployment Manifest: {self.project_name}",
            f"  Region: {self.region}",
            f"  Environment: {self.environment}",
            f"  Lambda functions: {len(self.lambdas)}",
            f"  State machine: {'Yes' if self.state_machine_definition else 'No'}",
            f"  Schedule: {self.schedule_expression or 'None'}",
        ]
        for lam in self.lambdas:
            lines.append(f"    • {lam.function_name} ({lam.memory_mb}MB, {lam.timeout_seconds}s)")
        return "\n".join(lines)


class PipelineDeployer:
    """Generates deployment infrastructure for evaluation pipelines.

    Usage:
        deployer = PipelineDeployer(config)
        manifest = deployer.generate_manifest(environment="prod")
        print(manifest.to_json())  # SAM template
    """

    def __init__(self, config: EvalConfig):
        self.config = config

    def generate_manifest(self, environment: str = "prod") -> DeploymentManifest:
        """Generate complete deployment manifest.

        Args:
            environment: Target environment (dev, staging, prod)

        Returns:
            DeploymentManifest with all infrastructure definitions
        """
        lambdas = self._generate_lambdas(environment)
        state_machine = self._generate_state_machine(lambdas)
        schedule = self._generate_schedule()

        return DeploymentManifest(
            project_name=self.config.project_name,
            region=self.config.model.region,
            environment=environment,
            lambdas=lambdas,
            state_machine_definition=state_machine,
            schedule_expression=schedule,
        )

    def _generate_lambdas(self, environment: str) -> List[LambdaConfig]:
        """Generate Lambda function configs for each pipeline step."""
        base_env = {
            "PROJECT_NAME": self.config.project_name,
            "ENVIRONMENT": environment,
            "USE_CASE_TYPE": self.config.use_case_type.value,
            "MODEL_ID": self.config.model.model_id,
        }

        lambdas = [
            LambdaConfig(
                function_name="data-loader",
                handler="evalforge.pipeline.handlers.load_data",
                memory_mb=256,
                timeout_seconds=60,
                environment=base_env,
                description="Loads test data for evaluation",
            ),
            LambdaConfig(
                function_name="metric-evaluator",
                handler="evalforge.pipeline.handlers.evaluate_metrics",
                memory_mb=512,
                timeout_seconds=300,
                environment={**base_env, "METRICS": ",".join(self.config.metrics)},
                description="Runs metric evaluations on test samples",
            ),
            LambdaConfig(
                function_name="drift-checker",
                handler="evalforge.pipeline.handlers.check_drift",
                memory_mb=256,
                timeout_seconds=60,
                environment=base_env,
                description="Checks for quality drift against baseline",
            ),
            LambdaConfig(
                function_name="reporter",
                handler="evalforge.pipeline.handlers.generate_report",
                memory_mb=256,
                timeout_seconds=60,
                environment=base_env,
                description="Generates evaluation report and sends alerts",
            ),
        ]

        return lambdas

    def _generate_state_machine(self, lambdas: List[LambdaConfig]) -> Dict[str, Any]:
        """Generate Step Functions state machine definition."""
        return {
            "Comment": f"EvalForge pipeline: {self.config.project_name}",
            "StartAt": "LoadData",
            "States": {
                "LoadData": {
                    "Type": "Task",
                    "Resource": f"arn:aws:lambda:${{Region}}:${{AccountId}}:function:{self.config.project_name}-data-loader",
                    "Next": "EvaluateMetrics",
                    "Retry": [{"ErrorEquals": ["States.ALL"], "MaxAttempts": 2}],
                },
                "EvaluateMetrics": {
                    "Type": "Task",
                    "Resource": f"arn:aws:lambda:${{Region}}:${{AccountId}}:function:{self.config.project_name}-metric-evaluator",
                    "Next": "CheckDrift",
                    "TimeoutSeconds": 300,
                },
                "CheckDrift": {
                    "Type": "Task",
                    "Resource": f"arn:aws:lambda:${{Region}}:${{AccountId}}:function:{self.config.project_name}-drift-checker",
                    "Next": "DriftDecision",
                },
                "DriftDecision": {
                    "Type": "Choice",
                    "Choices": [
                        {
                            "Variable": "$.drift_detected",
                            "BooleanEquals": True,
                            "Next": "AlertAndReport",
                        }
                    ],
                    "Default": "GenerateReport",
                },
                "AlertAndReport": {
                    "Type": "Parallel",
                    "Branches": [
                        {"StartAt": "SendAlert", "States": {"SendAlert": {"Type": "Task", "Resource": "arn:aws:sns:${Region}:${AccountId}:eval-alerts", "End": True}}},
                        {"StartAt": "Report", "States": {"Report": {"Type": "Task", "Resource": f"arn:aws:lambda:${{Region}}:${{AccountId}}:function:{self.config.project_name}-reporter", "End": True}}},
                    ],
                    "End": True,
                },
                "GenerateReport": {
                    "Type": "Task",
                    "Resource": f"arn:aws:lambda:${{Region}}:${{AccountId}}:function:{self.config.project_name}-reporter",
                    "End": True,
                },
            },
        }

    def _generate_schedule(self) -> str:
        """Generate EventBridge schedule expression."""
        freq = self.config.schedule.frequency
        schedule_map = {
            "hourly": "rate(1 hour)",
            "daily": f"cron(0 {self.config.schedule.time.split(':')[0]} * * ? *)",
            "weekly": f"cron(0 {self.config.schedule.time.split(':')[0]} ? * MON *)",
            "on_demand": "",
        }
        return schedule_map.get(freq, "")
