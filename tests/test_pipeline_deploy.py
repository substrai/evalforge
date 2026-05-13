"""Tests for deployment, CI/CD, and reporting."""

import pytest
from evalforge.core.config import EvalConfig
from evalforge.core.result import EvalResult, MetricScore
from evalforge.pipeline.deployer import PipelineDeployer, DeploymentManifest
from evalforge.pipeline.cicd import CICDIntegration, GateDecision
from evalforge.pipeline.reports import ReportGenerator, ComplianceReport


SAMPLE_CONFIG = """
project:
  name: "test-eval"
  version: "1.0.0"
use_case:
  type: rag
  description: "Test"
  model:
    provider: bedrock
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    region: us-east-1
evaluation:
  metrics: auto
schedule:
  frequency: daily
  time: "02:00"
"""


def _make_result(pass_rate: float = 0.90, scores: list = None) -> EvalResult:
    if scores is None:
        scores = [
            MetricScore(name="faithfulness", score=pass_rate, threshold=0.85, passed=pass_rate >= 0.85),
            MetricScore(name="toxicity", score=0.02, threshold=0.05, passed=True),
            MetricScore(name="answer_relevancy", score=pass_rate - 0.05, threshold=0.80, passed=(pass_rate - 0.05) >= 0.80),
        ]
    return EvalResult(
        project_name="test", use_case_type="rag",
        scores=scores, total_samples=100, total_latency_ms=5000, model="bedrock/claude-3-haiku",
    )


class TestPipelineDeployer:
    def test_generate_manifest(self):
        config = EvalConfig.from_yaml(SAMPLE_CONFIG)
        deployer = PipelineDeployer(config)
        manifest = deployer.generate_manifest(environment="prod")
        assert manifest.project_name == "test-eval"
        assert manifest.region == "us-east-1"
        assert len(manifest.lambdas) == 4

    def test_manifest_has_state_machine(self):
        config = EvalConfig.from_yaml(SAMPLE_CONFIG)
        deployer = PipelineDeployer(config)
        manifest = deployer.generate_manifest()
        assert "States" in manifest.state_machine_definition
        assert "LoadData" in manifest.state_machine_definition["States"]

    def test_manifest_has_schedule(self):
        config = EvalConfig.from_yaml(SAMPLE_CONFIG)
        deployer = PipelineDeployer(config)
        manifest = deployer.generate_manifest()
        assert "cron" in manifest.schedule_expression

    def test_sam_template_generation(self):
        config = EvalConfig.from_yaml(SAMPLE_CONFIG)
        deployer = PipelineDeployer(config)
        manifest = deployer.generate_manifest()
        template = manifest.to_sam_template()
        assert "Resources" in template
        assert "AWSTemplateFormatVersion" in template

    def test_manifest_json(self):
        config = EvalConfig.from_yaml(SAMPLE_CONFIG)
        deployer = PipelineDeployer(config)
        manifest = deployer.generate_manifest()
        json_str = manifest.to_json()
        assert "test-eval" in json_str

    def test_manifest_summary(self):
        config = EvalConfig.from_yaml(SAMPLE_CONFIG)
        deployer = PipelineDeployer(config)
        manifest = deployer.generate_manifest()
        summary = manifest.summary()
        assert "test-eval" in summary
        assert "Lambda functions: 4" in summary

    def test_on_demand_no_schedule(self):
        yaml_content = SAMPLE_CONFIG.replace("frequency: daily", "frequency: on_demand")
        config = EvalConfig.from_yaml(yaml_content)
        deployer = PipelineDeployer(config)
        manifest = deployer.generate_manifest()
        assert manifest.schedule_expression == ""


class TestCICDIntegration:
    def test_pass_all_metrics(self):
        cicd = CICDIntegration(min_pass_rate=0.90)
        result = _make_result(0.92)
        gate = cicd.check(result)
        assert gate.decision == GateDecision.PASS
        assert not gate.should_block_deploy
        assert gate.exit_code == 0

    def test_fail_pass_rate(self):
        cicd = CICDIntegration(min_pass_rate=0.95)
        result = _make_result(0.70)
        gate = cicd.check(result)
        assert gate.decision == GateDecision.FAIL
        assert gate.should_block_deploy
        assert gate.exit_code == 1

    def test_fail_required_metric(self):
        cicd = CICDIntegration(required_metrics=["faithfulness"])
        result = _make_result(0.70)  # faithfulness below 0.85
        gate = cicd.check(result)
        assert gate.decision == GateDecision.FAIL
        assert any("faithfulness" in m for m in gate.failing_metrics)

    def test_warn_on_degradation(self):
        previous = _make_result(0.92)
        cicd = CICDIntegration(previous_result=previous, warn_on_degradation=True)
        current = _make_result(0.85)  # dropped from 0.92
        gate = cicd.check(current)
        assert gate.decision == GateDecision.WARN or len(gate.warnings) > 0

    def test_summary_output(self):
        cicd = CICDIntegration()
        result = _make_result(0.92)
        gate = cicd.check(result)
        summary = gate.summary()
        assert "Quality Gate" in summary

    def test_github_output(self):
        cicd = CICDIntegration(min_pass_rate=0.99)
        result = _make_result(0.70)
        gate = cicd.check(result)
        output = gate.to_github_output()
        assert "::error::" in output


class TestReportGenerator:
    def test_generate_report(self):
        gen = ReportGenerator()
        results = [_make_result(0.90), _make_result(0.88), _make_result(0.92)]
        report = gen.generate(results, project_name="test", period="7d")
        assert report.total_evaluations == 3
        assert report.pass_rate > 0

    def test_report_markdown(self):
        gen = ReportGenerator()
        results = [_make_result(0.90)]
        report = gen.generate(results, project_name="test", period="7d")
        md = report.to_markdown()
        assert "# Evaluation Quality Report" in md
        assert "test" in md

    def test_report_json(self):
        gen = ReportGenerator()
        results = [_make_result(0.90)]
        report = gen.generate(results, period="7d")
        json_str = report.to_json()
        assert "pass_rate" in json_str

    def test_executive_summary(self):
        gen = ReportGenerator()
        results = [_make_result(0.90), _make_result(0.85)]
        summary = gen.generate_executive_summary(results)
        assert "evaluations" in summary
        assert "Pass rate" in summary

    def test_recommendations_generated(self):
        gen = ReportGenerator()
        # Low scores should trigger recommendations
        scores = [MetricScore(name="faithfulness", score=0.50, threshold=0.85, passed=False)]
        results = [EvalResult(project_name="t", use_case_type="rag", scores=scores, total_samples=10, total_latency_ms=100, model="m")]
        report = gen.generate(results, period="7d")
        assert len(report.recommendations) > 0
        assert any("faithfulness" in r for r in report.recommendations)

    def test_empty_results(self):
        gen = ReportGenerator()
        summary = gen.generate_executive_summary([])
        assert "No evaluation data" in summary
