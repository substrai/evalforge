/**
 * EvalForge pipeline - orchestrates metric execution.
 */

import { EvalConfig, createConfig, DEFAULT_METRICS } from "./config";
import { BaseMetric, MetricInput, METRIC_REGISTRY } from "./metrics";
import { EvalResult, MetricScore, createResult } from "./result";

export interface TestSample {
  query: string;
  response: string;
  context?: string;
  reference?: string;
  category?: string;
}

export class EvalPipeline {
  private config: EvalConfig;
  private metrics: BaseMetric[] = [];

  constructor(config: EvalConfig) {
    this.config = config;
    this.loadMetrics();
  }

  static forUseCase(useCase: string, projectName: string = "evaluation"): EvalPipeline {
    return new EvalPipeline(createConfig(useCase, projectName));
  }

  private loadMetrics(): void {
    for (const name of this.config.metrics) {
      const MetricClass = METRIC_REGISTRY[name];
      if (MetricClass) {
        this.metrics.push(new MetricClass());
      }
    }
  }

  run(samples?: TestSample[], metricFilter?: string[]): EvalResult {
    const start = Date.now();
    const testSamples = samples || this.defaultSamples();

    const metricsToRun = metricFilter
      ? this.metrics.filter((m) => metricFilter.includes(m.name))
      : this.metrics;

    const scores: MetricScore[] = [];

    for (const metric of metricsToRun) {
      const threshold = this.config.getThreshold(metric.name);
      const metricStart = Date.now();

      const outputs = testSamples.map((sample) =>
        metric.evaluate(
          { query: sample.query, response: sample.response, context: sample.context, reference: sample.reference },
          threshold
        )
      );

      const avgScore = outputs.length > 0 ? outputs.reduce((s, o) => s + o.score, 0) / outputs.length : 0;
      const isSafety = metric.category === "safety";
      const passed = isSafety ? avgScore <= threshold : avgScore >= threshold;

      scores.push({
        name: metric.name,
        score: Math.round(avgScore * 10000) / 10000,
        threshold,
        passed,
        samplesEvaluated: outputs.length,
        latencyMs: Date.now() - metricStart,
      });
    }

    return createResult(
      this.config.projectName,
      this.config.useCaseType,
      scores,
      testSamples.length,
      Date.now() - start,
      `${this.config.model.provider}/${this.config.model.modelId}`
    );
  }

  listMetrics(): string[] {
    return this.metrics.map((m) => m.name);
  }

  private defaultSamples(): TestSample[] {
    return [
      {
        query: "What is the return policy?",
        response: "Our return policy allows returns within 30 days with a valid receipt.",
        context: "Return Policy: Customers may return items within 30 days. A valid receipt is required.",
        reference: "Returns accepted within 30 days with receipt.",
      },
      {
        query: "How do I reset my password?",
        response: "Go to Settings > Security > Reset Password.",
        context: "Password Reset: Navigate to Settings, then Security, then Reset Password.",
        reference: "Go to Settings > Security > Reset Password.",
      },
    ];
  }
}
