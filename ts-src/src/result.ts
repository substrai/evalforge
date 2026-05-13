/**
 * EvalForge result models.
 */

export interface MetricScore {
  name: string;
  score: number;
  threshold: number;
  passed: boolean;
  samplesEvaluated: number;
  latencyMs: number;
}

export interface EvalResult {
  projectName: string;
  useCaseType: string;
  timestamp: number;
  scores: MetricScore[];
  totalSamples: number;
  totalLatencyMs: number;
  model: string;
  allPassing: boolean;
  overallScore: number;
  passRate: number;
}

export function createResult(
  projectName: string,
  useCaseType: string,
  scores: MetricScore[],
  totalSamples: number,
  totalLatencyMs: number,
  model: string
): EvalResult {
  const allPassing = scores.every((s) => s.passed);
  const overallScore = scores.length > 0 ? scores.reduce((sum, s) => sum + s.score, 0) / scores.length : 0;
  const passRate = scores.length > 0 ? scores.filter((s) => s.passed).length / scores.length : 0;

  return {
    projectName,
    useCaseType,
    timestamp: Date.now() / 1000,
    scores,
    totalSamples,
    totalLatencyMs,
    model,
    allPassing,
    overallScore: Math.round(overallScore * 10000) / 10000,
    passRate: Math.round(passRate * 10000) / 10000,
  };
}

export function summarizeResult(result: EvalResult): string {
  const status = result.allPassing ? "PASS" : "FAIL";
  const lines = [
    `EvalForge Results: ${result.projectName}`,
    `Status: ${status} (${result.scores.filter((s) => s.passed).length}/${result.scores.length} metrics passing)`,
    `Use case: ${result.useCaseType}`,
    `Samples: ${result.totalSamples}`,
    "",
    "Metrics:",
  ];
  for (const s of result.scores) {
    const icon = s.passed ? "✓" : "✗";
    lines.push(`  ${icon} ${s.name}: ${s.score.toFixed(4)} (threshold: ${s.threshold})`);
  }
  return lines.join("\n");
}
