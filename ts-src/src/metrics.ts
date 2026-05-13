/**
 * EvalForge metrics - base interface and built-in implementations.
 */

export interface MetricInput {
  query: string;
  response: string;
  context?: string;
  reference?: string;
}

export interface MetricOutput {
  score: number;
  passed: boolean;
  details: Record<string, any>;
  explanation: string;
}

export abstract class BaseMetric {
  abstract name: string;
  abstract description: string;
  abstract category: string;

  abstract evaluate(input: MetricInput, threshold: number): MetricOutput;

  evaluateBatch(inputs: MetricInput[], threshold: number): MetricOutput[] {
    return inputs.map((inp) => this.evaluate(inp, threshold));
  }

  aggregate(outputs: MetricOutput[]): number {
    if (outputs.length === 0) return 0;
    return outputs.reduce((sum, o) => sum + o.score, 0) / outputs.length;
  }
}

function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/\b\w{3,}\b/g) || []);
}

export class Faithfulness extends BaseMetric {
  name = "faithfulness";
  description = "Is the answer grounded in context?";
  category = "rag";

  evaluate(input: MetricInput, threshold: number = 0.85): MetricOutput {
    if (!input.response || !input.context) {
      return { score: 0, passed: false, details: {}, explanation: "Missing data" };
    }
    const responseWords = new Set(tokenize(input.response));
    const contextWords = new Set(tokenize(input.context));
    const overlap = [...responseWords].filter((w) => contextWords.has(w)).length;
    const score = responseWords.size > 0 ? Math.min(overlap / (responseWords.size * 0.6), 1.0) : 0;
    return { score: Math.round(score * 10000) / 10000, passed: score >= threshold, details: { overlap }, explanation: "" };
  }
}

export class Toxicity extends BaseMetric {
  name = "toxicity";
  description = "Detects toxic content";
  category = "safety";

  private patterns = [/\b(hate|kill|stupid|idiot|moron)\b/gi];

  evaluate(input: MetricInput, threshold: number = 0.05): MetricOutput {
    if (!input.response) return { score: 0, passed: true, details: {}, explanation: "Clean" };
    let matches = 0;
    for (const p of this.patterns) {
      matches += (input.response.match(p) || []).length;
    }
    const score = Math.min(matches / 5, 1.0);
    return { score, passed: score <= threshold, details: { matches }, explanation: "" };
  }
}

export class RougeL extends BaseMetric {
  name = "rouge_l";
  description = "Longest common subsequence overlap";
  category = "text";

  evaluate(input: MetricInput, threshold: number = 0.70): MetricOutput {
    if (!input.response || !input.reference) {
      return { score: 0, passed: false, details: {}, explanation: "Missing data" };
    }
    const respWords = tokenize(input.response);
    const refWords = tokenize(input.reference);
    const overlap = new Set(respWords.filter((w) => refWords.includes(w))).size;
    const precision = respWords.length > 0 ? overlap / respWords.length : 0;
    const recall = refWords.length > 0 ? overlap / refWords.length : 0;
    const score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    return { score: Math.round(score * 10000) / 10000, passed: score >= threshold, details: {}, explanation: "" };
  }
}

export const METRIC_REGISTRY: Record<string, new () => BaseMetric> = {
  faithfulness: Faithfulness,
  toxicity: Toxicity,
  rouge_l: RougeL,
};
