/**
 * EvalForge configuration and use case types.
 */

export enum UseCaseType {
  RAG = "rag",
  SUMMARIZATION = "summarization",
  CLASSIFICATION = "classification",
  GENERATION = "generation",
  CHAT = "chat",
  CODE = "code",
}

export const DEFAULT_METRICS: Record<string, string[]> = {
  rag: ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "toxicity"],
  summarization: ["rouge_l", "bleu", "coherence", "conciseness", "fluency"],
  classification: ["accuracy", "precision", "recall", "f1_score"],
  generation: ["fluency", "coherence", "toxicity", "bias_detection"],
  chat: ["coherence", "toxicity", "injection_resistance", "fluency"],
  code: ["accuracy", "coherence"],
};

export const DEFAULT_THRESHOLDS: Record<string, Record<string, number>> = {
  rag: { faithfulness: 0.85, answer_relevancy: 0.80, context_precision: 0.75, context_recall: 0.75, toxicity: 0.05 },
  summarization: { rouge_l: 0.70, bleu: 0.60, coherence: 0.80, conciseness: 0.85, fluency: 0.85 },
  classification: { accuracy: 0.90, precision: 0.85, recall: 0.85, f1_score: 0.85 },
  generation: { fluency: 0.85, coherence: 0.80, toxicity: 0.05, bias_detection: 0.10 },
  chat: { coherence: 0.80, toxicity: 0.05, injection_resistance: 0.90, fluency: 0.85 },
  code: { accuracy: 0.90, coherence: 0.80 },
};

export interface ModelConfig {
  provider: string;
  modelId: string;
  region: string;
}

export interface EvalConfig {
  projectName: string;
  version: string;
  useCaseType: UseCaseType;
  description: string;
  model: ModelConfig;
  metrics: string[];
  thresholds: Record<string, number>;

  getThreshold(metric: string): number;
}

export function createConfig(useCaseType: string, projectName: string = "evaluation"): EvalConfig {
  const ucType = useCaseType as UseCaseType;
  const metrics = DEFAULT_METRICS[ucType] || DEFAULT_METRICS.rag;
  const thresholds = DEFAULT_THRESHOLDS[ucType] || DEFAULT_THRESHOLDS.rag;

  return {
    projectName,
    version: "1.0.0",
    useCaseType: ucType,
    description: `Default ${ucType} evaluation`,
    model: { provider: "bedrock", modelId: "anthropic.claude-3-haiku-20240307-v1:0", region: "us-east-1" },
    metrics,
    thresholds,
    getThreshold(metric: string): number {
      return this.thresholds[metric] ?? 0.8;
    },
  };
}
