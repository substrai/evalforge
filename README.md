# EvalForge

**Automated LLM evaluation pipeline generator.**

> Built by [SubstrAI](https://github.com/substrai) — Open-source GenAI frameworks for serverless infrastructure.

[![PyPI version](https://badge.fury.io/py/substrai-evalforge.svg)](https://pypi.org/project/substrai-evalforge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## The Problem

Every team deploying LLMs builds evaluation pipelines from scratch. RAGAS and DeepEval are libraries — they don't generate infrastructure, schedule runs, detect drift, or route to human reviewers.

## The Solution

Describe your use case → EvalForge generates the complete evaluation pipeline:

```yaml
# evalforge.yaml
use_case:
  type: rag
  description: "Customer support chatbot"
  model:
    provider: bedrock
    model_id: anthropic.claude-3-haiku-20240307-v1:0

evaluation:
  metrics: auto  # auto-selects: faithfulness, relevancy, precision, recall, toxicity
```

```bash
evalforge run
# Faithfulness:      0.91 ✓ (threshold: 0.85)
# Answer Relevancy:  0.87 ✓ (threshold: 0.80)
# Context Precision: 0.78 ✓ (threshold: 0.75)
# Toxicity:          0.02 ✓ (threshold: 0.05)
# Overall: PASS (4/4 metrics passing)
```

## Features

- **Use-case-driven metric selection** — describe your app, get optimal metrics
- **6 use case types** — RAG, summarization, classification, generation, chat, code
- **16+ built-in metrics** — faithfulness, ROUGE, BLEU, toxicity, injection resistance, F1
- **Synthetic test data generation** — adversarial, edge cases, domain-specific
- **Drift detection** — alerts when quality degrades over time
- **Human-in-the-loop** — route uncertain evaluations to reviewers
- **Scheduled pipelines** — daily/weekly automated evaluation runs
- **Benchmark registry** — compare against published benchmarks
- **One-command deploy** — Step Functions + Lambda infrastructure

## Installation

```bash
pip install substrai-evalforge
```

## Quick Start

```bash
# Scaffold project
evalforge init my-eval --use-case rag

# Run evaluation
cd my-eval
evalforge run

# List available metrics
evalforge metrics --use-case rag
```

## Python SDK

```python
from evalforge import EvalPipeline

# Quick start for any use case
pipeline = EvalPipeline.for_use_case("rag")
results = pipeline.run()
print(results.summary())
print(f"All passing: {results.all_passing}")
```

## Supported Use Cases & Auto-Selected Metrics

| Use Case | Auto-Selected Metrics |
|---|---|
| **rag** | faithfulness, answer_relevancy, context_precision, context_recall, toxicity |
| **summarization** | rouge_l, bleu, coherence, conciseness, fluency |
| **classification** | accuracy, precision, recall, f1_score |
| **generation** | fluency, coherence, toxicity, bias_detection |
| **chat** | coherence, toxicity, injection_resistance, fluency |
| **code** | accuracy, coherence |

## License

MIT — see [LICENSE](LICENSE)

## Author

**Gaurav Kumar Sinha** — Founder, [SubstrAI](https://github.com/substrai)

- Email: gaurav@substrai.dev
- GitHub: [@substrai](https://github.com/substrai)
