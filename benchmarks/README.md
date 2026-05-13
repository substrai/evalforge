# EvalForge Benchmarks

Real AWS Bedrock benchmarks demonstrating EvalForge performance and accuracy.

## Prerequisites

1. **Python 3.9+** with EvalForge installed:
   ```bash
   pip install substrai-evalforge[aws]
   ```

2. **AWS credentials** configured with Bedrock access:
   ```bash
   aws configure
   # Or use SSO:
   aws sso login
   ```

3. **Bedrock model access** — ensure `anthropic.claude-3-haiku-20240307-v1:0` is enabled in your AWS account (us-east-1).

## Running the Benchmark

```bash
# From the repo root
cd benchmarks
python run_aws_benchmark.py
```

## What It Tests

| Benchmark | Description |
|-----------|-------------|
| **1. Pipeline Overhead** | Measures EvalForge framework overhead (50 runs) |
| **2. Real LLM Evaluation** | Invokes Bedrock Haiku, evaluates responses with 5 metrics |
| **3. Adversarial Detection** | Tests prompt injection resistance on real LLM responses |
| **4. Synthetic Generation** | Measures data generation speed (50-200 samples) |
| **5. Drift Detection** | Validates drift detection with stable vs degraded inputs |
| **6. Benchmark Comparison** | Compares system scores against published baselines |
| **7. CI/CD Gate** | Tests quality gate pass/fail decision |

## Expected Output

```
======================================================================
EVALFORGE AWS BENCHMARK
======================================================================

--- Benchmark 1: Pipeline Execution Overhead ---
  Pipeline execution (50 runs, 3 default samples, 5 metrics):
    Mean:   0.08 ms
    ...

--- Benchmark 2: Real LLM Evaluation with Bedrock ---
  Q: What is the return policy for electronics?
  A: The return policy for electronics is that they have a 15-day return window.
  ...

BENCHMARK SUMMARY
  EvalForge pipeline overhead: 0.08ms
  Adversarial detection: 3/3 blocked
  Benchmark percentile: 100th
  CI/CD gate: pass
```

## Results

See [RESULTS.md](RESULTS.md) for the full benchmark results from our latest run.

## Cost

Running the full benchmark costs approximately **$0.001** (less than 1 cent) — it makes ~6 Bedrock Haiku calls with minimal tokens.

## Customizing

Edit `run_aws_benchmark.py` to:
- Change the model (`anthropic.claude-3-haiku-20240307-v1:0`)
- Add more test questions
- Adjust thresholds
- Test different use case types
