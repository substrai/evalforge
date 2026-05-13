"""EvalForge CLI - command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evalforge.core.config import EvalConfig, UseCaseType
from evalforge.core.pipeline import EvalPipeline


def cmd_init(args):
    """Initialize a new EvalForge project."""
    project_name = args.name or "my-evaluation"
    use_case = args.use_case or "rag"
    project_dir = Path(project_name)

    if project_dir.exists():
        print(f"Error: Directory '{project_name}' already exists")
        sys.exit(1)

    # Create structure
    dirs = [
        project_dir / "metrics",
        project_dir / "data" / "golden",
        project_dir / "data" / "synthetic",
        project_dir / "judges",
        project_dir / "reports",
        project_dir / "tests",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Create evalforge.yaml
    config = f"""project:
  name: "{project_name}"
  version: "1.0.0"

use_case:
  type: {use_case}
  description: "Evaluation pipeline for {use_case} application"
  model:
    provider: bedrock
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    region: us-east-1

evaluation:
  metrics: auto  # auto-selects based on use_case.type
  thresholds: {{}}  # uses defaults for {use_case}

test_data:
  source: synthetic
  count: 100
  categories: [simple, complex, adversarial, edge_cases]

schedule:
  frequency: daily
  time: "02:00"
"""
    (project_dir / "evalforge.yaml").write_text(config)
    (project_dir / "metrics" / "__init__.py").write_text("")
    (project_dir / "judges" / "__init__.py").write_text("")

    # README
    readme = f"""# {project_name}

LLM evaluation pipeline managed by [EvalForge](https://github.com/substrai/evalforge).

## Quick Start

```bash
evalforge validate
evalforge run
evalforge report --last 7d
```
"""
    (project_dir / "README.md").write_text(readme)

    print(f"✓ Created EvalForge project: {project_name}/")
    print(f"  Use case: {use_case}")
    print(f"  ├── evalforge.yaml")
    print(f"  ├── metrics/")
    print(f"  ├── data/golden/")
    print(f"  ├── data/synthetic/")
    print(f"  └── reports/")
    print(f"\nNext steps:")
    print(f"  cd {project_name}")
    print(f"  evalforge validate")
    print(f"  evalforge run")


def cmd_validate(args):
    """Validate configuration."""
    config_path = Path(args.config or "evalforge.yaml")
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)

    try:
        config = EvalConfig.from_file(config_path)
        print(f"✓ Configuration valid")
        print(config.summary())
    except (ValueError, FileNotFoundError) as e:
        print(f"✗ Configuration error: {e}")
        sys.exit(1)


def cmd_run(args):
    """Run evaluation pipeline."""
    config_path = Path(args.config or "evalforge.yaml")

    if config_path.exists():
        pipeline = EvalPipeline.from_config(config_path)
    else:
        use_case = args.use_case or "rag"
        pipeline = EvalPipeline.for_use_case(use_case)

    metrics_filter = args.metrics.split(",") if args.metrics else None
    results = pipeline.run(metrics=metrics_filter)

    print(results.summary())

    if not results.all_passing:
        sys.exit(1)


def cmd_metrics(args):
    """List available metrics."""
    from evalforge.metrics.registry import MetricRegistry
    registry = MetricRegistry()

    if args.use_case:
        metrics = registry.get_metrics_for(args.use_case)
        print(f"Metrics for '{args.use_case}':")
        for m in metrics:
            print(f"  • {m.name} — {m.description}")
    else:
        print("All available metrics:")
        for name in registry.list_metrics():
            metric = registry.get(name)
            print(f"  • {name} [{metric.category}] — {metric.description}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="evalforge",
        description="EvalForge - Automated LLM Evaluation Pipeline Generator",
    )
    subparsers = parser.add_subparsers(dest="command")

    # init
    init_p = subparsers.add_parser("init", help="Initialize a new project")
    init_p.add_argument("name", nargs="?", default=None)
    init_p.add_argument("--use-case", default="rag", choices=["rag", "summarization", "classification", "generation", "chat", "code"])

    # validate
    val_p = subparsers.add_parser("validate", help="Validate configuration")
    val_p.add_argument("--config", default=None)

    # run
    run_p = subparsers.add_parser("run", help="Run evaluation pipeline")
    run_p.add_argument("--config", default=None)
    run_p.add_argument("--metrics", default=None, help="Comma-separated metrics to run")
    run_p.add_argument("--use-case", default=None)

    # metrics
    met_p = subparsers.add_parser("metrics", help="List available metrics")
    met_p.add_argument("--use-case", default=None)

    args = parser.parse_args()
    commands = {"init": cmd_init, "validate": cmd_validate, "run": cmd_run, "metrics": cmd_metrics}

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
