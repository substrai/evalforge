"""Parallel metric execution with configurable concurrency.

Provides a ParallelExecutor that runs independent evaluation metrics
concurrently using ThreadPoolExecutor, with dependency-aware scheduling
for metrics that depend on other metrics' outputs.

Key features:
- ThreadPoolExecutor for CPU-bound metric computations
- Configurable max_workers for concurrency control
- Dependency graph resolution (topological sort)
- Per-metric timeout enforcement
- Graceful degradation on metric failures
- Execution statistics and timing
"""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class MetricTask:
    """A metric computation task with dependency information.

    Attributes:
        name: Unique metric identifier.
        fn: The callable that computes the metric.
        depends_on: Names of metrics this task depends on.
        timeout_seconds: Maximum execution time for this metric.
        weight: Priority weight (higher = scheduled first within a layer).
    """

    name: str
    fn: Callable[..., "MetricResult"]
    depends_on: List[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    weight: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    """Result of a single metric execution."""

    name: str
    score: float
    passed: bool
    latency_ms: float = 0.0
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class ExecutionConfig:
    """Configuration for parallel execution.

    Attributes:
        max_workers: Maximum number of concurrent threads.
        default_timeout: Default timeout per metric in seconds.
        fail_fast: Stop execution on first failure.
        retry_count: Number of retries for failed metrics.
        retry_delay_seconds: Delay between retries.
        on_metric_error: Error handling strategy ("skip", "fail", "default").
        default_score: Default score when using "default" error strategy.
    """

    max_workers: int = 4
    default_timeout: float = 30.0
    fail_fast: bool = False
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    on_metric_error: str = "skip"  # "skip", "fail", "default"
    default_score: float = 0.0


@dataclass
class ExecutionReport:
    """Summary report of a parallel execution run.

    Contains all metric results, timing information, and execution statistics.
    """

    results: Dict[str, MetricResult] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    metrics_executed: int = 0
    metrics_succeeded: int = 0
    metrics_failed: int = 0
    metrics_skipped: int = 0
    execution_layers: int = 0

    @property
    def success_rate(self) -> float:
        """Percentage of metrics that succeeded."""
        if self.metrics_executed == 0:
            return 0.0
        return self.metrics_succeeded / self.metrics_executed

    @property
    def all_passed(self) -> bool:
        """Whether all executed metrics passed their thresholds."""
        return all(r.passed for r in self.results.values() if r.success)

    def get_score(self, metric_name: str) -> Optional[float]:
        """Get the score for a specific metric."""
        result = self.results.get(metric_name)
        return result.score if result else None

    def get_failed_metrics(self) -> List[MetricResult]:
        """Get all metrics that failed execution."""
        return [r for r in self.results.values() if not r.success]

    def get_below_threshold(self) -> List[MetricResult]:
        """Get all metrics that scored below their threshold."""
        return [r for r in self.results.values() if r.success and not r.passed]


class ParallelExecutor:
    """Executes evaluation metrics in parallel with dependency awareness.

    Uses ThreadPoolExecutor for concurrent metric computation and respects
    dependency ordering between metrics. Independent metrics run in parallel
    while dependent metrics wait for their prerequisites.

    Example:
        executor = ParallelExecutor(config=ExecutionConfig(max_workers=8))

        executor.add_task(MetricTask(name="relevance", fn=compute_relevance))
        executor.add_task(MetricTask(name="coherence", fn=compute_coherence))
        executor.add_task(MetricTask(
            name="composite",
            fn=compute_composite,
            depends_on=["relevance", "coherence"],
        ))

        report = executor.execute(samples=test_data)
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self._tasks: Dict[str, MetricTask] = {}
        self._execution_history: List[ExecutionReport] = []

    def add_task(self, task: MetricTask) -> "ParallelExecutor":
        """Register a metric task for execution.

        Args:
            task: The MetricTask to add.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If a task with the same name already exists.
        """
        if task.name in self._tasks:
            raise ValueError(f"Task '{task.name}' already registered")
        self._tasks[task.name] = task
        return self

    def remove_task(self, name: str) -> "ParallelExecutor":
        """Remove a registered task by name."""
        self._tasks.pop(name, None)
        return self

    def execute(
        self,
        samples: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionReport:
        """Execute all registered metrics in parallel with dependency ordering.

        Args:
            samples: The evaluation data to pass to each metric function.
            context: Optional shared context available to all metrics.

        Returns:
            ExecutionReport with all results and statistics.
        """
        start = time.perf_counter()
        context = context or {}

        # Build execution layers
        layers = self._build_execution_layers()

        report = ExecutionReport(execution_layers=len(layers))
        completed_results: Dict[str, MetricResult] = {}

        for layer in layers:
            if self.config.fail_fast and report.metrics_failed > 0:
                # Skip remaining layers
                report.metrics_skipped += sum(
                    len(l) for l in layers[layers.index(layer):]
                )
                break

            # Execute all tasks in this layer concurrently
            layer_results = self._execute_layer(
                layer, samples, context, completed_results
            )

            for result in layer_results:
                completed_results[result.name] = result
                report.results[result.name] = result
                report.metrics_executed += 1

                if result.success:
                    report.metrics_succeeded += 1
                else:
                    report.metrics_failed += 1

        report.total_latency_ms = (time.perf_counter() - start) * 1000
        self._execution_history.append(report)
        return report

    def _execute_layer(
        self,
        layer: List[MetricTask],
        samples: Any,
        context: Dict[str, Any],
        prior_results: Dict[str, MetricResult],
    ) -> List[MetricResult]:
        """Execute all tasks in a layer using ThreadPoolExecutor.

        All tasks in a layer are independent and can run concurrently.
        """
        results: List[MetricResult] = []

        # Sort by weight (higher priority first)
        sorted_layer = sorted(layer, key=lambda t: t.weight, reverse=True)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_task: Dict[Future, MetricTask] = {}

            for task in sorted_layer:
                future = executor.submit(
                    self._run_single_metric,
                    task,
                    samples,
                    context,
                    prior_results,
                )
                future_to_task[future] = task

            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result(timeout=task.timeout_seconds)
                    results.append(result)
                except TimeoutError:
                    results.append(MetricResult(
                        name=task.name,
                        score=self.config.default_score,
                        passed=False,
                        error=f"Metric '{task.name}' timed out after {task.timeout_seconds}s",
                    ))
                except Exception as e:
                    results.append(self._handle_metric_error(task, e))

        return results

    def _run_single_metric(
        self,
        task: MetricTask,
        samples: Any,
        context: Dict[str, Any],
        prior_results: Dict[str, MetricResult],
    ) -> MetricResult:
        """Execute a single metric with retry support.

        Passes samples, context, and dependency results to the metric function.
        """
        last_error: Optional[Exception] = None
        attempts = 1 + self.config.retry_count

        for attempt in range(attempts):
            try:
                start = time.perf_counter()

                # Build kwargs for the metric function
                kwargs: Dict[str, Any] = {
                    "samples": samples,
                    "context": context,
                }

                # Include dependency results if the metric has dependencies
                if task.depends_on:
                    dep_results = {
                        dep: prior_results[dep]
                        for dep in task.depends_on
                        if dep in prior_results
                    }
                    kwargs["dependencies"] = dep_results

                result = task.fn(**kwargs)
                latency = (time.perf_counter() - start) * 1000

                # If fn returns a MetricResult, use it directly
                if isinstance(result, MetricResult):
                    result.latency_ms = latency
                    return result

                # If fn returns a dict, convert to MetricResult
                if isinstance(result, dict):
                    return MetricResult(
                        name=task.name,
                        score=result.get("score", 0.0),
                        passed=result.get("passed", False),
                        latency_ms=latency,
                        details=result.get("details", {}),
                    )

                # If fn returns a float, treat as score
                if isinstance(result, (int, float)):
                    return MetricResult(
                        name=task.name,
                        score=float(result),
                        passed=True,
                        latency_ms=latency,
                    )

                raise TypeError(
                    f"Metric '{task.name}' returned unsupported type: {type(result)}"
                )

            except Exception as e:
                last_error = e
                if attempt < attempts - 1:
                    time.sleep(self.config.retry_delay_seconds)

        return self._handle_metric_error(task, last_error)

    def _handle_metric_error(self, task: MetricTask, error: Exception) -> MetricResult:
        """Handle a metric execution error based on configuration."""
        if self.config.on_metric_error == "fail":
            raise RuntimeError(
                f"Metric '{task.name}' failed: {error}"
            ) from error

        return MetricResult(
            name=task.name,
            score=self.config.default_score,
            passed=False,
            error=f"{type(error).__name__}: {error}",
        )

    def _build_execution_layers(self) -> List[List[MetricTask]]:
        """Build execution layers using topological sort.

        Tasks with no dependencies go in layer 0.
        Tasks depending on layer-0 tasks go in layer 1, etc.
        """
        if not self._tasks:
            return []

        task_map = dict(self._tasks)
        in_degree: Dict[str, int] = {name: 0 for name in task_map}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for name, task in task_map.items():
            for dep in task.depends_on:
                if dep in task_map:
                    in_degree[name] += 1
                    dependents[dep].append(name)

        # Kahn's algorithm
        layers: List[List[MetricTask]] = []
        ready = [name for name, deg in in_degree.items() if deg == 0]

        while ready:
            layer = [task_map[name] for name in ready]
            layers.append(layer)

            next_ready = []
            for name in ready:
                for dep_name in dependents[name]:
                    in_degree[dep_name] -= 1
                    if in_degree[dep_name] == 0:
                        next_ready.append(dep_name)
            ready = next_ready

        # Check for cycles
        scheduled = sum(len(layer) for layer in layers)
        if scheduled < len(task_map):
            unscheduled = set(task_map.keys()) - {
                t.name for layer in layers for t in layer
            }
            raise ValueError(
                f"Circular dependency detected among metrics: {unscheduled}"
            )

        return layers

    def validate_dependencies(self) -> List[str]:
        """Validate that all task dependencies are satisfiable.

        Returns:
            List of error messages (empty if valid).
        """
        errors = []
        all_names = set(self._tasks.keys())

        for name, task in self._tasks.items():
            for dep in task.depends_on:
                if dep not in all_names:
                    errors.append(
                        f"Task '{name}' depends on '{dep}' which is not registered"
                    )

        # Check for cycles
        try:
            self._build_execution_layers()
        except ValueError as e:
            errors.append(str(e))

        return errors

    @property
    def task_count(self) -> int:
        """Number of registered tasks."""
        return len(self._tasks)

    @property
    def execution_history(self) -> List[ExecutionReport]:
        """History of all execution runs."""
        return list(self._execution_history)

    def __repr__(self) -> str:
        return (
            f"ParallelExecutor(tasks={len(self._tasks)}, "
            f"max_workers={self.config.max_workers})"
        )
