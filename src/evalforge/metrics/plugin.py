"""Custom metric plugin system with @metric decorator for EvalForge.

Provides a decorator-based plugin system for defining custom evaluation
metrics with auto-registration, plugin directory auto-discovery, and
a MetricRegistry for managing custom metrics.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type

from evalforge.metrics.base import BaseMetric, MetricInput, MetricOutput

logger = logging.getLogger("evalforge.metrics.plugin")


@dataclass
class MetricMetadata:
    """Metadata for a registered metric plugin.

    Attributes:
        name: Unique metric name.
        description: Human-readable description.
        category: Metric category (rag, text, safety, etc.).
        version: Metric version string.
        author: Metric author.
        tags: Tags for categorization and discovery.
        source_file: Path to the source file where the metric is defined.
    """
    name: str
    description: str = ""
    category: str = "custom"
    version: str = "1.0.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)
    source_file: Optional[str] = None


class MetricPlugin(BaseMetric):
    """Wrapper that turns a decorated function into a BaseMetric-compatible class.

    This allows functions decorated with @metric to be used anywhere
    a BaseMetric is expected.
    """

    def __init__(
        self,
        func: Callable,
        metadata: MetricMetadata,
        threshold: float = 0.8,
    ):
        self._func = func
        self._metadata = metadata
        self._threshold = threshold
        self.name = metadata.name
        self.description = metadata.description
        self.category = metadata.category

    @property
    def metadata(self) -> MetricMetadata:
        """Get the metric metadata."""
        return self._metadata

    def evaluate(self, input: MetricInput, threshold: float = 0.8) -> MetricOutput:
        """Evaluate using the decorated function.

        The decorated function can accept various signatures:
        - func(input: MetricInput) -> MetricOutput
        - func(input: MetricInput) -> float (score only)
        - func(query, response, context) -> float
        - func(**kwargs) -> float | dict | MetricOutput
        """
        effective_threshold = threshold or self._threshold

        try:
            result = self._call_func(input)
            return self._normalize_result(result, effective_threshold)
        except Exception as e:
            logger.error(f"Metric '{self.name}' evaluation failed: {e}")
            return MetricOutput(
                score=0.0,
                passed=False,
                details={"error": str(e)},
                explanation=f"Evaluation failed: {e}",
            )

    def _call_func(self, input: MetricInput) -> Any:
        """Call the decorated function with appropriate arguments."""
        sig = inspect.signature(self._func)
        params = list(sig.parameters.keys())

        # Try different calling conventions
        if len(params) == 1:
            param = sig.parameters[params[0]]
            if param.annotation == MetricInput or params[0] == "input":
                return self._func(input)
            return self._func(input)

        # Multi-argument: try (query, response, context)
        kwargs = {}
        param_mapping = {
            "query": input.query,
            "response": input.response,
            "context": input.context,
            "reference": input.reference,
            "metadata": input.metadata,
            "input": input,
        }

        for param_name in params:
            if param_name in param_mapping:
                kwargs[param_name] = param_mapping[param_name]

        if kwargs:
            return self._func(**kwargs)

        # Fallback: pass the input object
        return self._func(input)

    def _normalize_result(self, result: Any, threshold: float) -> MetricOutput:
        """Normalize various return types into MetricOutput."""
        if isinstance(result, MetricOutput):
            return result

        if isinstance(result, (int, float)):
            score = float(result)
            return MetricOutput(
                score=score,
                passed=score >= threshold,
            )

        if isinstance(result, dict):
            score = float(result.get("score", 0.0))
            return MetricOutput(
                score=score,
                passed=result.get("passed", score >= threshold),
                details={k: v for k, v in result.items() if k not in ("score", "passed", "explanation")},
                explanation=result.get("explanation", ""),
            )

        # Fallback: try to convert to float
        try:
            score = float(result)
            return MetricOutput(score=score, passed=score >= threshold)
        except (TypeError, ValueError):
            return MetricOutput(
                score=0.0,
                passed=False,
                details={"raw_result": str(result)},
                explanation="Could not interpret metric result",
            )


class PluginRegistry:
    """Registry for managing custom metric plugins.

    Provides auto-discovery from plugin directories, registration via
    the @metric decorator, and lookup/listing of registered metrics.

    Usage:
        registry = PluginRegistry()

        # Register via decorator
        @registry.metric(name="my_metric", category="custom")
        def my_metric(input: MetricInput) -> float:
            return 0.95

        # Discover plugins from directory
        registry.discover_plugins("/path/to/plugins")

        # Use registered metrics
        metric = registry.get("my_metric")
        result = metric.evaluate(MetricInput(query="test", response="answer"))
    """

    def __init__(self):
        self._plugins: Dict[str, MetricPlugin] = {}
        self._metadata: Dict[str, MetricMetadata] = {}
        self._discovery_paths: List[Path] = []

    @property
    def registered_metrics(self) -> List[str]:
        """List all registered metric names."""
        return sorted(self._plugins.keys())

    @property
    def plugin_count(self) -> int:
        """Get the number of registered plugins."""
        return len(self._plugins)

    def metric(
        self,
        name: Optional[str] = None,
        description: str = "",
        category: str = "custom",
        version: str = "1.0.0",
        author: str = "",
        tags: Optional[List[str]] = None,
        threshold: float = 0.8,
    ) -> Callable:
        """Decorator to register a function as a metric plugin.

        Args:
            name: Metric name (defaults to function name).
            description: Human-readable description.
            category: Metric category.
            version: Version string.
            author: Author name.
            tags: Tags for categorization.
            threshold: Default pass/fail threshold.

        Example:
            @registry.metric(name="coherence_v2", category="text")
            def coherence_v2(input: MetricInput) -> float:
                # Custom coherence scoring logic
                return compute_coherence(input.response)
        """
        def decorator(func: Callable) -> MetricPlugin:
            metric_name = name or func.__name__
            metric_desc = description or (func.__doc__ or "").strip()

            metadata = MetricMetadata(
                name=metric_name,
                description=metric_desc,
                category=category,
                version=version,
                author=author,
                tags=tags or [],
                source_file=inspect.getfile(func) if hasattr(func, "__code__") else None,
            )

            plugin = MetricPlugin(
                func=func,
                metadata=metadata,
                threshold=threshold,
            )

            self._plugins[metric_name] = plugin
            self._metadata[metric_name] = metadata
            logger.debug(f"Registered metric plugin: {metric_name} (v{version})")

            return plugin

        return decorator

    def register(self, plugin: MetricPlugin) -> None:
        """Manually register a metric plugin.

        Args:
            plugin: The MetricPlugin instance to register.
        """
        self._plugins[plugin.name] = plugin
        self._metadata[plugin.name] = plugin.metadata

    def get(self, name: str) -> MetricPlugin:
        """Get a registered metric by name.

        Args:
            name: The metric name.

        Returns:
            The MetricPlugin instance.

        Raises:
            KeyError: If the metric is not registered.
        """
        if name not in self._plugins:
            available = self.registered_metrics
            raise KeyError(
                f"Metric '{name}' not found. Available: {available}"
            )
        return self._plugins[name]

    def get_metadata(self, name: str) -> MetricMetadata:
        """Get metadata for a registered metric."""
        if name not in self._metadata:
            raise KeyError(f"Metric '{name}' not found")
        return self._metadata[name]

    def list_by_category(self, category: str) -> List[str]:
        """List metrics filtered by category."""
        return [
            name for name, meta in self._metadata.items()
            if meta.category == category
        ]

    def list_by_tag(self, tag: str) -> List[str]:
        """List metrics that have a specific tag."""
        return [
            name for name, meta in self._metadata.items()
            if tag in meta.tags
        ]

    def discover_plugins(self, directory: str, recursive: bool = True) -> int:
        """Auto-discover and load metric plugins from a directory.

        Scans the directory for Python files and imports them, which
        triggers any @metric decorators to register their metrics.

        Args:
            directory: Path to the plugin directory.
            recursive: Whether to scan subdirectories.

        Returns:
            Number of new metrics discovered.
        """
        plugin_dir = Path(directory)
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory does not exist: {directory}")
            return 0

        self._discovery_paths.append(plugin_dir)
        initial_count = self.plugin_count
        discovered_files: List[Path] = []

        if recursive:
            discovered_files = list(plugin_dir.rglob("*.py"))
        else:
            discovered_files = list(plugin_dir.glob("*.py"))

        # Filter out __pycache__ and __init__ files
        discovered_files = [
            f for f in discovered_files
            if "__pycache__" not in str(f) and f.name != "__init__.py"
        ]

        for file_path in discovered_files:
            try:
                self._load_plugin_file(file_path)
            except Exception as e:
                logger.warning(f"Failed to load plugin {file_path}: {e}")

        new_count = self.plugin_count - initial_count
        logger.info(f"Discovered {new_count} new metrics from {directory}")
        return new_count

    def _load_plugin_file(self, file_path: Path) -> None:
        """Load a single plugin file by importing it."""
        module_name = f"evalforge_plugin_{file_path.stem}"

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            del sys.modules[module_name]
            raise RuntimeError(f"Error loading plugin {file_path.name}: {e}") from e

    def unregister(self, name: str) -> bool:
        """Unregister a metric plugin.

        Args:
            name: The metric name to unregister.

        Returns:
            True if the metric was found and removed.
        """
        if name in self._plugins:
            del self._plugins[name]
            del self._metadata[name]
            return True
        return False

    def clear(self) -> None:
        """Remove all registered plugins."""
        self._plugins.clear()
        self._metadata.clear()

    def __contains__(self, name: str) -> bool:
        """Check if a metric is registered."""
        return name in self._plugins

    def __len__(self) -> int:
        """Get the number of registered plugins."""
        return self.plugin_count

    def __repr__(self) -> str:
        return f"PluginRegistry(metrics={self.registered_metrics})"


# Global registry instance
_global_registry = PluginRegistry()


def metric(
    name: Optional[str] = None,
    description: str = "",
    category: str = "custom",
    version: str = "1.0.0",
    author: str = "",
    tags: Optional[List[str]] = None,
    threshold: float = 0.8,
) -> Callable:
    """Module-level @metric decorator using the global registry.

    Example:
        from evalforge.metrics.plugin import metric

        @metric(name="custom_score", category="text")
        def custom_score(input: MetricInput) -> float:
            return len(input.response) / 100.0
    """
    return _global_registry.metric(
        name=name,
        description=description,
        category=category,
        version=version,
        author=author,
        tags=tags,
        threshold=threshold,
    )


def get_global_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return _global_registry
