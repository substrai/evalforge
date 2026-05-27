"""Tests for the custom metric plugin system."""

import os
import tempfile
from pathlib import Path

import pytest

from evalforge.metrics.base import BaseMetric, MetricInput, MetricOutput
from evalforge.metrics.plugin import (
    MetricMetadata,
    MetricPlugin,
    PluginRegistry,
    get_global_registry,
    metric,
)


@pytest.fixture
def registry():
    """Create a fresh plugin registry for each test."""
    return PluginRegistry()


@pytest.fixture
def sample_input():
    """Create a sample MetricInput."""
    return MetricInput(
        query="What is Python?",
        response="Python is a programming language.",
        context="Python was created by Guido van Rossum.",
    )


class TestMetricDecorator:
    """Tests for the @metric decorator."""

    def test_basic_registration(self, registry):
        """Test that @metric registers a function as a metric."""
        @registry.metric(name="test_metric")
        def test_metric(input: MetricInput) -> float:
            return 0.95

        assert "test_metric" in registry
        assert registry.plugin_count == 1

    def test_default_name_from_function(self, registry):
        """Test that metric name defaults to function name."""
        @registry.metric()
        def my_custom_metric(input: MetricInput) -> float:
            return 0.8

        assert "my_custom_metric" in registry

    def test_metadata_stored(self, registry):
        """Test that metadata is properly stored."""
        @registry.metric(
            name="scored_metric",
            description="A test metric",
            category="text",
            version="2.0.0",
            author="Test Author",
            tags=["quality", "text"],
        )
        def scored_metric(input: MetricInput) -> float:
            return 0.9

        metadata = registry.get_metadata("scored_metric")
        assert metadata.name == "scored_metric"
        assert metadata.description == "A test metric"
        assert metadata.category == "text"
        assert metadata.version == "2.0.0"
        assert metadata.author == "Test Author"
        assert "quality" in metadata.tags

    def test_docstring_as_description(self, registry):
        """Test that function docstring is used as description if not provided."""
        @registry.metric(name="documented")
        def documented(input: MetricInput) -> float:
            """This metric measures documentation quality."""
            return 0.85

        metadata = registry.get_metadata("documented")
        assert "documentation quality" in metadata.description

    def test_decorator_returns_metric_plugin(self, registry):
        """Test that the decorator returns a MetricPlugin instance."""
        @registry.metric(name="plugin_test")
        def plugin_test(input: MetricInput) -> float:
            return 0.7

        assert isinstance(plugin_test, MetricPlugin)


class TestMetricPlugin:
    """Tests for MetricPlugin evaluation."""

    def test_evaluate_returns_float(self, registry, sample_input):
        """Test metric that returns a float score."""
        @registry.metric(name="float_metric")
        def float_metric(input: MetricInput) -> float:
            return 0.85

        result = float_metric.evaluate(sample_input)
        assert isinstance(result, MetricOutput)
        assert result.score == 0.85
        assert result.passed is True  # 0.85 >= 0.8 threshold

    def test_evaluate_returns_metric_output(self, registry, sample_input):
        """Test metric that returns a MetricOutput directly."""
        @registry.metric(name="output_metric")
        def output_metric(input: MetricInput) -> MetricOutput:
            return MetricOutput(
                score=0.92,
                passed=True,
                details={"method": "custom"},
                explanation="High quality response",
            )

        result = output_metric.evaluate(sample_input)
        assert result.score == 0.92
        assert result.details["method"] == "custom"

    def test_evaluate_returns_dict(self, registry, sample_input):
        """Test metric that returns a dictionary."""
        @registry.metric(name="dict_metric")
        def dict_metric(input: MetricInput) -> dict:
            return {
                "score": 0.75,
                "passed": False,
                "explanation": "Below threshold",
                "details_key": "extra_info",
            }

        result = dict_metric.evaluate(sample_input)
        assert result.score == 0.75
        assert result.passed is False
        assert result.explanation == "Below threshold"

    def test_threshold_applied(self, registry, sample_input):
        """Test that threshold determines pass/fail."""
        @registry.metric(name="threshold_test", threshold=0.9)
        def threshold_test(input: MetricInput) -> float:
            return 0.85

        result = threshold_test.evaluate(sample_input, threshold=0.9)
        assert result.score == 0.85
        assert result.passed is False  # 0.85 < 0.9

    def test_multi_arg_function(self, registry, sample_input):
        """Test metric with multiple named arguments."""
        @registry.metric(name="multi_arg")
        def multi_arg(query: str, response: str, context: str) -> float:
            if context and response:
                return 0.9
            return 0.5

        result = multi_arg.evaluate(sample_input)
        assert result.score == 0.9

    def test_error_handling(self, registry, sample_input):
        """Test that evaluation errors are handled gracefully."""
        @registry.metric(name="failing_metric")
        def failing_metric(input: MetricInput) -> float:
            raise ValueError("Something went wrong")

        result = failing_metric.evaluate(sample_input)
        assert result.score == 0.0
        assert result.passed is False
        assert "error" in result.details

    def test_is_base_metric_compatible(self, registry):
        """Test that MetricPlugin is a BaseMetric subclass."""
        @registry.metric(name="compat_test")
        def compat_test(input: MetricInput) -> float:
            return 0.8

        assert isinstance(compat_test, BaseMetric)


class TestPluginRegistry:
    """Tests for PluginRegistry management."""

    def test_get_registered_metric(self, registry, sample_input):
        """Test retrieving a registered metric."""
        @registry.metric(name="retrievable")
        def retrievable(input: MetricInput) -> float:
            return 0.9

        metric = registry.get("retrievable")
        assert metric.name == "retrievable"
        result = metric.evaluate(sample_input)
        assert result.score == 0.9

    def test_get_nonexistent_raises(self, registry):
        """Test that getting a non-existent metric raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_list_by_category(self, registry):
        """Test listing metrics by category."""
        @registry.metric(name="text_m1", category="text")
        def text_m1(input: MetricInput) -> float:
            return 0.8

        @registry.metric(name="safety_m1", category="safety")
        def safety_m1(input: MetricInput) -> float:
            return 0.9

        @registry.metric(name="text_m2", category="text")
        def text_m2(input: MetricInput) -> float:
            return 0.7

        text_metrics = registry.list_by_category("text")
        assert len(text_metrics) == 2
        assert "text_m1" in text_metrics
        assert "text_m2" in text_metrics

    def test_list_by_tag(self, registry):
        """Test listing metrics by tag."""
        @registry.metric(name="tagged1", tags=["quality", "fast"])
        def tagged1(input: MetricInput) -> float:
            return 0.8

        @registry.metric(name="tagged2", tags=["quality", "slow"])
        def tagged2(input: MetricInput) -> float:
            return 0.9

        quality_metrics = registry.list_by_tag("quality")
        assert len(quality_metrics) == 2

        fast_metrics = registry.list_by_tag("fast")
        assert len(fast_metrics) == 1
        assert "tagged1" in fast_metrics

    def test_unregister(self, registry):
        """Test unregistering a metric."""
        @registry.metric(name="removable")
        def removable(input: MetricInput) -> float:
            return 0.5

        assert "removable" in registry
        assert registry.unregister("removable") is True
        assert "removable" not in registry
        assert registry.unregister("removable") is False

    def test_clear(self, registry):
        """Test clearing all plugins."""
        @registry.metric(name="m1")
        def m1(input: MetricInput) -> float:
            return 0.5

        @registry.metric(name="m2")
        def m2(input: MetricInput) -> float:
            return 0.6

        registry.clear()
        assert registry.plugin_count == 0

    def test_manual_register(self, registry):
        """Test manually registering a MetricPlugin."""
        def custom_fn(input: MetricInput) -> float:
            return 0.88

        plugin = MetricPlugin(
            func=custom_fn,
            metadata=MetricMetadata(name="manual_metric", category="custom"),
        )
        registry.register(plugin)

        assert "manual_metric" in registry
        assert registry.get("manual_metric") is plugin


class TestPluginDiscovery:
    """Tests for plugin auto-discovery from directories."""

    def test_discover_from_directory(self, registry):
        """Test discovering plugins from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a plugin file
            plugin_file = Path(tmpdir) / "my_plugin.py"
            plugin_file.write_text('''
from evalforge.metrics.base import MetricInput, MetricOutput
from evalforge.metrics.plugin import get_global_registry

_reg = get_global_registry()

@_reg.metric(name="discovered_metric", category="custom")
def discovered_metric(input: MetricInput) -> float:
    """A metric discovered from a plugin file."""
    return 0.77
''')

            global_reg = get_global_registry()
            initial_count = global_reg.plugin_count
            count = global_reg.discover_plugins(tmpdir)

            assert count >= 1
            assert "discovered_metric" in global_reg

            # Cleanup
            global_reg.unregister("discovered_metric")

    def test_discover_nonexistent_directory(self, registry):
        """Test that non-existent directory returns 0."""
        count = registry.discover_plugins("/nonexistent/path")
        assert count == 0

    def test_discover_empty_directory(self, registry):
        """Test discovering from an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            count = registry.discover_plugins(tmpdir)
            assert count == 0

    def test_discover_skips_init_files(self, registry):
        """Test that __init__.py files are skipped during discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = Path(tmpdir) / "__init__.py"
            init_file.write_text("# init file")

            count = registry.discover_plugins(tmpdir)
            assert count == 0


class TestGlobalRegistry:
    """Tests for the global registry and module-level decorator."""

    def test_global_registry_exists(self):
        """Test that the global registry is accessible."""
        reg = get_global_registry()
        assert isinstance(reg, PluginRegistry)

    def test_module_level_decorator(self):
        """Test the module-level @metric decorator."""
        # Use the global decorator
        @metric(name="_test_global_metric", category="test")
        def _test_global_metric(input: MetricInput) -> float:
            return 0.99

        reg = get_global_registry()
        assert "_test_global_metric" in reg

        # Cleanup
        reg.unregister("_test_global_metric")
