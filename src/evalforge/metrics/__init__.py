"""Built-in metrics for EvalForge."""
from evalforge.metrics.registry import MetricRegistry, get_metrics_for_use_case
from evalforge.metrics.base import BaseMetric, MetricInput, MetricOutput
from evalforge.metrics.text import RougeL, Bleu, Coherence, Conciseness, Fluency
from evalforge.metrics.safety import Toxicity, BiasDetection, InjectionResistance
from evalforge.metrics.rag import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from evalforge.metrics.classification import Accuracy, Precision, Recall, F1Score
