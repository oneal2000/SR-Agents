"""Stage 3: evaluation.

An evaluator maps ``(raw_output, instance) → {extracted_answer, correct, ...}``.
Each dataset has its own; the dispatcher routes by ``instance["dataset"]``.

**raw_output contract**: ``raw_output`` contains only model-generated tokens.
System-injected content (loaded skills, tool results, observation framing)
is persisted in a separate ``transcript`` field by the inference stage;
evaluators must never read that field.

Adding a new evaluator: see :mod:`sragents.evaluate.base`.
"""

# Trigger registration of built-in dataset evaluators.
from sragents.evaluate import datasets as _datasets  # noqa: F401
from sragents.evaluate.base import Evaluator, get, list_datasets, register


def evaluate(raw_output: str, instance: dict) -> dict:
    """Dispatch to the per-dataset evaluator."""
    return get(instance["dataset"])(raw_output, instance)


__all__ = [
    "Evaluator",
    "register",
    "get",
    "list_datasets",
    "evaluate",
]
