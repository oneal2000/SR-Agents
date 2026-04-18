"""Per-dataset evaluation contract and registry.

An :class:`Evaluator` takes the model's ``raw_output`` plus the bench
instance, extracts an answer, and scores it against ground truth.

Adding a new dataset's evaluator::

    from sragents.evaluate.base import register

    @register("my_dataset")
    def evaluate(raw_output: str, instance: dict) -> dict:
        answer = my_extract(raw_output)
        return {
            "extracted_answer": answer,
            "correct": answer == instance["eval_data"]["answer"],
        }

The dispatcher in :mod:`sragents.evaluate` routes by
``instance["dataset"]``.
"""

from typing import Callable, Protocol


class Evaluator(Protocol):
    """Callable ``(raw_output, instance) -> result_dict``.

    Result dict must contain at least ``extracted_answer`` (str) and
    ``correct`` (bool). Evaluators may add any extra fields (they flow
    through to the per-instance details list).
    """

    def __call__(self, raw_output: str, instance: dict) -> dict: ...


_REGISTRY: dict[str, Callable[[str, dict], dict]] = {}


def register(dataset: str):
    def wrap(fn):
        _REGISTRY[dataset] = fn
        return fn
    return wrap


def get(dataset: str) -> Callable[[str, dict], dict]:
    if dataset not in _REGISTRY:
        raise KeyError(
            f"No evaluator for dataset {dataset!r}. "
            f"Registered: {list_datasets()}"
        )
    return _REGISTRY[dataset]


def list_datasets() -> list[str]:
    return sorted(_REGISTRY)
