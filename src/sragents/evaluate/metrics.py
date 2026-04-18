"""Aggregate metrics computation for evaluation results."""


def compute_accuracy(results: list[dict]) -> dict:
    """Compute overall accuracy from per-instance evaluation results.

    Args:
        results: List of evaluation result dicts (each has 'correct' bool).

    Returns:
        Dict with accuracy, correct, total.
    """
    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }
