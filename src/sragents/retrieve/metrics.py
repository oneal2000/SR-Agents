"""Retrieval evaluation metrics: Recall@K, nDCG@K."""

import numpy as np

_MAX_K = 50
_LOG2_DISCOUNT = 1.0 / np.log2(np.arange(2, _MAX_K + 2))  # index 0 → rank 1


def compute_retrieval_metrics(
    results: list[dict],
    top_k: int = 10,
) -> dict[str, float]:
    """Compute Recall@K and nDCG@K for K ∈ {1, 5, 10, 50} (bounded by ``top_k``).

    Each ``results`` entry must have ``gold_skill_ids`` and ``retrieved``
    (a list of ``{skill_id, score}`` dicts).
    """
    ks = [k for k in (1, 5, 10, 50) if k <= top_k]
    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}

    for r in results:
        gold = set(r["gold_skill_ids"])
        retrieved = [entry["skill_id"] for entry in r["retrieved"]]

        n = min(max(ks), len(retrieved))
        rels = np.array([1.0 if sid in gold else 0.0 for sid in retrieved[:n]])

        for k in ks:
            hits = int(rels[:k].sum()) if k <= len(rels) else int(rels.sum())
            recalls[k].append(hits / len(gold) if gold else 0.0)

            cutoff = min(k, len(rels))
            dcg = float(rels[:cutoff] @ _LOG2_DISCOUNT[:cutoff])
            ideal_n = min(len(gold), k)
            idcg = float(_LOG2_DISCOUNT[:ideal_n].sum())
            ndcgs[k].append(dcg / idcg if idcg > 0 else 0.0)

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"Recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
        metrics[f"nDCG@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
    return metrics
