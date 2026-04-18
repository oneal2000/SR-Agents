"""Round-robin fusion of two saved retrieval-result files.

Consumes two ``sragents retrieve`` outputs, interleaves their ranked
lists with deduplication, and produces a new retrieval-result file in
the same schema.
"""

from pathlib import Path

from sragents.retrieve.metrics import compute_retrieval_metrics
from sragents.retrieve.schema import (
    RetrievalRecord,
    RetrievalResults,
    load,
)


def round_robin_merge(
    file_a: Path,
    file_b: Path,
    top_k: int = 50,
) -> RetrievalResults:
    """Merge two retrieval result files into a single RetrievalResults.

    Deduplicates by skill_id. The first appearance wins. Metrics are
    recomputed against the merged ranking.
    """
    import sys

    data_a = load(file_a)
    data_b = load(file_b)

    map_b = {r["instance_id"]: r for r in data_b["results"]}

    records: list[RetrievalRecord] = []
    dropped = 0
    for ra in data_a["results"]:
        rb = map_b.get(ra["instance_id"])
        if rb is None:
            dropped += 1
            continue

        list_a = ra["retrieved"]
        list_b = rb["retrieved"]

        seen: set[str] = set()
        merged: list[dict] = []
        max_rank = max(len(list_a), len(list_b))
        for rank in range(max_rank):
            for lst in (list_a, list_b):
                if rank < len(lst):
                    sid = lst[rank]["skill_id"]
                    if sid not in seen:
                        seen.add(sid)
                        merged.append(lst[rank])
                        if len(merged) == top_k:
                            break
            if len(merged) == top_k:
                break

        records.append(RetrievalRecord(
            instance_id=ra["instance_id"],
            gold_skill_ids=ra["gold_skill_ids"],
            retrieved=merged,
        ))

    if dropped:
        print(
            f"  [warn] hybrid: dropped {dropped} instances present in "
            f"{file_a} but missing in {file_b}",
            file=sys.stderr,
        )

    metrics = compute_retrieval_metrics(
        [{"gold_skill_ids": r.gold_skill_ids, "retrieved": r.retrieved}
         for r in records],
        top_k=top_k,
    )

    name_a = data_a["metadata"].get("retriever", Path(file_a).stem)
    name_b = data_b["metadata"].get("retriever", Path(file_b).stem)
    return RetrievalResults(
        retriever=f"hybrid_{name_a}_{name_b}",
        top_k=top_k,
        corpus_size=data_a["metadata"].get("corpus_size", 0),
        records=records,
        metrics=metrics,
        dataset=data_a["metadata"].get("dataset"),
        extra={"sources": [str(file_a), str(file_b)]},
    )
