"""JSON schema + IO helpers for retrieval results.

Contract::

    {
      "metadata": {
        "dataset": "theoremqa",   # inferred from instances
        "retriever": "bm25",
        "corpus_size": 26262,
        "top_k": 50,
        "n_queries": 747,
        "timestamp": "2026-04-17T12:00:00Z",
        "extra": {}               # retriever-specific fields
      },
      "metrics": {"Recall@1": 0.5, "nDCG@10": 0.7},   # optional
      "results": [
        {
          "instance_id": "...",
          "gold_skill_ids": ["..."],
          "retrieved": [{"skill_id": "...", "score": 1.23}, ...]
        }
      ]
    }

Any tool that can produce or consume this schema is interoperable with the
rest of the pipeline.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RetrievalRecord:
    instance_id: str
    gold_skill_ids: list[str]
    retrieved: list[dict]  # [{"skill_id": str, "score": float}]


@dataclass
class RetrievalResults:
    retriever: str
    top_k: int
    corpus_size: int
    records: list[RetrievalRecord]
    metrics: dict[str, float] = field(default_factory=dict)
    dataset: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def dump(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata: dict[str, Any] = {}
        if self.dataset:
            metadata["dataset"] = self.dataset
        metadata.update({
            "retriever": self.retriever,
            "top_k": self.top_k,
            "corpus_size": self.corpus_size,
            "n_queries": len(self.records),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "extra": self.extra,
        })
        payload = {
            "metadata": metadata,
            "metrics": self.metrics,
            "results": [asdict(r) for r in self.records],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def load(path: Path) -> dict:
    """Load a retrieval results file (returns raw dict for flexibility)."""
    return json.loads(Path(path).read_text())


def as_lookup(results_file: Path | dict) -> dict[str, list[dict]]:
    """Return ``{instance_id: retrieved_list}`` for quick access."""
    data = results_file if isinstance(results_file, dict) else load(results_file)
    return {r["instance_id"]: r["retrieved"] for r in data["results"]}
