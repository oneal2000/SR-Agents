"""Top-K provider: read pre-computed retrieval results and take the top K.

Suitable both for static full-skill injection (engine=direct) and for
the agent-style loading mode (engine=progressive_disclosure) — the
engine decides how to use the skills it receives.
"""

import json
import sys
from pathlib import Path

from sragents.corpus import load_corpus_dict
from sragents.infer.base import register_provider


@register_provider("topk")
class TopKProvider:
    """Top-K from a retrieval results file.

    Args:
        source: Path to a retrieval JSON produced by ``sragents retrieve``.
        k: Number of top skills to return per instance.
        corpus_path: Optional override for the skill corpus file.
    """

    def __init__(
        self,
        source: str,
        k: int = 1,
        corpus_path: str | None = None,
    ):
        self._k = int(k)
        self._source = str(source)
        self._corpus = (
            load_corpus_dict(corpus_path) if corpus_path else load_corpus_dict()
        )
        src = Path(source)
        if not src.exists():
            raise FileNotFoundError(
                f"Retrieval source file not found: {src}. "
                "Run `sragents retrieve` to produce it first."
            )
        data = json.loads(src.read_text())
        self._lookup = {r["instance_id"]: r["retrieved"] for r in data["results"]}
        self._warned: set[str] = set()

    def provide(self, instance: dict) -> list[dict]:
        inst_id = instance["instance_id"]
        if inst_id not in self._lookup:
            if inst_id not in self._warned:
                print(
                    f"  [warn] topk: {inst_id} not in {self._source} — "
                    "returning empty skill list",
                    file=sys.stderr,
                )
                self._warned.add(inst_id)
            return []
        retrieved = self._lookup[inst_id][: self._k]
        return [
            self._corpus[r["skill_id"]]
            for r in retrieved
            if r["skill_id"] in self._corpus
        ]
