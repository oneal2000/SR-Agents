"""Distractor provider: gold skills + hard-negative distractors.

Used for the skill-incorporation noise-robustness study: the model
receives the correct skill mixed with ``n`` hard-negative distractors
sampled alternately from a lexical retriever (BM25) and a semantic
retriever (BGE). Order is shuffled deterministically by
``instance_id`` for reproducibility.
"""

import hashlib
import json
import random
import sys
from pathlib import Path

from sragents.corpus import load_corpus_dict
from sragents.infer.base import register_provider


@register_provider("oracle_distractor")
class OracleDistractorProvider:
    """Gold + N hard-negative distractors, shuffled by a deterministic seed.

    Args:
        lexical_source: Retrieval results used for lexical hard negatives
            (typically ``bm25.json``). Required when ``n > 0``.
        semantic_source: Retrieval results used for semantic hard negatives
            (typically ``bge.json``). Required when ``n > 0``.
        n: Number of distractors to add (0 = oracle only).
        corpus_path: Optional corpus override.
    """

    def __init__(
        self,
        n: int = 0,
        lexical_source: str | None = None,
        semantic_source: str | None = None,
        corpus_path: str | None = None,
    ):
        self._n = int(n)
        self._corpus = (
            load_corpus_dict(corpus_path) if corpus_path else load_corpus_dict()
        )
        self._lex: dict = {}
        self._sem: dict = {}
        if self._n > 0:
            if not lexical_source or not semantic_source:
                raise ValueError(
                    "oracle_distractor with n > 0 requires both "
                    "lexical_source and semantic_source"
                )
            for label, src in (("lexical_source", lexical_source),
                               ("semantic_source", semantic_source)):
                if not Path(src).exists():
                    raise FileNotFoundError(
                        f"{label} file not found: {src}. "
                        "Run `sragents retrieve` to produce it first."
                    )
            lex_data = json.loads(Path(lexical_source).read_text())
            sem_data = json.loads(Path(semantic_source).read_text())
            self._lex = {r["instance_id"]: r["retrieved"] for r in lex_data["results"]}
            self._sem = {r["instance_id"]: r["retrieved"] for r in sem_data["results"]}

    def provide(self, instance: dict) -> list[dict]:
        corpus = self._corpus
        gold_ids = [
            sid for sid in instance.get("skill_annotations", [])
            if sid in corpus
        ]

        distractors: list[str] = []
        if self._n > 0:
            inst_id = instance["instance_id"]
            gold = set(gold_ids)
            pools = [
                [r["skill_id"] for r in self._lex.get(inst_id, [])
                 if r["skill_id"] not in gold and r["skill_id"] in corpus],
                [r["skill_id"] for r in self._sem.get(inst_id, [])
                 if r["skill_id"] not in gold and r["skill_id"] in corpus],
            ]
            seen = set(gold)
            ptrs = [0, 0]
            for i in range(self._n):
                added = False
                for attempt in (i % 2, 1 - i % 2):
                    pool = pools[attempt]
                    while ptrs[attempt] < len(pool) and pool[ptrs[attempt]] in seen:
                        ptrs[attempt] += 1
                    if ptrs[attempt] < len(pool):
                        sid = pool[ptrs[attempt]]
                        distractors.append(sid)
                        seen.add(sid)
                        ptrs[attempt] += 1
                        added = True
                        break
                if not added:
                    # Both retrieval pools exhausted of unseen non-gold
                    # candidates — stop rather than silently return < n.
                    print(
                        f"  [warn] oracle_distractor: {instance['instance_id']} "
                        f"only {len(distractors)}/{self._n} distractors available",
                        file=sys.stderr,
                    )
                    break

        all_ids = gold_ids + distractors
        seed = hashlib.md5(instance["instance_id"].encode()).hexdigest()
        random.Random(seed).shuffle(all_ids)
        return [corpus[sid] for sid in all_ids]
