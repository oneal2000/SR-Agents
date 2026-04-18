"""``sragents retrieve`` — Stage 1."""

import json
from pathlib import Path

from sragents.cli._common import parse_kv_list, require_exists
from sragents.corpus import load_corpus, skill_text
from sragents.prompts import build_prompt
from sragents.retrieve import compute_retrieval_metrics, get, list_retrievers
from sragents.retrieve.schema import RetrievalRecord, RetrievalResults


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "retrieve", help="Run skill retrieval (stage 1)",
        description="Index the skill library with a retriever and save top-K results per query.",
    )
    p.add_argument("--retriever", required=True,
                   help=f"Retriever name. Built-in: {', '.join(list_retrievers()) or '(loading...)'}")
    p.add_argument("--retriever-arg", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="Retriever-specific argument (repeatable)")
    p.add_argument("--corpus", type=Path, required=True,
                   help="Path to the skill library JSON (corpus file)")
    p.add_argument("--instances", type=Path, required=True,
                   help="Path to the instances JSON")
    p.add_argument("--output", type=Path, required=True,
                   help="Output path for the retrieval JSON")
    p.add_argument("--top-k", type=int, default=50,
                   help="Number of top results to store per query (default: 50)")
    p.set_defaults(func=run)


def _build_query(instance: dict) -> str:
    """Concatenate user + system + (if ToolQA) few-shots as the query."""
    system, user = build_prompt(instance)
    parts = [user]
    if system:
        parts.append(system)
    if instance["dataset"] == "toolqa":
        from sragents.toolqa.fewshots import TOOLQA_EXAMPLES
        parts.append(TOOLQA_EXAMPLES)
    return "\n".join(parts)


def run(args) -> None:
    require_exists(args.corpus, "corpus")
    require_exists(args.instances, "instances")

    corpus = load_corpus(args.corpus)
    corpus_ids = [s["skill_id"] for s in corpus]
    corpus_texts = [skill_text(s) for s in corpus]
    n_gold = sum(1 for cid in corpus_ids if not cid.startswith("web_"))
    print(f"Skill library: {len(corpus)} skills ({n_gold} gold "
          f"+ {len(corpus) - n_gold} web)")

    instances = json.loads(args.instances.read_text())
    dataset = instances[0]["dataset"] if instances else None
    queries = []
    gold: dict[str, list[str]] = {}
    for inst in instances:
        gold_ids = inst.get("skill_annotations", [])
        if not gold_ids:
            continue
        queries.append({
            "instance_id": inst["instance_id"],
            "query": _build_query(inst),
        })
        gold[inst["instance_id"]] = gold_ids
    print(f"Queries: {len(queries)}")

    retriever_kwargs = parse_kv_list(args.retriever_arg)
    retriever = get(args.retriever, **retriever_kwargs)

    retriever.build_index(corpus_ids, corpus_texts)
    raw_results = retriever.retrieve(
        [q["query"] for q in queries], args.top_k,
    )

    records = []
    for q, retrieved in zip(queries, raw_results):
        records.append(RetrievalRecord(
            instance_id=q["instance_id"],
            gold_skill_ids=gold[q["instance_id"]],
            retrieved=[{"skill_id": sid, "score": score}
                       for sid, score in retrieved],
        ))

    metrics = compute_retrieval_metrics(
        [{"gold_skill_ids": r.gold_skill_ids, "retrieved": r.retrieved}
         for r in records],
        top_k=args.top_k,
    )
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    results = RetrievalResults(
        retriever=args.retriever,
        top_k=args.top_k,
        corpus_size=len(corpus),
        records=records,
        metrics=metrics,
        dataset=dataset,
    )
    results.dump(args.output)
    print(f"Saved: {args.output}")
