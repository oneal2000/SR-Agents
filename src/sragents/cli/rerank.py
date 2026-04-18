"""``sragents rerank`` — LLM reranking of a retrieval result file."""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from sragents.cli._common import require_exists
from sragents.corpus import load_corpus_dict
from sragents.llm import create_llm_client, get_extra_body
from sragents.prompts import build_prompt
from sragents.retrieve import compute_retrieval_metrics
from sragents.retrieve.llm_rerank import LLMReranker
from sragents.retrieve.schema import RetrievalRecord, RetrievalResults


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "rerank", help="LLM-rerank a retrieval result file (stage 1)",
        description="Takes a retrieval JSON, uses an LLM to reorder the top-K "
                    "candidates per query, and writes a new retrieval JSON.",
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Input retrieval JSON (e.g. from sragents retrieve)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output retrieval JSON (reranked)")
    p.add_argument("--instances", type=Path, required=True,
                   help="Instances JSON (to re-build queries)")
    p.add_argument("--corpus", type=Path, default=None,
                   help="Corpus JSON (default: package default)")
    p.add_argument("--model", required=True, help="Model for reranking")
    p.add_argument("--api-base", default=None,
                   help="OpenAI-compatible endpoint (default: $OPENAI_API_BASE)")
    p.add_argument("--top-k", type=int, default=50,
                   help="Number of candidates to rerank per query")
    p.add_argument("--workers", type=int, default=32)
    p.set_defaults(func=run)


def run(args) -> None:
    require_exists(args.input, "input")
    require_exists(args.instances, "instances")
    if args.corpus is not None:
        require_exists(args.corpus, "corpus")

    corpus = load_corpus_dict(args.corpus)

    source = json.loads(args.input.read_text())
    source_records = source["results"]

    instances = {i["instance_id"]: i
                 for i in json.loads(args.instances.read_text())}

    # Resume
    existing: dict[str, dict] = {}
    if args.output.exists():
        try:
            prev = json.loads(args.output.read_text())
            if prev.get("metadata", {}).get("extra", {}).get("rerank_k") == args.top_k:
                for r in prev.get("results", []):
                    existing[r["instance_id"]] = r
        except (json.JSONDecodeError, KeyError):
            pass

    pending = [r for r in source_records if r["instance_id"] not in existing]
    if not pending:
        print(f"  all {len(source_records)} already reranked; nothing to do")
        return

    print(f"  {len(pending)} pending / {len(source_records)} total")

    client = create_llm_client(api_base=args.api_base)
    extra = get_extra_body(args.model, thinking=False)
    reranker = LLMReranker(client, args.model, extra_body=extra)
    lock = threading.Lock()
    results = dict(existing)

    def _one(entry: dict) -> dict:
        inst_id = entry["instance_id"]
        candidates_raw = entry["retrieved"][: args.top_k]
        inst = instances.get(inst_id)

        if not candidates_raw or inst is None:
            return {
                "instance_id": inst_id,
                "gold_skill_ids": entry["gold_skill_ids"],
                "retrieved": candidates_raw,
            }

        candidate_skills = [corpus[c["skill_id"]] for c in candidates_raw
                            if c["skill_id"] in corpus]
        if not candidate_skills:
            return {
                "instance_id": inst_id,
                "gold_skill_ids": entry["gold_skill_ids"],
                "retrieved": [],
            }

        _, query = build_prompt(inst)
        reranked = reranker.rerank(query, candidate_skills)

        return {
            "instance_id": inst_id,
            "gold_skill_ids": entry["gold_skill_ids"],
            "retrieved": [{"skill_id": sid, "score": score}
                          for sid, score in reranked],
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_one, e): e for e in pending}
        with tqdm(total=len(pending), desc="  rerank", file=sys.stderr) as bar:
            for fut in as_completed(futures):
                r = fut.result()
                with lock:
                    results[r["instance_id"]] = r
                bar.update(1)

    sorted_records = sorted(results.values(), key=lambda x: x["instance_id"])
    metrics = compute_retrieval_metrics(sorted_records, top_k=args.top_k)

    records_obj = [
        RetrievalRecord(
            instance_id=r["instance_id"],
            gold_skill_ids=r["gold_skill_ids"],
            retrieved=r["retrieved"],
        )
        for r in sorted_records
    ]
    output = RetrievalResults(
        retriever=f"rerank_{source['metadata'].get('retriever', 'source')}",
        top_k=args.top_k,
        corpus_size=source["metadata"].get("corpus_size", len(corpus)),
        records=records_obj,
        metrics=metrics,
        extra={
            "source": str(args.input),
            "model": args.model,
            "rerank_k": args.top_k,
        },
    )
    output.dump(args.output)
    print(f"  Saved: {args.output}")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
