"""Experiment runner: drive ``sragents infer`` + ``sragents evaluate`` for
every (dataset, method) in an :class:`ExperimentSpec`.

File layout under ``--workspace`` (default ``./results``)::

    {workspace}/retrieval/{dataset}-{source}.json        ← stage 1 outputs
    {workspace}/inference/{dataset}/{model}/{label}.jsonl ← stage 2
    {workspace}/eval/{dataset}/{model}/{label}.json      ← stage 3

Each stage's CLI supports per-instance / per-file resume, so re-running is
safe and idempotent.
"""

import subprocess
import sys
from pathlib import Path

from sragents.config import (
    CORPUS_PATH,
    INSTANCES_DIR,
    RESULTS_DIR,
    discover_datasets,
    model_short_name,
)
from sragents.experiments.definitions import EXPERIMENTS, ExperimentSpec, Method


def _retrieval_file(workspace: Path, dataset: str, source: str,
                    rerank_model: str | None = None) -> Path:
    base = workspace / "retrieval"
    if source == "rerank_bm25" and rerank_model:
        return base / f"{dataset}-rerank_bm25-{model_short_name(rerank_model)}.json"
    return base / f"{dataset}-{source}.json"


def _infer_file(workspace: Path, dataset: str, model: str, label: str) -> Path:
    return workspace / "inference" / dataset / model_short_name(model) / f"{label}.jsonl"


def _eval_file(workspace: Path, dataset: str, model: str, label: str) -> Path:
    return workspace / "eval" / dataset / model_short_name(model) / f"{label}.json"


def _run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _maybe_rerank(
    workspace: Path, dataset: str, model: str, api_base: str,
    instances_file: Path, corpus_path: Path, workers: int,
) -> Path | None:
    """Ensure a reranked BM25 file exists for this (dataset, model). Returns its path."""
    rerank_path = _retrieval_file(workspace, dataset, "rerank_bm25",
                                  rerank_model=model)
    if rerank_path.exists():
        return rerank_path
    bm25_path = _retrieval_file(workspace, dataset, "bm25")
    if not bm25_path.exists():
        print(f"  [skip rerank] {bm25_path} missing", file=sys.stderr)
        return None
    rc = _run([
        sys.executable, "-m", "sragents.cli.main", "rerank",
        "--input", str(bm25_path),
        "--output", str(rerank_path),
        "--instances", str(instances_file),
        "--corpus", str(corpus_path),
        "--model", model,
        "--api-base", api_base,
        "--top-k", "50",
        "--workers", str(workers),
    ])
    if rc != 0:
        print(f"  [warn] rerank failed (exit={rc})", file=sys.stderr)
        return None
    return rerank_path


def run_experiment(
    exp: ExperimentSpec,
    model: str,
    api_base: str,
    *,
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
    workspace: Path | None = None,
    corpus_path: Path | None = None,
    instances_dir: Path | None = None,
    workers: int = 32,
    eval_workers: int = 32,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    thinking: bool = False,
) -> None:
    workspace = Path(workspace) if workspace else RESULTS_DIR
    corpus_path = Path(corpus_path) if corpus_path else CORPUS_PATH
    instances_dir = Path(instances_dir) if instances_dir else INSTANCES_DIR

    dss = datasets or discover_datasets()
    method_filter = set(methods) if methods else None

    label_suffix = "_thinking" if thinking else ""

    for m in exp.methods:
        if method_filter and m.label not in method_filter:
            continue
        for ds in dss:
            instances_file = instances_dir / f"{ds}.json"
            if not instances_file.exists():
                print(f"  [skip] {ds}: instances file missing", file=sys.stderr)
                continue

            # Resolve retrieval-source paths for this provider.
            provider_args = dict(m.provider_args)
            skip_this_cell = False

            source_name = provider_args.get("source")
            if source_name is not None:
                if source_name == "rerank_bm25":
                    rerank_path = _maybe_rerank(
                        workspace, ds, model, api_base,
                        instances_file, corpus_path, workers,
                    )
                    if rerank_path is None:
                        continue
                    provider_args["source"] = str(rerank_path)
                else:
                    p = _retrieval_file(workspace, ds, source_name)
                    if not p.exists():
                        print(f"  [skip] {ds}/{m.label}: retrieval file "
                              f"missing: {p}", file=sys.stderr)
                        continue
                    provider_args["source"] = str(p)

            # Hard-negative distractor sources (oracle_distractor provider).
            for key in ("lexical_source", "semantic_source"):
                if key not in provider_args:
                    continue
                p = _retrieval_file(workspace, ds, provider_args[key])
                if not p.exists():
                    print(f"  [skip] {ds}/{m.label}: hard-negative "
                          f"distractor file missing: {p}", file=sys.stderr)
                    skip_this_cell = True
                    break
                provider_args[key] = str(p)
            if skip_this_cell:
                continue

            # Forward corpus_path to the provider only when the user
            # explicitly chose a non-default location.
            if Path(corpus_path) != Path(CORPUS_PATH):
                provider_args.setdefault("corpus_path", str(corpus_path))

            label = m.label + label_suffix
            infer_path = _infer_file(workspace, ds, model, label)
            eval_path = _eval_file(workspace, ds, model, label)

            cmd = [
                sys.executable, "-m", "sragents.cli.main", "infer",
                "--instances", str(instances_file),
                "--output", str(infer_path),
                "--model", model,
                "--api-base", api_base,
                "--provider", m.provider,
                "--engine", m.resolve_engine(ds),
                "--label", label,
                "--workers", str(workers),
                "--temperature", str(temperature),
                "--max-tokens", str(max_tokens),
            ]
            if thinking:
                cmd.append("--thinking")
            for k, v in provider_args.items():
                cmd += ["--provider-arg", f"{k}={v}"]
            for k, v in m.engine_args.items():
                cmd += ["--engine-arg", f"{k}={v}"]
            infer_rc = _run(cmd)
            if infer_rc != 0:
                print(f"  [warn] infer failed for {ds}/{label} "
                      f"(exit={infer_rc}); skipping evaluate",
                      file=sys.stderr)
                continue

            if infer_path.exists() and not eval_path.exists():
                _run([
                    sys.executable, "-m", "sragents.cli.main", "evaluate",
                    "--input", str(infer_path),
                    "--instances", str(instances_file),
                    "--output", str(eval_path),
                    "--workers", str(eval_workers),
                ])


def list_experiments() -> str:
    lines = ["Available experiments:\n"]
    for name, exp in EXPERIMENTS.items():
        lines.append(f"  {name:<18s} {exp.description}")
        for m in exp.methods:
            lines.append(f"      • {m.label}")
    return "\n".join(lines)
