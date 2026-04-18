"""``sragents experiment`` — run a named experiment end-to-end."""

from pathlib import Path

from sragents.experiments import EXPERIMENTS
from sragents.experiments.runner import run_experiment


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "experiment", help="Run a named paper experiment end-to-end",
        description=(
            "Iterates datasets × methods defined by the experiment, "
            "invoking `sragents infer` + `sragents evaluate` for each cell. "
            "Use `sragents list experiments` to see available catalogs."
        ),
    )
    p.add_argument("--exp", required=True,
                   metavar="NAME",
                   help="Experiment name (see `sragents list experiments`)")
    p.add_argument("--model", required=True)
    p.add_argument("--api-base", required=True,
                   help="OpenAI-compatible endpoint URL")
    p.add_argument("--dataset", nargs="*", default=None,
                   help="Restrict to specific dataset(s); default: all "
                        "datasets with both a registered evaluator and "
                        "prompt builder")
    p.add_argument("--methods", nargs="*", default=None,
                   help="Restrict to specific method name(s)")
    p.add_argument("--workspace", type=Path, default=None,
                   help="Root directory for all outputs "
                        "(default: ./results)")
    p.add_argument("--corpus", type=Path, default=None,
                   help="Skill library path (corpus JSON; default: data/bench/corpus/corpus.json)")
    p.add_argument("--instances-dir", type=Path, default=None,
                   help="Instances directory (default: data/bench/instances/)")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--eval-workers", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--thinking", action="store_true")
    p.set_defaults(func=run)


def run(args) -> None:
    if args.exp not in EXPERIMENTS:
        import sys
        sys.exit(
            f"sragents: error: unknown experiment {args.exp!r}. "
            f"Available: {sorted(EXPERIMENTS)}"
        )
    exp = EXPERIMENTS[args.exp]
    print(f"Experiment: {exp.name} — {exp.description}")
    print(f"Model: {args.model}")

    run_experiment(
        exp=exp,
        model=args.model,
        api_base=args.api_base,
        datasets=args.dataset,
        methods=args.methods,
        workspace=args.workspace,
        corpus_path=args.corpus,
        instances_dir=args.instances_dir,
        workers=args.workers,
        eval_workers=args.eval_workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
    )
