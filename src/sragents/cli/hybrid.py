"""``sragents hybrid`` — round-robin fusion of two retrieval-result files."""

from pathlib import Path

from sragents.cli._common import require_exists
from sragents.retrieve.hybrid import round_robin_merge


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "hybrid", help="Round-robin fuse two retrieval-result files",
        description="Interleave the ranked lists from two retrieval "
                    "JSON files and write a fused retrieval JSON in the "
                    "same schema.",
    )
    p.add_argument("--input", type=Path, nargs=2, required=True,
                   metavar=("FILE_A", "FILE_B"),
                   help="Two retrieval JSON files (produced by "
                        "`sragents retrieve`)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output retrieval JSON (fused)")
    p.add_argument("--top-k", type=int, default=50,
                   help="Max candidates per query after fusion (default: 50)")
    p.set_defaults(func=run)


def run(args) -> None:
    file_a, file_b = args.input
    require_exists(file_a, "input[0]")
    require_exists(file_b, "input[1]")

    results = round_robin_merge(file_a, file_b, top_k=args.top_k)
    results.dump(args.output)
    print(f"Saved: {args.output}")
    if results.metrics:
        print("  " + "  ".join(f"{k}={v:.4f}"
                               for k, v in results.metrics.items()))
