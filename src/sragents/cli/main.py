"""Unified ``sragents`` CLI entry point.

Subcommands:

* ``retrieve`` — Stage 1. Run a retriever over corpus × instances.
* ``hybrid`` — Stage 1. Round-robin fuse two retrieval-result files.
* ``rerank`` — Stage 1. LLM-rerank a retrieval result file.
* ``infer`` — Stage 2. Run Provider × Engine to produce raw_output.
* ``evaluate`` — Stage 3. Score inference results against ground truth.
* ``experiment`` — Run a named paper experiment end-to-end.
* ``list`` — Enumerate registered plugins / datasets / experiments.

External plugins are loaded via ``--plugin my_pkg.my_module`` (repeatable)
or declared as ``[project.entry-points."sragents.retrievers"]`` (or
``.providers`` / ``.engines`` / ``.evaluators``) in any installed
package's pyproject.
"""

import argparse
import importlib
import sys

from sragents.cli import (
    evaluate as _evaluate,
    experiment as _experiment,
    hybrid as _hybrid,
    infer as _infer,
    listing as _listing,
    rerank as _rerank,
    retrieve as _retrieve,
)

_ENTRY_POINT_GROUPS = [
    "sragents.retrievers",
    "sragents.providers",
    "sragents.engines",
    "sragents.evaluators",
    "sragents.prompt_builders",
]


def _load_entry_point_plugins() -> None:
    """Import any plugins declared by installed packages."""
    from importlib.metadata import entry_points
    for group in _ENTRY_POINT_GROUPS:
        for ep in entry_points(group=group):
            try:
                ep.load()
            except Exception as e:  # noqa: BLE001
                print(f"  warning: failed to load plugin {ep.name}: {e}",
                      file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    # BCB single-instance subprocess shim (called by sragents.cli.evaluate).
    if len(sys.argv) >= 5 and sys.argv[1] == "--_eval-one":
        _evaluate._main_shim()
        return

    # Pre-scan for --plugin before argparse initialization so
    # plugin-registered components appear in --help text and in
    # choices= validators.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--plugin", action="append", default=[])
    known, _ = pre.parse_known_args(argv)
    for module in known.plugin:
        try:
            importlib.import_module(module)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"sragents: error: failed to import plugin {module!r}: {e}")
    _load_entry_point_plugins()

    parser = argparse.ArgumentParser(
        prog="sragents",
        description="SR-Agents: baseline skill-retrieval-augmented agents "
                    "on the SRA-Bench benchmark.",
    )
    parser.add_argument(
        "--plugin", action="append", default=[], metavar="MODULE",
        help="Import an external plugin module (repeatable) before "
             "argparse runs. Alternative: declare entry points in your "
             "package's pyproject.toml "
             "([project.entry-points.\"sragents.retrievers\"] etc.).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _retrieve.add_parser(sub)
    _hybrid.add_parser(sub)
    _rerank.add_parser(sub)
    _infer.add_parser(sub)
    _evaluate.add_parser(sub)
    _experiment.add_parser(sub)
    _listing.add_parser(sub)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
