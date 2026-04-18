"""Configuration and path constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"
BENCH_DIR = DATA_DIR / "bench"
CORPUS_PATH = BENCH_DIR / "corpus" / "corpus.json"
INSTANCES_DIR = BENCH_DIR / "instances"
RESULTS_DIR = PROJECT_ROOT / "results"

# The six datasets bundled with this release. User-registered datasets are
# discovered dynamically via :func:`discover_datasets` — the experiment
# runner uses that to include plugin datasets automatically.
BUILTIN_DATASETS = [
    "theoremqa", "logicbench", "toolqa", "champ", "medcalcbench", "bigcodebench",
]


def discover_datasets() -> list[str]:
    """Return all datasets that have both a registered evaluator and a
    registered prompt builder — i.e., datasets the pipeline can actually run
    end-to-end. Includes any plugin-registered additions.
    """
    from sragents.evaluate.base import list_datasets as eval_list
    from sragents.prompts import list_datasets as prompt_list
    evaluators = set(eval_list())
    builders = set(prompt_list())
    runnable = evaluators & builders
    # Stable order: built-ins first (in their canonical order), then new ones alphabetically.
    ordered = [d for d in BUILTIN_DATASETS if d in runnable]
    ordered += sorted(runnable - set(BUILTIN_DATASETS))
    return ordered


# Convenience alias for the static built-in list. Use discover_datasets()
# when plugin-registered datasets should also be included.
ALL_DATASETS = BUILTIN_DATASETS


def model_short_name(model: str) -> str:
    """Short directory-safe name for a model (basename of its path)."""
    return Path(model).name
