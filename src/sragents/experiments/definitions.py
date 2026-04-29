"""Paper experiment catalog.

Each :class:`Method` names a provider + engine combination with their
arguments. A :class:`Method`'s ``label`` is what appears in output file
names and aggregation tables.

Experiments defined here are the ones in the paper; adding more just
means appending entries to :data:`EXPERIMENTS` (or declaring a new
plugin).

Some methods need a different engine for ToolQA (which uses ReAct) than
for the rest of the datasets. That is expressed by ``engine_toolqa`` on
the :class:`Method`. When unset, the engine applies to all datasets.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Method:
    """One concrete (provider, engine) combination with a label.

    ``label`` is the short snake_case identifier used in output file
    paths and on the CLI. ``display_name`` is the human-readable name
    (as it appears in the paper); when unset, the label is used.
    """

    label: str
    provider: str
    provider_args: dict = field(default_factory=dict)
    engine: str = "direct"
    engine_args: dict = field(default_factory=dict)
    engine_toolqa: str | None = None
    """Override engine for the ToolQA dataset (typically ``react`` or
    ``react_progressive_disclosure``)."""
    display_name: str | None = None

    def resolve_engine(self, dataset: str) -> str:
        if dataset == "toolqa" and self.engine_toolqa:
            return self.engine_toolqa
        return self.engine

    def display(self) -> str:
        return self.display_name or self.label


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    description: str
    methods: list[Method] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Paper experiments
# ---------------------------------------------------------------------------
# Retrieval sources produced by ``sragents retrieve`` are referenced here
# by *logical name* (e.g. ``bm25``); the runner resolves to actual file
# paths under the workspace: ``{workspace}/retrieval/{dataset}-{source}.json``.

_MAIN = ExperimentSpec(
    name="main",
    description="Main experiment: 5 skill-use methods × all datasets.",
    methods=[
        Method(
            label="llm_direct",
            display_name="LLM Direct",
            provider="none",
            engine="direct",
            engine_toolqa="react",
        ),
        Method(
            label="oracle_skill",
            display_name="Oracle Skill",
            provider="oracle",
            engine="direct",
            engine_toolqa="react",
        ),
        Method(
            label="bm25_top1",
            display_name="Full-Skill Injection",
            provider="topk",
            provider_args={"source": "bm25", "k": 1},
            engine="direct",
            engine_toolqa="react",
        ),
        Method(
            label="bm25_select",
            display_name="LLM Selection",
            provider="llm_select",
            provider_args={"source": "bm25", "pool": 50},
            engine="direct",
            engine_toolqa="react",
        ),
        Method(
            label="progressive_disclosure",
            display_name="Progressive Disclosure",
            provider="topk",
            provider_args={"source": "bm25", "k": 50},
            engine="progressive_disclosure",
            engine_toolqa="react_progressive_disclosure",
        ),
    ],
)

_RETRIEVAL = ExperimentSpec(
    name="retrieval_comparison",
    description="End-to-end performance under different retrievers (rank 1). "
                "`bm25_top1` is shared with `main` and `topk_sweep`; if any "
                "of those has run first, the runner reuses its result here.",
    methods=[
        # ``bm25_top1`` is shared with ``_MAIN`` and ``_TOPK_SWEEP``. Kept
        # here so this experiment is self-contained (running it standalone
        # produces all 5 rows of the paper's retriever-comparison table).
        # The shared label means the runner reuses whichever sibling cell
        # already exists instead of re-running bm25_top1 from scratch.
        Method(
            label="bm25_top1",
            display_name="BM25",
            provider="topk",
            provider_args={"source": "bm25", "k": 1},
            engine="direct",
            engine_toolqa="react",
        ),
        Method(
            label="tfidf_top1",
            display_name="TF-IDF",
            provider="topk",
            provider_args={"source": "tfidf", "k": 1},
            engine_toolqa="react",
        ),
        Method(
            label="bge_top1",
            display_name="BGE",
            provider="topk",
            provider_args={"source": "bge", "k": 1},
            engine_toolqa="react",
        ),
        Method(
            label="contriever_top1",
            display_name="Contriever",
            provider="topk",
            provider_args={"source": "contriever", "k": 1},
            engine_toolqa="react",
        ),
        Method(
            label="rerank_bm25",
            display_name="BM25 + Rerank",
            provider="topk",
            provider_args={"source": "rerank_bm25", "k": 1},
            engine_toolqa="react",
        ),
    ],
)

# Retrieval-depth sweep: how much of the BM25 ranking to expose to the
# agent. K ∈ {1, 2, 4, 8} under both skill-exposure modes from the paper:
#   * Full-Skill Injection — full content of all top-K skills prepended
#     to the prompt;
#   * Progressive Disclosure — top-K skills shown as a compact catalog,
#     full content revealed only on explicit load.
#
# Overlap notes:
#   * Full-Skill Injection K=1 (``bm25_top1``) is shared with
#     ``_MAIN.bm25_top1`` and ``_RETRIEVAL.bm25_top1`` — the shared
#     label means the runner reuses any sibling-cell result that
#     already exists.
#   * PD K=1 (``pd_bm25_top1``) has no counterpart in ``_MAIN``: main's
#     Progressive Disclosure shows a BM25 top-50 catalog, whereas this
#     row shows only the top-1 candidate.
_TOPK_SWEEP_INJECTION = [
    Method(
        label=f"bm25_top{k}",
        display_name=f"Full-Skill Injection (K={k})",
        provider="topk",
        provider_args={"source": "bm25", "k": k},
        engine="direct",
        engine_toolqa="react",
    )
    for k in (1, 2, 4, 8)
]

_TOPK_SWEEP_PD = [
    Method(
        label=f"pd_bm25_top{k}",
        display_name=f"Progressive Disclosure (K={k})",
        provider="topk",
        provider_args={"source": "bm25", "k": k},
        engine="progressive_disclosure",
        engine_toolqa="react_progressive_disclosure",
    )
    for k in (1, 2, 4, 8)
]

_TOPK_SWEEP = ExperimentSpec(
    name="topk_sweep",
    description="Retrieval-depth sweep: BM25 top-K skills under both "
                "Full-Skill Injection and Progressive Disclosure "
                "exposure modes (K ∈ {1, 2, 4, 8}). `bm25_top1` is "
                "shared with `main` and `retrieval_comparison`.",
    methods=_TOPK_SWEEP_INJECTION + _TOPK_SWEEP_PD,
)

_TOPK_SWEEP_INJECTION_ONLY = ExperimentSpec(
    name="topk_sweep_injection",
    description="Retrieval-depth sweep, Full-Skill Injection exposure only.",
    methods=_TOPK_SWEEP_INJECTION,
)

_TOPK_SWEEP_PD_ONLY = ExperimentSpec(
    name="topk_sweep_progressive_disclosure",
    description="Retrieval-depth sweep, Progressive Disclosure exposure only.",
    methods=_TOPK_SWEEP_PD,
)

# RQ2 distractor experiment (paper §5.2). Gold skill always included; N
# hard-negative distractors drawn from BM25 / BGE top-50. Two skill-exposure
# modes from the paper figure:
#   * Full Skill Injection — full content of all candidates prepended to
#     the prompt;
#   * Progressive Disclosure — catalog + on-demand loading.
#
# Both sweeps include N ∈ {0, 2, 4, 8} so each experiment is self-contained.
# Overlap notes:
#   * Full Skill Injection N=0 reuses the ``oracle_skill`` label from
#     ``_MAIN`` — when ``n=0`` the ``oracle_distractor`` provider is
#     mathematically identical to the ``oracle`` provider (gold skill
#     only, no distractors), so we point at the same cell to avoid
#     duplicate inference. Display name is still
#     ``Full Skill Injection (N=0)`` so the distractor table reads
#     naturally.
#   * PD N=0 (``pd_oracle_d0``) has no counterpart in ``_MAIN``:
#     main's PD shows a BM25 top-50 catalog, whereas PD N=0 shows only
#     the gold skill.
_DISTRACTOR_INJECTION = [
    # N=0: shared label and provider with ``_MAIN.oracle_skill`` to dedupe.
    Method(
        label="oracle_skill",
        display_name="Full Skill Injection (N=0)",
        provider="oracle",
        engine="direct",
        engine_toolqa="react",
    ),
    *[
        Method(
            label=f"oracle_d{n}",
            display_name=f"Full Skill Injection (N={n})",
            provider="oracle_distractor",
            provider_args={"n": n,
                           "lexical_source": "bm25",
                           "semantic_source": "bge"},
            engine="direct",
            engine_toolqa="react",
        )
        for n in (2, 4, 8)
    ],
]

_DISTRACTOR_PD = [
    Method(
        label=f"pd_oracle_d{n}",
        display_name=f"Progressive Disclosure (N={n})",
        provider="oracle_distractor",
        provider_args={"n": n,
                       "lexical_source": "bm25",
                       "semantic_source": "bge"},
        engine="progressive_disclosure",
        engine_toolqa="react_progressive_disclosure",
    )
    for n in (0, 2, 4, 8)
]

_DISTRACTOR = ExperimentSpec(
    name="distractor",
    description="Noise robustness (paper §5.2 RQ2): oracle + N "
                "hard-negative distractors under Full Skill Injection "
                "and Progressive Disclosure exposure modes.",
    methods=_DISTRACTOR_INJECTION + _DISTRACTOR_PD,
)

_DISTRACTOR_INJECTION_ONLY = ExperimentSpec(
    name="distractor_injection",
    description="Noise robustness, Full Skill Injection exposure only.",
    methods=_DISTRACTOR_INJECTION,
)

_DISTRACTOR_PD_ONLY = ExperimentSpec(
    name="distractor_progressive_disclosure",
    description="Noise robustness, Progressive Disclosure exposure only.",
    methods=_DISTRACTOR_PD,
)


EXPERIMENTS: dict[str, ExperimentSpec] = {
    e.name: e for e in (
        _MAIN, _RETRIEVAL,
        _TOPK_SWEEP, _TOPK_SWEEP_INJECTION_ONLY, _TOPK_SWEEP_PD_ONLY,
        _DISTRACTOR, _DISTRACTOR_INJECTION_ONLY, _DISTRACTOR_PD_ONLY,
    )
}
