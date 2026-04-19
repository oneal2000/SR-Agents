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
            display_name="BM25 Top-1",
            provider="topk",
            provider_args={"source": "bm25", "k": 1},
            engine="direct",
            engine_toolqa="react",
        ),
        Method(
            label="bm25_select",
            display_name="BM25 Select",
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

# Context skill-count sweep: BM25 top-K skills prepended to the prompt.
# K ∈ {1, 2, 4, 8}; K=1 is shared with ``_MAIN.bm25_top1`` and
# ``_RETRIEVAL.bm25_top1`` — kept here for self-containment, and the shared
# label means the runner reuses any sibling-cell result that already exists.
_TOPK_SWEEP = ExperimentSpec(
    name="topk_sweep",
    description="Context skill-count sweep: BM25 top-K skills in context "
                "(K ∈ {1, 2, 4, 8}). `bm25_top1` is shared with `main` and "
                "`retrieval_comparison`.",
    methods=[
        Method(
            label=f"bm25_top{k}",
            display_name=f"BM25 Top-{k}",
            provider="topk",
            provider_args={"source": "bm25", "k": k},
            engine="direct",
            engine_toolqa="react",
        )
        for k in (1, 2, 4, 8)
    ],
)

# RQ2 distractor experiment (paper §4.2). Gold skill always included; N
# hard-negative distractors drawn from BM25 / BGE top-50. Two skill-exposure
# modes from the paper figure:
#   * Context — full content of all candidates prepended to the prompt;
#   * Progressive Disclosure — catalog + on-demand loading.
#
# Both sweeps include N ∈ {0, 2, 4, 8} so each experiment is self-contained.
# Overlap note:
#   * Context N=0 (``oracle_d0``) produces the same prompt as
#     ``_MAIN.oracle_skill`` (gold skill only, no distractors), but has
#     a different label so it runs separately.
#   * PD N=0 (``pd_oracle_d0``) has no counterpart in ``_MAIN``:
#     main's PD shows a BM25 top-50 catalog, whereas PD N=0 shows only
#     the gold skill.
_DISTRACTOR_CONTEXT = [
    Method(
        label=f"oracle_d{n}",
        display_name=f"Context (N={n})",
        provider="oracle_distractor",
        provider_args={"n": n,
                       "lexical_source": "bm25",
                       "semantic_source": "bge"},
        engine="direct",
        engine_toolqa="react",
    )
    for n in (0, 2, 4, 8)
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
    description="Noise robustness (paper §4.2 RQ2): oracle + N "
                "hard-negative distractors under Context and "
                "Progressive Disclosure exposure modes.",
    methods=_DISTRACTOR_CONTEXT + _DISTRACTOR_PD,
)

_DISTRACTOR_CTX = ExperimentSpec(
    name="distractor_context",
    description="Noise robustness, Context exposure only.",
    methods=_DISTRACTOR_CONTEXT,
)

_DISTRACTOR_PD_ONLY = ExperimentSpec(
    name="distractor_progressive_disclosure",
    description="Noise robustness, Progressive Disclosure exposure only.",
    methods=_DISTRACTOR_PD,
)


EXPERIMENTS: dict[str, ExperimentSpec] = {
    e.name: e for e in (
        _MAIN, _RETRIEVAL, _TOPK_SWEEP,
        _DISTRACTOR, _DISTRACTOR_CTX, _DISTRACTOR_PD_ONLY,
    )
}
