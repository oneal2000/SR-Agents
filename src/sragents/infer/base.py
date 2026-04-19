"""Inference contracts and registries.

An inference run is ``SkillProvider`` × ``InferenceEngine``:

* A :class:`SkillProvider` supplies the **candidate skills** for an
  instance. It can return one skill already chosen (``oracle``,
  ``topk(k=1)``, ``llm_select``) or a larger pool for the engine to
  narrow further (``topk(k=50)``, ``oracle_distractor``). Providers
  may call the LLM — ``llm_select`` does — but only to filter/rank
  candidates, never to solve the task.
* An :class:`InferenceEngine` consumes those candidates and produces
  the task output. Simple engines use them as-is (``direct`` prepends
  every candidate); agentic engines (``progressive_disclosure``,
  ``react``, ``react_progressive_disclosure``) interleave further
  candidate selection with solving — the model sees a catalog and
  loads skills on demand inside its reasoning loop. Engines own the
  task-solving LLM call(s).

The paper's three stages — retrieval, incorporation, application —
map cleanly onto ``retrieve / Provider / Engine`` for non-agentic
methods. Agentic engines fold the incorporation decision into the
application loop, so the Provider/Engine boundary is pragmatic rather
than a strict Stage 2 / Stage 3 split.

This split lets you add new methods by writing one small module.

Adding a new component::

    from sragents.infer.base import register_provider, register_engine

    @register_provider("my_provider")
    class MyProvider:
        def __init__(self, **cfg): ...
        def provide(self, instance): ...

    @register_engine("my_engine")
    class MyEngine:
        def run(self, instance, skills, client, model, **cfg): ...
"""

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable


@dataclass
class InferenceResult:
    """Output of one inference run.

    Attributes:
        raw_output: Model-generated text only. This is the evaluator input;
            system-injected content (loaded skill bodies, tool results)
            must NOT appear here.
        transcript: Optional full trace including system-injected content
            (for agent modes). Evaluators never read this.
        skill_ids_used: IDs of skills actually used by the model.
        meta: Engine-specific metadata (e.g. ``n_steps`` for ReAct).
    """
    raw_output: str
    transcript: str | None = None
    skill_ids_used: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@runtime_checkable
class SkillProvider(Protocol):
    """Selects skills for an instance.

    Providers are instantiated once per CLI invocation and reused across
    all instances. Heavy one-time setup (loading a retrieval-results
    file, loading the corpus, creating an LLM client) belongs in
    ``__init__``. ``provide`` should be thread-safe — the runner calls
    it from a thread pool.

    Constructor ``**kwargs`` are forwarded from ``--provider-arg KEY=VALUE``
    on the CLI (JSON-parsed when possible, so numbers and bools come
    through typed). A provider that wants the LLM ``model`` / ``api_base``
    simply declares them as ``__init__`` parameters; :mod:`sragents.cli.infer`
    auto-forwards them.
    """

    def provide(self, instance: dict) -> list[dict]:
        """Return the skill dicts to use for this instance.

        Each skill dict must have at minimum ``skill_id`` and ``content``
        (plus ``name`` / ``description`` if the engine may present them
        to the model). Order is significant: engines present skills in
        the order returned here. May be empty.
        """


@runtime_checkable
class InferenceEngine(Protocol):
    """Runs one inference call given an instance and its skills.

    Engines encapsulate *how* the LLM is called: single chat, multi-turn
    agent loop, ReAct, critique-then-revise, etc.
    """

    def run(
        self,
        instance: dict,
        skills: list[dict],
        client,
        model: str,
        **kwargs,
    ) -> InferenceResult:
        """Execute one inference run.

        Args:
            instance: Bench instance dict (``instance_id``, ``dataset``,
                ``question``, ``eval_data``, ``skill_annotations``).
            skills: The list returned by the :class:`SkillProvider` for
                this instance (may be empty).
            client: OpenAI-compatible client, shared across threads.
            model: Model name / path passed to the endpoint.
            **kwargs: Reserved for future orthogonal flags; the CLI
                auto-forwards ``temperature``, ``max_tokens``, and
                ``thinking`` when the engine's ``__init__`` declares them.

        Returns:
            :class:`InferenceResult`. Crucially, ``raw_output`` must
            contain **only model-generated tokens** — system-injected
            content (loaded skill bodies, tool results, observation
            framing) belongs in ``transcript`` and is never read by
            evaluators.
        """
        ...


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, Callable[..., SkillProvider]] = {}
_ENGINES: dict[str, Callable[..., InferenceEngine]] = {}


def register_provider(name: str):
    def wrap(cls_or_factory):
        _PROVIDERS[name] = cls_or_factory
        return cls_or_factory
    return wrap


def register_engine(name: str):
    def wrap(cls_or_factory):
        _ENGINES[name] = cls_or_factory
        return cls_or_factory
    return wrap


def get_provider(name: str, **kwargs) -> SkillProvider:
    if name not in _PROVIDERS:
        raise KeyError(
            f"Unknown provider {name!r}. Available: {list_providers()}"
        )
    return _PROVIDERS[name](**kwargs)


def get_engine(name: str, **kwargs) -> InferenceEngine:
    if name not in _ENGINES:
        raise KeyError(
            f"Unknown engine {name!r}. Available: {list_engines()}"
        )
    return _ENGINES[name](**kwargs)


def list_providers() -> list[str]:
    return sorted(_PROVIDERS)


def list_engines() -> list[str]:
    return sorted(_ENGINES)
