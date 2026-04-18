"""Retriever contract and registry.

A :class:`Retriever` indexes the skill library and returns ranked
candidates for each query. Implementations register themselves via :func:`register`
and become available through :func:`get` and on the ``sragents retrieve``
CLI.

Adding a new retriever:

.. code-block:: python

    from sragents.retrieve.base import register

    @register("my_retriever")
    class MyRetriever:
        def build_index(self, ids, texts): ...
        def retrieve(self, queries, top_k): ...

External plugins are loaded via ``sragents --plugin my_pkg.my_module`` or
declared as ``[project.entry-points."sragents.retrievers"]`` in a
``pyproject.toml``.
"""

from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class Retriever(Protocol):
    """Indexes a corpus and ranks skills for each query.

    Lifecycle: :meth:`build_index` is called exactly once before any call
    to :meth:`retrieve`. The same ``Retriever`` instance then handles all
    queries. Calling :meth:`build_index` a second time replaces the index.
    """

    def build_index(
        self, corpus_ids: list[str], corpus_texts: list[str]
    ) -> None:
        """Build the retriever's internal index over the corpus.

        ``corpus_ids[i]`` corresponds to ``corpus_texts[i]``. ``corpus_ids``
        are the ``skill_id`` strings returned verbatim by :meth:`retrieve`.
        """

    def retrieve(
        self, queries: list[str], top_k: int
    ) -> list[list[tuple[str, float]]]:
        """Rank skills for each query.

        Returns a list of length ``len(queries)``. Each inner list contains
        up to ``top_k`` ``(skill_id, score)`` pairs **sorted by descending
        score** (most relevant first). Score semantics are retriever-specific
        (cosine, BM25, etc.); only the relative ordering is consumed
        downstream.
        """


_REGISTRY: dict[str, Callable[..., Retriever]] = {}


def register(name: str):
    """Decorator: register a :class:`Retriever` factory under ``name``."""
    def wrap(cls_or_factory):
        _REGISTRY[name] = cls_or_factory
        return cls_or_factory
    return wrap


def get(name: str, **kwargs) -> Retriever:
    """Instantiate the retriever registered under ``name``."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown retriever {name!r}. Available: {list_retrievers()}"
        )
    return _REGISTRY[name](**kwargs)


def list_retrievers() -> list[str]:
    return sorted(_REGISTRY)
