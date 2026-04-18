"""Dense retrieval using sentence-transformer models (BGE, Contriever, ...).

Two factories are registered:

* ``bge`` → ``BAAI/bge-base-en-v1.5`` with the BGE query prefix.
* ``contriever`` → ``facebook/contriever-msmarco``.

Any other HuggingFace model can be used by calling :class:`DenseRetriever`
directly::

    DenseRetriever(model_name_or_path="intfloat/e5-base-v2",
                   query_prefix="query: ")
"""

import time

import numpy as np

from sragents.retrieve.base import register


class DenseRetriever:
    """Dense retriever over any ``sentence-transformers``-compatible model."""

    def __init__(
        self,
        model_name_or_path: str,
        query_prefix: str = "",
        batch_size: int = 256,
    ):
        self._model_path = model_name_or_path
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading model: {self._model_path}")
            self._model = SentenceTransformer(self._model_path)

    def build_index(self, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        """Encode the corpus once. ``query_prefix`` is **not** applied to
        documents — it is a model-side convention only applied at query
        time (see :meth:`retrieve`)."""
        self._corpus_ids = corpus_ids
        self._load_model()

        print(f"  Encoding corpus ({len(corpus_texts)} docs)...", end=" ", flush=True)
        t0 = time.time()
        self._corpus_emb = self._model.encode(
            corpus_texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        print(f"{time.time() - t0:.1f}s")

    def retrieve(
        self, queries: list[str], top_k: int = 10
    ) -> list[list[tuple[str, float]]]:
        """Encode queries (prepending ``query_prefix`` if set), compute
        cosine similarity against the indexed corpus, return top-K per query
        sorted by descending score."""
        self._load_model()
        query_texts = [self._query_prefix + q for q in queries]

        print(f"  Encoding queries ({len(query_texts)})...", end=" ", flush=True)
        t0 = time.time()
        query_emb = self._model.encode(
            query_texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        print(f"{time.time() - t0:.1f}s")

        print("  Scoring...", end=" ", flush=True)
        t0 = time.time()
        scores = query_emb @ self._corpus_emb.T
        print(f"{time.time() - t0:.1f}s")

        results = []
        for i in range(len(queries)):
            top_indices = np.argsort(scores[i])[::-1][:top_k]
            results.append([
                (self._corpus_ids[idx], float(scores[i][idx]))
                for idx in top_indices
            ])
        return results


@register("bge")
def _bge_factory(
    model_path: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 256,
) -> DenseRetriever:
    return DenseRetriever(
        model_name_or_path=model_path,
        query_prefix="Represent this sentence for searching relevant passages: ",
        batch_size=batch_size,
    )


@register("contriever")
def _contriever_factory(
    model_path: str = "facebook/contriever-msmarco",
    batch_size: int = 256,
) -> DenseRetriever:
    return DenseRetriever(
        model_name_or_path=model_path,
        query_prefix="",
        batch_size=batch_size,
    )
