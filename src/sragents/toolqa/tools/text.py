"""Text retrieval tools for ToolQA: RetrieveAgenda, RetrieveScirex.

Uses sentence-transformers embeddings + numpy cosine similarity.
Embedding model and corpus are lazy-loaded on the first query.

Thread safety: ``TextRetriever`` instances are shared across threads
via ``get_shared_retriever()``. ``_ensure_index()`` uses double-checked
locking to prevent concurrent initialization; ``query()`` is
thread-safe (numpy matmul + ``SentenceTransformer.encode`` produce
new tensors).
"""

import json
import threading
from pathlib import Path

import numpy as np

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Process-level cache: (corpus_path, text_field) -> TextRetriever (shared)
_retriever_cache: dict[tuple[str, str], "TextRetriever"] = {}
_retriever_cache_lock = threading.Lock()


def get_shared_retriever(
    corpus_path: Path,
    text_field: str,
    model_name: str = EMBED_MODEL_NAME,
    top_k: int = 3,
) -> "TextRetriever":
    """Return a shared TextRetriever, creating it on first call (thread-safe)."""
    key = (str(corpus_path), text_field)
    if key in _retriever_cache:
        return _retriever_cache[key]
    with _retriever_cache_lock:
        if key not in _retriever_cache:
            retriever = TextRetriever(corpus_path, text_field, model_name, top_k)
            retriever._ensure_index()
            _retriever_cache[key] = retriever
    return _retriever_cache[key]


class TextRetriever:
    """Semantic text retriever using sentence-transformers + cosine similarity."""

    def __init__(
        self,
        corpus_path: Path,
        text_field: str,
        model_name: str = EMBED_MODEL_NAME,
        top_k: int = 3,
    ):
        self.corpus_path = Path(corpus_path)
        self.text_field = text_field
        self.model_name = model_name
        self.top_k = top_k

        self._model = None
        self._texts: list[str] | None = None
        self._embeddings: np.ndarray | None = None
        self._init_lock = threading.Lock()

    def _ensure_index(self):
        """Lazy-load corpus and build embedding index on first use.

        Uses double-checked locking so concurrent threads don't duplicate work.
        """
        if self._embeddings is not None:
            return

        with self._init_lock:
            if self._embeddings is not None:
                return

            # Load corpus texts
            texts = []
            with open(self.corpus_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    texts.append(item[self.text_field])
            self._texts = texts

            # Load embedding model
            import sentence_transformers
            self._model = sentence_transformers.SentenceTransformer(self.model_name)

            # Encode all documents
            print(f"  Encoding {len(texts)} documents with {self.model_name}...")
            self._embeddings = self._model.encode(texts, show_progress_bar=True)
            # Normalize for cosine similarity
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._embeddings = self._embeddings / norms

    def query(self, query_text: str, top_k: int | None = None) -> str:
        """Return top-k most relevant documents as newline-separated text."""
        self._ensure_index()
        k = top_k or self.top_k

        # Encode query
        query_emb = self._model.encode([query_text])
        query_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        query_norm = np.where(query_norm == 0, 1, query_norm)
        query_emb = query_emb / query_norm

        # Cosine similarity
        scores = (self._embeddings @ query_emb.T).squeeze()
        top_indices = np.argsort(scores)[-k:][::-1]

        results = [self._texts[i] for i in top_indices]
        return "\n".join(results)
