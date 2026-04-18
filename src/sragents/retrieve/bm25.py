"""BM25 Okapi retriever using scipy sparse matrices."""

import time

import numpy as np
from scipy import sparse

from sragents.retrieve._sparse_core import tokenize
from sragents.retrieve.base import register


@register("bm25")
class BM25Retriever:
    """BM25 Okapi."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def build_index(self, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        self._corpus_ids = corpus_ids

        print("  Building vocabulary...", end=" ", flush=True)
        t0 = time.time()
        vocab: dict[str, int] = {}
        tokenized: list[list[str]] = []
        doc_lens: list[int] = []
        for text in corpus_texts:
            tokens = tokenize(text)
            tokenized.append(tokens)
            doc_lens.append(len(tokens))
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        avgdl = np.mean(doc_lens) if doc_lens else 1.0
        print(f"{time.time() - t0:.1f}s ({len(vocab)} terms)")

        n_docs = len(corpus_texts)
        n_terms = len(vocab)
        self._vocab = vocab

        print("  Building BM25 matrix...", end=" ", flush=True)
        t0 = time.time()
        rows, cols, vals = [], [], []
        for i, tokens in enumerate(tokenized):
            if not tokens:
                continue
            token_counts: dict[int, int] = {}
            for t in tokens:
                tid = vocab[t]
                token_counts[tid] = token_counts.get(tid, 0) + 1
            for tid, count in token_counts.items():
                rows.append(i)
                cols.append(tid)
                vals.append(count)

        tf = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n_docs, n_terms), dtype=np.float32
        )

        k1, b = self.k1, self.b
        dl = np.array(doc_lens, dtype=np.float32)
        denom_base = k1 * (1.0 - b + b * dl / avgdl)

        tf_coo = tf.tocoo()
        sat_vals = (tf_coo.data * (k1 + 1)) / (tf_coo.data + denom_base[tf_coo.row])
        bm25_tf = sparse.csr_matrix(
            (sat_vals, (tf_coo.row, tf_coo.col)),
            shape=(n_docs, n_terms),
            dtype=np.float32,
        )

        df = np.array((tf > 0).sum(axis=0), dtype=np.float32).flatten()
        # Lucene / ElasticSearch BM25 variant: clip IDF at zero so
        # extremely common terms contribute zero weight.
        idf = np.log((n_docs - df + 0.5) / (df + 0.5)).clip(min=0)

        self._matrix = bm25_tf.multiply(idf[np.newaxis, :]).tocsr()
        print(f"{time.time() - t0:.1f}s")

    def retrieve(
        self, queries: list[str], top_k: int = 10
    ) -> list[list[tuple[str, float]]]:
        print("  Encoding queries...", end=" ", flush=True)
        t0 = time.time()
        vocab = self._vocab
        q_rows, q_cols, q_vals = [], [], []
        for i, query in enumerate(queries):
            tokens = tokenize(query)
            seen: set[int] = set()
            for t in tokens:
                if t in vocab:
                    tid = vocab[t]
                    if tid not in seen:
                        seen.add(tid)
                        q_rows.append(i)
                        q_cols.append(tid)
                        q_vals.append(1.0)

        q_mat = sparse.csr_matrix(
            (q_vals, (q_rows, q_cols)),
            shape=(len(queries), len(vocab)),
            dtype=np.float32,
        )
        print(f"{time.time() - t0:.1f}s")

        print("  Scoring...", end=" ", flush=True)
        t0 = time.time()
        score_mat = q_mat.dot(self._matrix.T).toarray()
        print(f"{time.time() - t0:.1f}s")

        results = []
        for i in range(len(queries)):
            top_indices = np.argsort(score_mat[i])[::-1][:top_k]
            results.append([
                (self._corpus_ids[idx], float(score_mat[i][idx]))
                for idx in top_indices
            ])
        return results
