"""TF-IDF cosine-similarity retriever using scipy sparse matrices."""

import time

import numpy as np
from scipy import sparse

from sragents.retrieve._sparse_core import tokenize
from sragents.retrieve.base import register


@register("tfidf")
class TfidfRetriever:
    """TF-IDF with cosine similarity."""

    def build_index(self, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        self._corpus_ids = corpus_ids

        print("  Building vocabulary...", end=" ", flush=True)
        t0 = time.time()
        vocab: dict[str, int] = {}
        tokenized = []
        for text in corpus_texts:
            tokens = tokenize(text)
            tokenized.append(tokens)
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        print(f"{time.time() - t0:.1f}s ({len(vocab)} terms)")

        n_docs = len(corpus_texts)
        n_terms = len(vocab)
        self._vocab = vocab

        print("  Building TF-IDF matrix...", end=" ", flush=True)
        t0 = time.time()
        rows, cols, vals = [], [], []
        for i, tokens in enumerate(tokenized):
            if not tokens:
                continue
            token_counts: dict[int, int] = {}
            for t in tokens:
                tid = vocab[t]
                token_counts[tid] = token_counts.get(tid, 0) + 1
            doc_len = len(tokens)
            for tid, count in token_counts.items():
                rows.append(i)
                cols.append(tid)
                vals.append(count / doc_len)

        tf = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n_docs, n_terms), dtype=np.float32
        )

        df = np.array((tf > 0).sum(axis=0), dtype=np.float32).flatten()
        idf = np.log(n_docs / (1.0 + df))

        tfidf = tf.multiply(idf[np.newaxis, :])
        norms = sparse.linalg.norm(tfidf, axis=1)
        norms[norms == 0] = 1.0
        self._matrix = tfidf.multiply(1.0 / norms[:, np.newaxis]).tocsr()
        self._idf = idf
        print(f"{time.time() - t0:.1f}s")

    def retrieve(
        self, queries: list[str], top_k: int = 10
    ) -> list[list[tuple[str, float]]]:
        print("  Encoding queries...", end=" ", flush=True)
        t0 = time.time()
        vocab = self._vocab
        idf = self._idf
        q_rows, q_cols, q_vals = [], [], []
        for i, query in enumerate(queries):
            tokens = tokenize(query)
            if not tokens:
                continue
            token_counts: dict[int, int] = {}
            for t in tokens:
                if t in vocab:
                    tid = vocab[t]
                    token_counts[tid] = token_counts.get(tid, 0) + 1
            in_vocab_len = sum(token_counts.values())
            if in_vocab_len == 0:
                continue
            for tid, count in token_counts.items():
                q_rows.append(i)
                q_cols.append(tid)
                q_vals.append(count / in_vocab_len)

        q_mat = sparse.csr_matrix(
            (q_vals, (q_rows, q_cols)),
            shape=(len(queries), len(vocab)),
            dtype=np.float32,
        )
        q_mat = q_mat.multiply(idf[np.newaxis, :])
        q_norms = sparse.linalg.norm(q_mat, axis=1)
        q_norms[q_norms == 0] = 1.0
        q_mat = q_mat.multiply(1.0 / q_norms[:, np.newaxis]).tocsr()
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
