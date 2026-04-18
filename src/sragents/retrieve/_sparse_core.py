"""Shared helpers for sparse retrievers (BM25 / TF-IDF)."""

import re


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()
