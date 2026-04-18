"""Oracle provider: use ground-truth skill annotations.

Looks up ``instance['skill_annotations']`` in the corpus. This is the
upper bound — how well a model does when given the right skills.
"""

from sragents.corpus import load_corpus_dict
from sragents.infer.base import register_provider


@register_provider("oracle")
class OracleProvider:
    def __init__(self, corpus_path: str | None = None):
        self._corpus = load_corpus_dict(corpus_path) if corpus_path else load_corpus_dict()

    def provide(self, instance: dict) -> list[dict]:
        return [
            self._corpus[sid]
            for sid in instance.get("skill_annotations", [])
            if sid in self._corpus
        ]
