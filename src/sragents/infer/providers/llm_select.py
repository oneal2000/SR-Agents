"""LLM-Select provider: show top-N candidates to the LLM and use its
single chosen skill for downstream inference."""

import json
import re
from pathlib import Path

from sragents.corpus import load_corpus_dict
from sragents.infer.base import register_provider
from sragents.llm import chat, create_llm_client, get_extra_body, strip_think_tags
from sragents.prompts import build_prompt

_PROMPT = """\
Given the following problem, select the ONE most relevant skill. \
Output ONLY the skill number.

Problem:
{query}

Skills:
{candidates}

Most relevant skill number:"""


def _format_candidates(candidates: list[dict]) -> str:
    from sragents.corpus import display_name
    lines = []
    for i, s in enumerate(candidates, 1):
        desc = s.get("description", "")
        lines.append(f"[{i}] {display_name(s, i)}: {desc}")
    return "\n".join(lines)


def _parse_first_number(response: str, n: int) -> int | None:
    response = strip_think_tags(response)
    for x in re.findall(r"\d+", response):
        m = int(x)
        if 1 <= m <= n:
            return m - 1
    return None


@register_provider("llm_select")
class LLMSelectProvider:
    """Show top-N candidates to the LLM; use its pick as the single skill.

    On parse failure retries up to ``max_retries`` times, then falls back
    to candidate rank 1.

    Args:
        source: Retrieval results file (e.g. from ``sragents retrieve``).
        pool: Number of top candidates shown to the LLM.
        model: The LLM used to make the pick.
        api_base: OpenAI-compatible endpoint.
        corpus_path: Optional corpus override.
        max_retries: Max LLM retries on parse failure.
    """

    def __init__(
        self,
        source: str,
        model: str,
        api_base: str | None = None,
        pool: int = 50,
        corpus_path: str | None = None,
        max_retries: int = 3,
    ):
        self._pool = int(pool)
        self._model = model
        self._client = create_llm_client(api_base=api_base)
        self._extra_body = get_extra_body(model, thinking=False)
        self._max_retries = max_retries
        self._corpus = (
            load_corpus_dict(corpus_path) if corpus_path else load_corpus_dict()
        )
        src = Path(source)
        if not src.exists():
            raise FileNotFoundError(
                f"Retrieval source file not found: {src}. "
                "Run `sragents retrieve` to produce it first."
            )
        data = json.loads(src.read_text())
        self._lookup = {r["instance_id"]: r["retrieved"] for r in data["results"]}

    def provide(self, instance: dict) -> list[dict]:
        retrieved = self._lookup.get(instance["instance_id"], [])[: self._pool]
        candidates = [
            self._corpus[r["skill_id"]]
            for r in retrieved
            if r["skill_id"] in self._corpus
        ]
        if not candidates:
            return []
        if len(candidates) == 1:
            return candidates

        _, query = build_prompt(instance)
        prompt = _PROMPT.format(query=query, candidates=_format_candidates(candidates))

        for _ in range(self._max_retries):
            response = chat(
                self._client, self._model, prompt,
                temperature=0.0, max_tokens=64,
                extra_body=self._extra_body,
            )
            idx = _parse_first_number(response, len(candidates))
            if idx is not None:
                return [candidates[idx]]

        return [candidates[0]]  # fallback: rank 1
