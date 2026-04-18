"""LLM-based listwise reranker.

Takes a pre-ranked list of skill candidates and returns a new list
ordered by the LLM's relevance judgments. Consumed and produced as
ordinary retrieval-stage artifacts.
"""

import re

from sragents.llm import chat, strip_think_tags

_RERANK_PROMPT = """\
Given the following problem, rank the skills below by relevance. \
Output ONLY the skill numbers in order from most to least relevant, \
separated by commas.

Problem:
{query}

Skills:
{candidates}

Most relevant first (numbers only):"""


def _format_candidates(candidates: list[dict]) -> str:
    from sragents.corpus import display_name
    lines = []
    for i, skill in enumerate(candidates, 1):
        desc = skill.get("description", "")
        lines.append(f"[{i}] {display_name(skill, i)}: {desc}")
    return "\n".join(lines)


def _parse_ranking(response: str, n_candidates: int) -> list[int]:
    """Return 0-based indices in the order the LLM listed them."""
    response = strip_think_tags(response)
    seen: set[int] = set()
    indices: list[int] = []
    for x in re.findall(r"\d+", response):
        n = int(x)
        if 1 <= n <= n_candidates and n not in seen:
            seen.add(n)
            indices.append(n - 1)
    return indices


class LLMReranker:
    """Listwise reranker: ask the LLM to order candidates by relevance.

    Retries on poor parses (fewer than half the candidates recovered) and
    falls back to the source order for any candidates the model omitted.
    """

    def __init__(
        self,
        client,
        model: str,
        extra_body: dict | None = None,
        max_retries: int = 3,
    ):
        self._client = client
        self._model = model
        self._extra_body = extra_body
        self._max_retries = max_retries

    def rerank(
        self,
        query: str,
        candidates: list[dict],
    ) -> list[tuple[str, float]]:
        """Return ``[(skill_id, score), ...]`` covering all candidates."""
        if not candidates:
            return []
        if len(candidates) == 1:
            return [(candidates[0]["skill_id"], 1.0)]

        candidate_text = _format_candidates(candidates)
        prompt = _RERANK_PROMPT.format(query=query, candidates=candidate_text)

        best_indices: list[int] = []
        for _ in range(self._max_retries):
            response = chat(
                self._client, self._model, prompt,
                temperature=0.0, max_tokens=4096,
                extra_body=self._extra_body,
            )
            indices = _parse_ranking(response, len(candidates))
            if len(indices) > len(best_indices):
                best_indices = indices
            if len(indices) >= len(candidates) // 2:
                break

        results: list[tuple[str, float]] = []
        for rank, idx in enumerate(best_indices, 1):
            results.append((candidates[idx]["skill_id"], 1.0 / rank))

        ranked = set(best_indices)
        next_rank = len(results) + 1
        for idx, c in enumerate(candidates):
            if idx not in ranked:
                results.append((c["skill_id"], 1.0 / next_rank))
                next_rank += 1

        return results
