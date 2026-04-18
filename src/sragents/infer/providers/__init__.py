"""Skill providers: where inference-time skills come from.

Built-in providers:

* ``none`` — no skills (skill-free baseline).
* ``oracle`` — ground-truth skills from ``instance['skill_annotations']``.
* ``topk`` — top-K from a retrieval results file.
* ``llm_select`` — show top-N candidates to the LLM, use its pick.
* ``oracle_distractor`` — oracle + N hard-negative distractors.

The provider decides **which** skills an instance receives; the engine
decides **how** they are handed to the model.
"""

from sragents.infer.providers import (  # noqa: F401
    distractor,
    llm_select,
    none,
    oracle,
    topk,
)
