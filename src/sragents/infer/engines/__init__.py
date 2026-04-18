"""Inference engines: how skills are handed to the model.

Built-in engines:

* ``direct`` — single chat call (tool loop auto-enabled if skills expose
  a ``tools`` field). Suitable for all non-ToolQA datasets.
* ``progressive_disclosure`` — multi-round agent loop exposing
  ``LOAD_SKILL`` / ``TOOL_CALL``. The model first sees only a compact
  skill catalog and chooses which skills to load on demand.
* ``react`` — ReAct loop (one LLM call per step). For ToolQA with
  skills injected into the system prompt.
* ``react_progressive_disclosure`` — ReAct + ``LoadSkill[index]``
  action. For ToolQA with candidate skills the model can load
  mid-trajectory.

The experiment runner picks the right engine for each dataset
automatically. Direct CLI users choose explicitly.
"""

from sragents.infer.engines import (  # noqa: F401
    direct,
    progressive_disclosure,
    react,  # registers both `react` and `react_progressive_disclosure`
)
