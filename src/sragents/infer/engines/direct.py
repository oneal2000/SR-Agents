"""Direct engine: single-shot chat, with optional skill-provided tool loop.

If any of the supplied skills expose a ``tools`` field, the engine enters
the tool-call loop defined in :mod:`sragents.infer.engines.tool_loop`.
Otherwise it makes one chat call and returns its text.

The skills passed in are concatenated and injected into the user prompt by
the prompt builder (see :mod:`sragents.prompts`).
"""

from sragents.infer.base import InferenceResult, register_engine
from sragents.infer.engines.tool_loop import run_with_tools
from sragents.llm import chat, get_extra_body
from sragents.prompts import build_prompt


@register_engine("direct")
class DirectEngine:
    """Single LLM call per instance (+ optional tool loop)."""

    def __init__(
        self,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        thinking: bool = False,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking

    def run(
        self,
        instance: dict,
        skills: list[dict],
        client,
        model: str,
        **kwargs,
    ) -> InferenceResult:
        skill_texts = [s["content"] for s in skills if s.get("content")]
        system, user = build_prompt(instance, skills=skill_texts)

        extra = get_extra_body(model, thinking=self.thinking)

        tools = [t for s in skills for t in s.get("tools", [])]

        if tools:
            model_output, transcript = run_with_tools(
                client, model, system, user, tools,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body=extra,
            )
        else:
            model_output = chat(
                client, model, user, system=system,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body=extra,
            )
            transcript = None

        return InferenceResult(
            raw_output=model_output,
            transcript=transcript,
            skill_ids_used=[s["skill_id"] for s in skills],
        )
