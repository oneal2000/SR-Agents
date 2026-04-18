"""Progressive Disclosure engine: agent-style skill discovery and loading.

The model first sees only a compact catalog of candidate skills (index,
name, one-line description). It loads the full content of a skill on
demand by emitting ``LOAD_SKILL: <index>`` on its own line. Tools
provided by a loaded skill can then be invoked via
``TOOL_CALL: fn(args)``.

Returns an :class:`InferenceResult` with ``raw_output`` (model-generated
tokens only) and ``transcript`` (full conversation including injected
skill bodies).
"""

import re

from sragents.corpus import display_name, load_corpus_dict
from sragents.infer.base import InferenceResult, register_engine
from sragents.infer.engines.tool_loop import execute_tool, parse_tool_call
from sragents.llm import chat_messages, get_extra_body, strip_think_tags
from sragents.prompts import build_prompt

_LOAD_SKILL_RE = re.compile(
    r"^LOAD_SKILL:\s*\[?([A-Za-z0-9_]+)\]?", re.MULTILINE
)

# Markers that only appear on injected user messages. If the model
# generates one itself, the response is truncated before it so no
# synthetic content leaks into ``raw_output``.
_SELF_INJECT_RE = re.compile(
    r"\n\s*(?:Skill loaded|TOOL_RESULT)\s*:", re.IGNORECASE
)

_MAX_ROUNDS = 10

_INSTRUCTIONS = """\
You have access to a skill library. Each skill provides precise \
methodology and step-by-step procedures for a specific problem type \
— these often contain critical details that general knowledge may miss.

To use a skill, write on its own line:
LOAD_SKILL: <index>

For example: LOAD_SKILL: 0

You will receive the skill's full content and can then apply \
the methodology to solve the problem.

Available skills:
{skill_list}"""


def build_system_prompt(
    candidates: list[dict],
    base_system: str = "",
) -> tuple[str, dict[str, str]]:
    """Prepend the skill-list block to the dataset's base system prompt."""
    lines = []
    idx_map: dict[str, str] = {}
    for i, s in enumerate(candidates):
        desc = s.get("description", "")
        lines.append(f"{i} — {display_name(s, i)} — {desc}")
        idx_map[str(i)] = s["skill_id"]
    block = _INSTRUCTIONS.format(skill_list="\n".join(lines))
    return (f"{base_system}\n\n{block}" if base_system else block), idx_map


def _handle_load_skill(
    token: str,
    corpus: dict[str, dict],
    idx_map: dict[str, str],
    loaded_skill_ids: list[str],
    available_tools: dict[str, dict],
) -> str:
    """Build the ``user`` message that injects a loaded skill's body.

    Resolution order (all three scoped to the candidate set shown to the
    model):

    1. Numeric index from the catalog.
    2. Direct ``skill_id`` match.
    3. Case-insensitive name match.
    """
    skill: dict | None = None
    # 1. Numeric index.
    real_id = idx_map.get(token)
    if real_id:
        skill = corpus.get(real_id)

    candidates = {sid: corpus[sid] for sid in idx_map.values() if sid in corpus}

    # 2. Direct skill_id lookup.
    if skill is None and token in candidates:
        skill = candidates[token]

    # 3. Case-insensitive name match.
    if skill is None:
        token_lower = token.lower()
        for s in candidates.values():
            if s.get("name", "").lower() == token_lower:
                skill = s
                break

    if skill is None:
        return f"\nSkill '{token}' not found. Continue solving the problem."

    loaded_skill_ids.append(skill["skill_id"])
    if "tools" in skill:
        for t in skill["tools"]:
            available_tools[t["name"]] = t

    return (
        f"\nSkill loaded: {display_name(skill)}\n"
        f"---\n{skill.get('content', '')}\n---\n"
        f"\nContinue solving the problem."
    )


@register_engine("progressive_disclosure")
class ProgressiveDisclosureEngine:
    """Agent-style engine: the model loads skills on demand during one run."""

    def __init__(
        self,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        thinking: bool = False,
        max_rounds: int = _MAX_ROUNDS,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.max_rounds = max_rounds

    def run(
        self,
        instance: dict,
        skills: list[dict],
        client,
        model: str,
        **kwargs,
    ) -> InferenceResult:
        """``skills`` here are **candidates** — the full set the model can load."""
        corpus = kwargs.get("corpus") or load_corpus_dict()

        base_system, user = build_prompt(instance)
        system, idx_map = build_system_prompt(skills, base_system=base_system)
        extra = get_extra_body(model, thinking=self.thinking)

        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        model_output = ""
        transcript = ""
        loaded_skill_ids: list[str] = []
        available_tools: dict[str, dict] = {}

        for _ in range(self.max_rounds):
            raw = chat_messages(
                client, model, messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body=extra,
            )
            if not raw:
                break

            response = strip_think_tags(raw)
            # Strip any self-generated system framing.
            m = _SELF_INJECT_RE.search(response)
            if m:
                response = response[: m.start()].rstrip()

            # Priority 1: LOAD_SKILL
            load_match = _LOAD_SKILL_RE.search(response)
            if load_match:
                head = response[: load_match.end()]
                model_output += head
                transcript += head

                inject = _handle_load_skill(
                    load_match.group(1), corpus, idx_map,
                    loaded_skill_ids, available_tools,
                )
                transcript += inject + "\n"

                messages.append({"role": "assistant", "content": head})
                messages.append({"role": "user", "content": inject.strip()})
                continue

            # Priority 2: TOOL_CALL
            if available_tools:
                parsed = parse_tool_call(response, available_tools)
                if parsed is not None:
                    head, tool_name, args = parsed
                    try:
                        result = execute_tool(available_tools[tool_name], args)
                    except Exception as e:  # noqa: BLE001
                        result = f"Error: {e}"
                    inject = f"TOOL_RESULT: {result}"
                    model_output += head
                    transcript += head + f"\n{inject}\n"

                    messages.append({"role": "assistant", "content": head})
                    messages.append({"role": "user", "content": inject})
                    continue

            # No action: model has finished.
            model_output += response
            transcript += response
            break

        return InferenceResult(
            raw_output=model_output,
            transcript=transcript,
            skill_ids_used=loaded_skill_ids,
        )
