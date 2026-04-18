"""Shared helpers for skill-provided tool execution.

A "tool" is a small Python function a skill optionally exposes alongside
its prose content. When the model emits ``TOOL_CALL: fn(arg=value)`` in its
output, an engine can intercept the call, execute the function, and splice
the result back into the conversation as ``TOOL_RESULT: ...``.

This module provides:

* :func:`parse_tool_call` — detect a tool call in model text.
* :func:`execute_tool` — run a tool's implementation in a restricted ns.
* :func:`run_with_tools` — full interception loop for a single-turn prompt.

``run_with_tools`` returns ``(model_output, transcript)``: the first
contains only model-generated tokens (evaluator input); the second also
contains injected ``TOOL_RESULT`` lines (diagnostic).

Tool schema (optional ``tools`` field in a skill)::

    {
      "tools": [
        {
          "name": "compute_ibw",
          "description": "Compute ideal body weight (kg)",
          "parameters": {"height_cm": "float", "gender": "str"},
          "implementation": "def compute_ibw(height_cm, gender): ..."
        }
      ]
    }
"""

import ast
import math
import re

from sragents.llm import chat_messages

_TOOL_CALL_RE = re.compile(
    r"^TOOL_CALL:\s*(\w+)\(([^)]*)\)\s*$",
    re.MULTILINE,
)

_MAX_TOOL_ROUNDS = 5

_SAFE_BUILTINS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "int": int, "float": float, "str": str, "len": len,
    "sum": sum, "pow": pow, "bool": bool,
    "True": True, "False": False, "None": None,
    "range": range, "enumerate": enumerate, "zip": zip,
    "isinstance": isinstance,
}


def parse_tool_call(
    text: str,
    available_tools: dict[str, dict],
) -> tuple[str, str, dict] | None:
    """Find the first tool call that matches an available tool.

    Returns ``(text_up_to_call_inclusive, tool_name, parsed_args)`` or
    ``None`` if no valid call is present.
    """
    for match in _TOOL_CALL_RE.finditer(text):
        name = match.group(1)
        if name in available_tools:
            args = _parse_call_args(
                match.group(2).strip(),
                available_tools[name],
            )
            if args is not None:
                return text[: match.end()], name, args
    return None


def _parse_call_args(args_str: str, tool_def: dict) -> dict | None:
    """Parse ``fn(a=1, b="x")`` or ``fn(1, "x")`` into a kwargs dict."""
    if not args_str:
        return {}
    try:
        tree = ast.parse(f"_f({args_str})", mode="eval")
        call = tree.body
        result: dict = {}
        param_names = list(tool_def.get("parameters", {}).keys())
        for i, arg in enumerate(call.args):
            key = param_names[i] if i < len(param_names) else f"_pos_{i}"
            result[key] = ast.literal_eval(arg)
        for kw in call.keywords:
            result[kw.arg] = ast.literal_eval(kw.value)
        return result
    except Exception:
        return None


def execute_tool(tool_def: dict, args: dict) -> str:
    """Run a tool implementation in a restricted namespace."""
    namespace = {"__builtins__": _SAFE_BUILTINS, "math": math}
    exec(tool_def["implementation"], namespace)  # noqa: S102
    func = namespace[tool_def["name"]]
    return str(func(**args))


def run_with_tools(
    client,
    model: str,
    system: str,
    user: str,
    tools: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    extra_body: dict | None = None,
    max_rounds: int = _MAX_TOOL_ROUNDS,
) -> tuple[str, str]:
    """Run generation with tool interception.

    Returns ``(model_output, transcript)``. ``model_output`` contains only
    model-generated tokens; ``transcript`` additionally includes injected
    ``TOOL_RESULT: ...`` lines for diagnostics.
    """
    tool_index = {t["name"]: t for t in tools}

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    model_output = ""
    transcript = ""

    for _ in range(max_rounds):
        response = chat_messages(
            client, model, messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        if not response:
            break

        parsed = parse_tool_call(response, tool_index)
        if parsed is None:
            model_output += response
            transcript += response
            break

        head, tool_name, args = parsed
        try:
            result = execute_tool(tool_index[tool_name], args)
        except Exception as e:  # noqa: BLE001
            result = f"Error: {e}"

        model_output += head
        transcript += head + f"\nTOOL_RESULT: {result}\n"

        messages.append({"role": "assistant", "content": head})
        messages.append({"role": "user", "content": f"TOOL_RESULT: {result}"})

    return model_output, transcript
