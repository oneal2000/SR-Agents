"""BigCodeBench evaluation: code extraction (tree-sitter) + unittest execution."""

from sragents.evaluate.base import register
from sragents.evaluate.common import strip_think_tags
from sragents.evaluate.datasets.bigcodebench.execution import (
    PASS,
    untrusted_check,
)
from sragents.evaluate.datasets.bigcodebench.sanitize import sanitize

# Default resource limits (same as BigCodeBench defaults)
_MAX_AS_LIMIT = 30 * 1024   # 30 GB
_MAX_DATA_LIMIT = 30 * 1024  # 30 GB
_MAX_STACK_LIMIT = 10        # 10 MB


def _extract(raw_output: str, eval_data: dict) -> str:
    entry_point = eval_data.get("entry_point", "")
    return sanitize(raw_output, entrypoint=entry_point)


@register("bigcodebench")
def evaluate(raw_output: str, instance: dict) -> dict:
    eval_data = instance["eval_data"]
    extracted = _extract(strip_think_tags(raw_output), eval_data)

    test_code = eval_data.get("test", "")
    entry_point = eval_data.get("entry_point", "")
    code_prompt = eval_data.get("code_prompt", "")

    if not test_code:
        return {"extracted_answer": extracted, "correct": False, "result": "no test code"}

    # Calibrated mode: prepend code_prompt
    if code_prompt:
        solution = code_prompt + "\n    pass\n" + extracted
    else:
        solution = extracted

    stat, details = untrusted_check(
        solution, test_code, entry_point,
        max_as_limit=_MAX_AS_LIMIT,
        max_data_limit=_MAX_DATA_LIMIT,
        max_stack_limit=_MAX_STACK_LIMIT,
    )

    return {
        "extracted_answer": extracted,
        "correct": stat == PASS,
        "result": stat,
        "details": details,
    }
