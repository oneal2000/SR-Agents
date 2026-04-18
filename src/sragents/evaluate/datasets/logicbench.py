"""LogicBench evaluation: answer extraction + exact-match scoring."""

import re

from sragents.evaluate.base import register

from sragents.evaluate.common import extract_from_trigger, strip_think_tags


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract_bqa(text: str) -> str:
    lower = text.lower()

    # Last-match semantics: a model may state a tentative answer early
    # and revise it later; the final stated answer wins.
    matches = list(re.finditer(
        r"(?:the\s+)?answer\s+is[:\s]*\**\s*(yes|no|true|false)\b", lower
    ))
    if matches:
        return "yes" if matches[-1].group(1) in ("yes", "true") else "no"

    m = re.match(r"\**(yes|no)\**[.,!\s]", lower)
    if m:
        return m.group(1)

    tail = lower[-500:]
    neg_patterns = [
        r"cannot\s+(?:be\s+)?(?:conclude|infer|say|determine)",
        r"not\s+necessarily\s+true",
        r"not\s+(?:possible|correct|true|valid)",
        r"cannot\s+(?:logically|necessarily|definitively)",
        r"\bno,\s",
    ]
    for pat in neg_patterns:
        if re.search(pat, tail):
            return "no"

    matches = list(re.finditer(r"\b(yes|no)\b", tail))
    if matches:
        return matches[-1].group(1)

    if "true" in tail:
        return "yes"
    if "false" in tail:
        return "no"
    return text.split("\n")[-1].strip().lower()


def _extract_mcqa(text: str, question: str) -> str:
    lower = text.lower()

    matches = list(re.finditer(r"choice[_ ]?(\d+)", lower))
    if matches:
        return f"choice_{matches[-1].group(1)}"

    m = re.search(
        r"(?:answer|option)\s*(?:is)?[:\s]*\**\s*(?:choice[_ ]?)?(\d+)\b", lower
    )
    if m:
        return f"choice_{m.group(1)}"

    # Try to match actual choice text from the question
    choice_texts = []
    for i in range(1, 6):
        qm = re.search(
            rf"choice_{i}:\s*(.+?)(?:\n|choice_|$)",
            question,
            re.IGNORECASE,
        )
        if qm:
            choice_texts.append((i, qm.group(1).strip()))

    tail = lower[-500:]
    for idx, ct in reversed(choice_texts):
        if ct.lower()[:40] in tail:
            return f"choice_{idx}"

    answer = extract_from_trigger(text)
    if answer is None:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        answer = lines[-1] if lines else ""
    answer = answer.strip().lower()

    m = re.search(r"\b([1-5])\b", answer)
    if m:
        return f"choice_{m.group(1)}"
    return answer


def _extract(raw_output: str, instance: dict) -> str:
    text = raw_output.strip()
    task_type = instance["eval_data"].get("task_type", "BQA")
    if task_type == "BQA":
        return _extract_bqa(text)
    return _extract_mcqa(text, instance.get("question", ""))


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

@register("logicbench")
def evaluate(raw_output: str, instance: dict) -> dict:
    eval_data = instance["eval_data"]
    extracted = _extract(strip_think_tags(raw_output), instance)
    gt = eval_data["answer"].strip().lower()
    pred = extracted.strip().lower()
    return {
        "extracted_answer": extracted,
        "correct": pred == gt,
        "task_type": eval_data.get("task_type", "BQA"),
    }
