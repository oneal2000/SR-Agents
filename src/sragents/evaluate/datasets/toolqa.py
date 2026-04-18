"""ToolQA evaluation: answer extraction + normalized exact-match scoring."""

import re
import string as _string

from sragents.evaluate.base import register

from sragents.evaluate.common import extract_from_trigger, strip_think_tags


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract(raw_output: str) -> str:
    # ToolQA raw_output is a ReAct scratchpad — find the last Finish[...] action
    matches = re.findall(r"Finish\[([^\]]*)\]", raw_output)
    if matches:
        return matches[-1].strip()

    # Fallback: trigger phrase
    answer = extract_from_trigger(raw_output)
    if answer is not None:
        return answer

    # Last non-empty line
    lines = [line.strip() for line in raw_output.strip().split("\n") if line.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Normalize answer for comparison (ported from ToolQA reference)."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the|usd)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(_string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

@register("toolqa")
def evaluate(raw_output: str, instance: dict) -> dict:
    eval_data = instance["eval_data"]
    gt = str(eval_data["answer"]).strip()
    extracted = _extract(strip_think_tags(raw_output))

    # Normalized exact match
    if _normalize(extracted) == _normalize(gt):
        return {"extracted_answer": extracted, "correct": True, "match_type": "exact"}

    # Numeric fallback
    try:
        if abs(float(extracted) - float(gt)) < 1e-6:
            return {"extracted_answer": extracted, "correct": True, "match_type": "numeric"}
    except (ValueError, TypeError):
        pass

    # Boolean fallback
    _bool_map = {"true": "yes", "false": "no"}
    pred_mapped = _bool_map.get(extracted.lower())
    if pred_mapped is not None and pred_mapped == gt.strip().lower():
        return {"extracted_answer": extracted, "correct": True, "match_type": "boolean"}

    return {"extracted_answer": extracted, "correct": False, "match_type": "none"}
