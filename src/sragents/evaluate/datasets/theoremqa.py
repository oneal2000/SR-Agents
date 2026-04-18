"""TheoremQA evaluation: answer extraction + exact-match scoring.

Ported from the upstream reference implementation at
https://github.com/TIGER-AI-Lab/TheoremQA (``utils.py`` + ``number_utils.py``).
"""

import math
import re

from sragents.evaluate.base import register

from sragents.evaluate.common import strip_think_tags, within_eps

_TRIGGERS = (
    "The answer is:",
    "the answer is:",
    "Therefore, the answer is",
    "therefore, the answer is",
)


# ---------------------------------------------------------------------------
# Numeric helpers (shared between extraction and evaluation)
# ---------------------------------------------------------------------------

def _clean_units(pred_str: str) -> str:
    def _convert_pi(s: str) -> str:
        s = s.replace("\\pi", "\u03c0")
        s = re.sub(r"(?<![\d}])\\?\u03c0", "3.14", s)
        s = re.sub(r"(\d)(\\?\u03c0)", r"\1*3.14", s)
        s = re.sub(r"\{(\\?\u03c0)\}", "3.14", s)
        s = re.sub(r"\*(\\?\u03c0)", "*3.14", s)
        return s

    pred_str = _convert_pi(pred_str)
    pred_str = pred_str.replace("%", "/100")
    pred_str = pred_str.replace("$", "")
    pred_str = pred_str.replace("\u00a5", "")
    pred_str = pred_str.replace("\u00b0C", "")
    pred_str = pred_str.replace(" C", "")
    pred_str = pred_str.replace("\u00b0", "")
    return pred_str


def _floatify(num) -> float | int | None:
    if isinstance(num, (int, float)):
        return num
    try:
        num = float(num)
        if num.is_integer():
            return round(num)
        return num
    except Exception:
        return None


def _number_it(num) -> float | int | None:
    if isinstance(num, (int, float)):
        return num
    num = _clean_units(str(num))
    try:
        from latex2sympy2 import latex2sympy
        num = str(latex2sympy(num))
    except Exception:
        pass
    result = _floatify(num)
    if result is not None:
        return result
    try:
        val = eval(num)  # noqa: S307
        if isinstance(val, (list, tuple)):
            val = val[0]
        result = _floatify(val)
        if result is not None:
            return result
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract_answer(pred: str, answer_flag: bool = True) -> str:
    """Core answer extraction, ported from TheoremQA utils.py."""
    if any(opt in pred.lower() for opt in ["yes", "true"]):
        return "True"
    if any(opt in pred.lower() for opt in ["no", "false"]):
        return "False"
    if any(opt in pred.lower() for opt in ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]):
        return pred

    if answer_flag:
        pred = pred.split("=")[-1].strip()
        pred = _clean_units(pred)
        try:
            from latex2sympy2 import latex2sympy
            tmp = str(latex2sympy(pred))
            pred = str(eval(tmp))  # noqa: S307
        except Exception:
            if re.match(r"-?[\d\.]+\s\D+$", pred):
                pred = pred.split(" ")[0]
            elif re.match(r"-?[\d\.]+\s[^\s]+$", pred):
                pred = pred.split(" ")[0]
    else:
        preds = re.findall(r"-?\d*\.?\d+", pred)
        if preds:
            pred = preds[-1]
        else:
            pred = ""
    return pred


def _extract(raw_output: str) -> str:
    pred = raw_output.strip("\n")

    # Detect ICL leakage
    icl = any(pred.count(t) > 1 for t in _TRIGGERS)
    if icl:
        pred = pred.split("\n\n")[0]

    preds = re.split("|".join(re.escape(t) for t in _TRIGGERS), pred)
    if len(preds) > 1:
        answer_flag = True
        pred = preds[-1]
    else:
        answer_flag = False

    pred = pred.strip("\n").rstrip(".").rstrip("/").strip()
    pred = _extract_answer(pred, answer_flag)
    pred = pred.rstrip(".").rstrip("/")
    return pred


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _compare_two_numbers(p, gt) -> bool:
    try:
        if p is None or math.isnan(p):
            return False
        if isinstance(gt, int):
            return round(p) == gt
        return within_eps(pred=p, gt=gt)
    except Exception:
        return False


def _compare_two_list(pred, gt) -> bool:
    if not isinstance(pred, list):
        return False
    if len(pred) != len(gt):
        return False
    if any(not isinstance(x, (int, float)) for x in pred):
        return False
    return all(
        _compare_two_numbers(p, g)
        for p, g in zip(sorted(pred), sorted(gt))
    )


def _compare_answer_with_groundtruth(answer, groundtruth_str, groundtruth_num=None):
    if groundtruth_str.lower() in ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)"):
        return groundtruth_str.lower() in answer.lower()
    if answer.lower() == groundtruth_str.lower():
        return True
    if groundtruth_num is not None:
        if isinstance(groundtruth_num, (int, float)):
            return _compare_two_numbers(_number_it(answer), groundtruth_num)
        else:
            if answer.startswith("(") and answer.endswith(")"):
                try:
                    answer_list = list(eval(answer))  # noqa: S307
                    answer_list = [_number_it(a) for a in answer_list]
                    return _compare_two_list(answer_list, groundtruth_num)
                except Exception:
                    return False
            return False
    return False


def _eval(extracted: str, eval_data: dict) -> dict:
    gt_str = str(eval_data["answer"]).strip()
    answer_type = eval_data.get("answer_type", "float")
    gt_num = None

    if "list" in answer_type:
        try:
            parsed = eval(gt_str)  # noqa: S307
            if isinstance(parsed, (list, tuple)):
                gt_num = list(parsed)
        except Exception:
            gt_num = None
    elif answer_type in ("integer", "float"):
        gt_num = _floatify(gt_str)

    correct = _compare_answer_with_groundtruth(extracted, gt_str, gt_num)
    return {"correct": correct, "answer_type": answer_type}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

@register("theoremqa")
def evaluate(raw_output: str, instance: dict) -> dict:
    extracted = _extract(strip_think_tags(raw_output))
    result = _eval(extracted, instance["eval_data"])
    result["extracted_answer"] = extracted
    return result
