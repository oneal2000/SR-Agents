"""MedCalc-Bench evaluation: answer extraction + type-aware scoring.

Ported from the upstream reference implementation at
https://github.com/ncbi-nlp/MedCalc-Bench (``evaluation/evaluate.py``).
"""

import re

from sragents.evaluate.base import register

from sragents.evaluate.common import extract_from_trigger, strip_think_tags

# Calculator IDs by output type (from reference evaluate.py)
_DATE_IDS = {13, 68}
_GESTATIONAL_ID = 69
_INTEGER_IDS = {
    4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48, 51, 69,
}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract(raw_output: str, eval_data: dict) -> str:
    output_type = eval_data.get("output_type", "decimal")
    calculator_id = eval_data.get("calculator_id", 0)

    # Structured ANSWER: line
    for line in reversed(raw_output.strip().split("\n")):
        line = line.strip()
        if line.upper().startswith("ANSWER:"):
            answer = line[len("ANSWER:"):].strip()
            answer = answer.strip("*").strip()
            return answer

    # JSON "answer" field
    m = re.search(r'[Aa]nswer":\s*(.*?)\}', raw_output)
    if m:
        answer = m.group(1).strip().strip('"').strip("'")
        if answer and answer not in (
            "str(short_and_direct_answer_of_the_question)",
            "str(value which is the answer to the question)",
            "X.XX",
        ):
            return answer

    # Trigger phrase
    answer = extract_from_trigger(raw_output)
    if answer is not None:
        return answer

    # Date pattern
    if output_type == "date" or calculator_id in _DATE_IDS:
        m = re.search(r"(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(\d{4})", raw_output)
        if m:
            return f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"

    # Gestational age pattern
    if calculator_id == _GESTATIONAL_ID:
        m = re.search(
            r"\(?[\"\']?(\d+)\s*(?:weeks?)?\s*,?\s*[\"\']?(\d+)\s*(?:days?)?[\"\']?\s*\)?",
            raw_output,
        )
        if m:
            return f"({m.group(1)}, {m.group(2)})"

    # Last number
    matches = re.findall(r"-?\d+\.?\d*", raw_output)
    if matches:
        return matches[-1]

    lines = [line.strip() for line in raw_output.strip().split("\n") if line.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _safe_parse_number(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$", s)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den != 0:
            return num / den
    return None


def _eval_date(extracted: str, eval_data: dict) -> dict:
    from datetime import datetime
    gt = str(eval_data["answer"]).strip()
    try:
        gt_dt = datetime.strptime(gt, "%m/%d/%Y")
        pred_dt = datetime.strptime(extracted.strip(), "%m/%d/%Y")
        correct = gt_dt == pred_dt
    except (ValueError, TypeError):
        correct = False
    return {"correct": correct, "output_type": "date"}


def _eval_gestational(extracted: str, eval_data: dict) -> dict:
    gt = str(eval_data["answer"]).strip()

    def _extract_weeks_days(s):
        m = re.search(
            r"\(?[\"\']?(\d+)\s*(?:weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(?:days?)?[\"\']?\s*\)?",
            s,
        )
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return None

    gt_tuple = _extract_weeks_days(gt)
    pred_tuple = _extract_weeks_days(extracted)
    correct = gt_tuple is not None and pred_tuple is not None and gt_tuple == pred_tuple
    return {"correct": correct, "output_type": "gestational_age"}


def _eval(extracted: str, eval_data: dict) -> dict:
    calid = eval_data.get("calculator_id", 0)
    output_type = eval_data.get("output_type", "decimal")

    if calid in _DATE_IDS:
        return _eval_date(extracted, eval_data)
    if calid == _GESTATIONAL_ID:
        return _eval_gestational(extracted, eval_data)

    gt_str = str(eval_data["answer"]).strip()

    # Integer
    if calid in _INTEGER_IDS or output_type == "integer":
        gt_num = _safe_parse_number(gt_str)
        pred_num = _safe_parse_number(extracted.strip())
        if gt_num is not None and pred_num is not None:
            correct = round(pred_num) == round(gt_num)
        else:
            correct = False
        return {"correct": correct, "output_type": "integer"}

    # Decimal (default): lower_limit <= pred <= upper_limit
    pred_num = _safe_parse_number(extracted.strip())
    lower = _safe_parse_number(str(eval_data.get("lower_limit", "")).strip())
    upper = _safe_parse_number(str(eval_data.get("upper_limit", "")).strip())

    if pred_num is not None and lower is not None and upper is not None:
        correct = lower <= pred_num <= upper
    else:
        correct = False
    return {"correct": correct, "output_type": "decimal"}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

@register("medcalcbench")
def evaluate(raw_output: str, instance: dict) -> dict:
    eval_data = instance["eval_data"]
    extracted = _extract(strip_think_tags(raw_output), eval_data)
    result = _eval(extracted, eval_data)
    result["extracted_answer"] = extracted
    return result
