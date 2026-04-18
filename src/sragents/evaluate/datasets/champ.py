"""CHAMP evaluation: answer extraction + multi-strategy exact-match scoring."""

import math
import re

from sragents.evaluate.base import register

from sragents.evaluate.common import (
    extract_from_trigger,
    strip_think_tags,
    within_eps,
)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract(raw_output: str) -> str:
    # Try structured ANSWER: line (last occurrence)
    for line in reversed(raw_output.strip().split("\n")):
        line = line.strip()
        if line.upper().startswith("ANSWER:"):
            answer = line[len("ANSWER:"):].strip()
            answer = answer.strip("*").strip()
            return answer

    answer = extract_from_trigger(raw_output)
    if answer is not None:
        return answer

    lines = [line.strip() for line in raw_output.strip().split("\n") if line.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _extract_alternatives(s: str) -> list[str]:
    """E.g. "C(10, 5), or equivalently 252" → ["252"]."""
    alts = []
    m = re.search(r",?\s*or\s+(?:equivalently|approximately)\s+(.+)$", s, flags=re.I)
    if m:
        alt_part = m.group(1).strip().rstrip(".")
        alt_part = re.sub(r"\s*\([^)]*\)\s*$", "", alt_part).strip()
        if alt_part:
            alts.append(alt_part)
    return alts


def _normalize_str(s: str) -> str:
    s = s.strip().rstrip(".")
    s = re.sub(r",?\s*or\s+(?:equivalently|approximately)\s+.*$", "", s, flags=re.I)
    s = re.sub(r"\s+\((?:i\.e\.[,\s]|none\b)[^)]*\)\s*$", "", s, flags=re.I)
    s = " ".join(s.split())
    return s.strip()


def _try_parse_number(s: str) -> float | int | None:
    s = s.strip()
    if not s:
        return None
    try:
        val = float(s)
        if val.is_integer():
            return int(val)
        return val
    except ValueError:
        pass
    m = re.match(r"^(-?\d+)\s*/\s*(\d+)$", s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den != 0:
            return num / den
    expr = s.replace("^", "**")
    expr = re.sub(r"\bsqrt\(([^)]+)\)", r"(\1)**0.5", expr)
    try:
        val = eval(expr)  # noqa: S307
        if isinstance(val, (int, float)):
            return val
    except Exception:
        pass
    return None


def _sympy_equal(s1: str, s2: str) -> bool:
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
            convert_xor,
        )
    except ImportError:
        return False

    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    def _prep(s: str) -> str:
        s = s.strip().replace("^", "**")
        s = re.sub(r"\bsqrt\(", "sqrt(", s)
        s = re.sub(r"\bC\(([^,]+),\s*([^)]+)\)", r"binomial(\1, \2)", s)
        s = re.sub(r"\b(\w+)!", r"factorial(\1)", s)
        return s

    try:
        e1 = parse_expr(_prep(s1), transformations=transformations)
        e2 = parse_expr(_prep(s2), transformations=transformations)
        return sympy.simplify(e1 - e2) == 0
    except Exception:
        return False


_CANONICAL_MAP = [
    (r"^no\s+(?:\w+\s+)*solutions?$", "no solutions"),
    (r"^no\s+(?:\w+\s+)*(?:possible\s+)?values?$", "no such values"),
    (r"^no such values$", "no such values"),
    (r"^0 pairs$", "0"),
    (r"^0 roots$", "0"),
    (r"^(.+?)\s+is the only possible value.*$", r"\1"),
    (r"^the limit exists and is equal to\s+(.+)$", r"\1"),
    (r"^the limit does not exist$", "limit does not exist"),
    (r"^at most 0\b.*$", "0"),
    (r"^exactly one\b.*$", "1"),
    (r"^(\d+)\s+values?$", r"\1"),
    (r"^(\d+)\s+inequalit(?:y|ies)$", r"\1"),
]


def _canonicalize(s: str) -> str:
    s_lower = s.strip().lower()
    for pattern, replacement in _CANONICAL_MAP:
        m = re.match(pattern, s_lower)
        if m:
            return m.expand(replacement)
    return s


def _try_match(gt_str: str, pred_str: str) -> dict | None:
    gt_canon = _canonicalize(gt_str)
    pred_canon = _canonicalize(pred_str)

    if gt_canon.lower() == pred_canon.lower():
        return {"correct": True, "match_type": "canonical"}

    gt_num = _try_parse_number(gt_canon)
    pred_num = _try_parse_number(pred_canon)
    if gt_num is not None and pred_num is not None:
        try:
            if isinstance(gt_num, int):
                correct = math.isfinite(pred_num) and round(pred_num) == gt_num
            else:
                correct = within_eps(pred_num, gt_num)
        except (OverflowError, ValueError):
            correct = False
        if correct:
            return {"correct": True, "match_type": "numeric"}

    if _sympy_equal(gt_canon, pred_canon):
        return {"correct": True, "match_type": "symbolic"}

    return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

@register("champ")
def evaluate(raw_output: str, instance: dict) -> dict:
    eval_data = instance["eval_data"]
    gt_raw = str(eval_data["answer"]).strip()
    extracted = _extract(strip_think_tags(raw_output))
    pred_raw = extracted.strip()

    gt_alternatives = _extract_alternatives(gt_raw)
    gt_norm = _normalize_str(gt_raw)
    pred_norm = _normalize_str(pred_raw)

    # Exact string match
    if gt_norm.lower() == pred_norm.lower():
        return {"extracted_answer": extracted, "correct": True, "match_type": "exact"}

    # Yes/No
    gt_lower = gt_norm.lower()
    pred_lower = pred_norm.lower()
    if gt_lower in ("yes", "no"):
        pred_yn = None
        if any(w in pred_lower for w in ("yes", "true")):
            pred_yn = "yes"
        elif any(w in pred_lower for w in ("no", "false")):
            pred_yn = "no"
        if pred_yn is not None:
            return {"extracted_answer": extracted, "correct": pred_yn == gt_lower, "match_type": "yes_no"}

    # Canonical, numeric, symbolic
    result = _try_match(gt_norm, pred_norm)
    if result is not None:
        result["extracted_answer"] = extracted
        return result

    # Alternative representations
    for alt in gt_alternatives:
        alt_norm = _normalize_str(alt)
        if alt_norm.lower() == pred_norm.lower():
            return {"extracted_answer": extracted, "correct": True, "match_type": "exact_alt"}
        result = _try_match(alt_norm, pred_norm)
        if result is not None:
            result["match_type"] += "_alt"
            result["extracted_answer"] = extracted
            return result

    return {"extracted_answer": extracted, "correct": False, "match_type": "none"}
