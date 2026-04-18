"""Calculator tool for ToolQA.

Evaluates arithmetic and simple aggregate expressions (mean, sum, etc.)
using Python's ``eval`` and the ``math`` / ``statistics`` stdlib.
"""

import math
import re
import statistics


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression, return result as string.

    Supports:
    - Basic arithmetic: 1+2*3, (-17)-(-7)
    - Functions via statistics: mean(...), median(...)
    - Functions via math: sqrt(), sin(), cos(), etc.
    """
    expr = expression.strip()

    # Handle mean(...) — most common in ToolQA
    mean_match = re.match(r"^mean\((.+)\)$", expr, re.IGNORECASE)
    if mean_match:
        nums = _parse_number_list(mean_match.group(1))
        result = statistics.mean(nums)
        return str(round(result, 3))

    # Handle median(...)
    median_match = re.match(r"^median\((.+)\)$", expr, re.IGNORECASE)
    if median_match:
        nums = _parse_number_list(median_match.group(1))
        result = statistics.median(nums)
        return str(round(result, 3))

    # Handle sum(...)
    sum_match = re.match(r"^sum\((.+)\)$", expr, re.IGNORECASE)
    if sum_match:
        nums = _parse_number_list(sum_match.group(1))
        result = sum(nums)
        return str(round(result, 3))

    # Handle max/min
    max_match = re.match(r"^max\((.+)\)$", expr, re.IGNORECASE)
    if max_match:
        nums = _parse_number_list(max_match.group(1))
        return str(max(nums))

    min_match = re.match(r"^min\((.+)\)$", expr, re.IGNORECASE)
    if min_match:
        nums = _parse_number_list(min_match.group(1))
        return str(min(nums))

    # General eval with math module available
    safe_dict = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "int": int,
        "float": float,
        "pow": pow,
        "sum": sum,
        "min": min,
        "max": max,
        "len": len,
    }
    # Add math functions
    for name in dir(math):
        if not name.startswith("_"):
            safe_dict[name] = getattr(math, name)

    result = eval(expr, safe_dict)  # noqa: S307
    if isinstance(result, float):
        # Round to avoid floating point noise
        if result == int(result):
            return str(int(result))
        return str(round(result, 3))
    return str(result)


def _parse_number_list(s: str) -> list[float]:
    """Parse comma-separated numbers, ignoring NaN values."""
    nums = []
    for part in s.split(","):
        part = part.strip()
        if not part or part.lower() == "nan":
            continue
        nums.append(float(part))
    return nums
