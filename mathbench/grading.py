"""Answer extraction and symbolic equivalence grading."""
from __future__ import annotations

import re
from dataclasses import dataclass

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_NUMBER = r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?"


def _boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    depth, start = 0, idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i]
            depth -= 1
    return None


def extract_answer(response: str) -> str | None:
    """Pull the final answer out of a free-form model response.

    Priority: \\boxed{...} > "Final answer: ..." line > last number in the text.
    """
    if not response:
        return None
    boxed = _boxed(response)
    if boxed is not None:
        return boxed.strip()
    m = re.findall(r"(?:final answer|answer)\s*(?:is|:|=)\s*\$?([^\n$]+)", response, flags=re.I)
    if m:
        return m[-1].strip().rstrip(".")
    nums = re.findall(_NUMBER, response)
    return nums[-1].replace(" ", "") if nums else None


def _latex_to_plain(s: str) -> str:
    s = s.replace("\\left", "").replace("\\right", "").replace("\\,", "").replace("$", "")
    s = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("^", "**")
    return s.strip()


def parse_math(s: str) -> sp.Expr | None:
    try:
        return parse_expr(_latex_to_plain(s), transformations=_TRANSFORMS, evaluate=True)
    except Exception:
        return None


@dataclass
class Grade:
    correct: bool
    extracted: str | None
    parsed: bool
    rel_error: float | None  # |pred - ref| / max(1, |ref|), when both are numeric


def grade(response: str, reference: str, tol: float = 1e-9) -> Grade:
    extracted = extract_answer(response)
    if extracted is None:
        return Grade(False, None, False, None)
    pred, ref = parse_math(extracted), sp.sympify(reference)
    if pred is None:
        return Grade(False, extracted, False, None)
    rel = None
    try:
        pv, rv = complex(sp.N(pred, 30)), complex(sp.N(ref, 30))
        rel = abs(pv - rv) / max(1.0, abs(rv))
    except (TypeError, ValueError):
        pass
    try:
        exact = sp.simplify(pred - ref) == 0
    except Exception:
        exact = False
    return Grade(exact or (rel is not None and rel < tol), extracted, True, rel)
