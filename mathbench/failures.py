"""Heuristic failure-mode tagging for incorrect responses."""
from __future__ import annotations

import sympy as sp

from .grading import Grade, parse_math

FAILURE_TAGS = (
    "no_final_answer",
    "unparseable_answer",
    "sign_error",
    "off_by_one",
    "arithmetic_slip",
    "wrong_magnitude",
    "wrong_value",
)


def tag_failure(grade: Grade, reference: str) -> str | None:
    """Return a failure tag for an incorrect grade, or None when correct."""
    if grade.correct:
        return None
    if grade.extracted is None:
        return "no_final_answer"
    if not grade.parsed:
        return "unparseable_answer"
    pred, ref = parse_math(grade.extracted), sp.sympify(reference)
    try:
        pv, rv = complex(sp.N(pred)), complex(sp.N(ref))
    except (TypeError, ValueError):
        return "wrong_value"
    if rv != 0 and abs(pv + rv) < 1e-9 * max(1, abs(rv)):
        return "sign_error"
    if abs(abs(pv - rv) - 1) < 1e-9:
        return "off_by_one"
    if grade.rel_error is not None and grade.rel_error < 0.05:
        return "arithmetic_slip"
    if rv != 0 and pv != 0 and (abs(pv / rv) > 1.9 or abs(pv / rv) < 0.52):
        return "wrong_magnitude"
    return "wrong_value"
