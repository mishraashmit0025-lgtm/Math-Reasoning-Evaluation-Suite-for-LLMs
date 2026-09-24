"""Registry of task families and the verified task builder."""
from __future__ import annotations

import hashlib
import random

import sympy as sp

from ..task import Task
from . import algebra_calculus, combinatorics_nt, probability_stats

GENERATORS = {
    **combinatorics_nt.GENERATORS,
    **probability_stats.GENERATORS,
    **algebra_calculus.GENERATORS,
}


class VerificationError(AssertionError):
    pass


def _agrees(answer: str, reference) -> bool:
    ans = sp.sympify(answer)
    try:
        ref = sp.nsimplify(reference) if not isinstance(reference, sp.Basic) else reference
        if sp.simplify(ans - ref) == 0:
            return True
    except (TypeError, ValueError, sp.SympifyError):
        pass
    return abs(complex(sp.N(ans, 30)) - complex(reference)) < 1e-9


def build_tasks(seed: int = 2024, per_family: int = 16) -> list[Task]:
    """Generate ``per_family`` tasks for every family, each cross-verified."""
    tasks: list[Task] = []
    for family, gen in GENERATORS.items():
        rng = random.Random(f"{seed}:{family}")
        seen: set[str] = set()
        attempts = 0
        while len(seen) < per_family:
            attempts += 1
            if attempts > per_family * 20:
                raise RuntimeError(f"{family}: could not produce {per_family} unique tasks")
            q, ans, sol, diff, tags, verify = gen(rng)
            if q in seen:
                continue
            if not _agrees(ans, verify()):
                raise VerificationError(f"{family}: answer {ans} disagrees with brute force for: {q}")
            seen.add(q)
            tid = family.replace("/", ".") + "-" + hashlib.sha1(q.encode()).hexdigest()[:8]
            topic = family.split("/")[0]
            tasks.append(Task(tid, topic, diff, q, ans, sol, [family.split("/")[1], *tags]))
    return tasks
