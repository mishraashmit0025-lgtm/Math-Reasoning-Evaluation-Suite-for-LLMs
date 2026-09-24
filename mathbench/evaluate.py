"""Run models over tasks, grade, tag failures, summarize, and build preference pairs."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from .failures import tag_failure
from .grading import grade
from .models import Model
from .task import Task


def evaluate(model: Model, tasks: list[Task], samples: int = 1, workers: int = 8) -> list[dict]:
    jobs = [(t, s) for t in tasks for s in range(samples)]

    def run(job):
        task, s = job
        try:
            resp = model.solve(task, s)
        except Exception as e:  # network errors etc. are recorded, not fatal
            resp = f"[error] {type(e).__name__}: {e}"
        g = grade(resp, task.answer)
        return {
            "task_id": task.id,
            "topic": task.topic,
            "difficulty": task.difficulty,
            "model": model.name,
            "sample": s,
            "response": resp,
            "extracted": g.extracted,
            "correct": g.correct,
            "failure": tag_failure(g, task.answer),
        }

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(run, jobs))


def summarize(results: list[dict]) -> dict:
    by_model: dict[str, dict] = {}
    groups = defaultdict(list)
    for r in results:
        groups[r["model"]].append(r)
    for name, rows in groups.items():
        topic = defaultdict(lambda: [0, 0])
        for r in rows:
            topic[r["topic"]][0] += r["correct"]
            topic[r["topic"]][1] += 1
        by_model[name] = {
            "n": len(rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "by_topic": {k: c / n for k, (c, n) in sorted(topic.items())},
            "failures": dict(Counter(r["failure"] for r in rows if r["failure"]).most_common()),
        }
    return by_model


def preference_pairs(results: list[dict], tasks: list[Task]) -> list[dict]:
    """Chosen = verified-correct response, rejected = incorrect response, for the same prompt.

    Rejected responses are ranked by failure severity so the most informative
    contrast (a confident wrong answer) is paired first.
    """
    severity = {
        "wrong_magnitude": 0, "wrong_value": 1, "sign_error": 2, "off_by_one": 3,
        "arithmetic_slip": 4, "unparseable_answer": 5, "no_final_answer": 6,
    }
    task_by_id = {t.id: t for t in tasks}
    per_task = defaultdict(list)
    for r in results:
        per_task[r["task_id"]].append(r)
    pairs = []
    for tid, rows in per_task.items():
        good = [r for r in rows if r["correct"]]
        bad = sorted((r for r in rows if not r["correct"]), key=lambda r: severity.get(r["failure"], 9))
        for g, b in zip(good, bad):
            pairs.append({
                "prompt": task_by_id[tid].question,
                "chosen": g["response"],
                "rejected": b["response"],
                "meta": {
                    "task_id": tid,
                    "reference_answer": task_by_id[tid].answer,
                    "chosen_model": g["model"],
                    "rejected_model": b["model"],
                    "rejected_failure": b["failure"],
                },
            })
    return pairs
