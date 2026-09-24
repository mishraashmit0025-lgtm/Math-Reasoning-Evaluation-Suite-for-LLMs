"""CLI: python -m mathbench {build,eval,report,pairs}"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import evaluate, preference_pairs, summarize
from .generators import build_tasks
from .models import get_model
from .task import load_tasks, read_jsonl, write_jsonl


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mathbench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="generate and verify the task set")
    b.add_argument("--seed", type=int, default=2024)
    b.add_argument("--per-family", type=int, default=16)
    b.add_argument("--out", default="data/tasks.jsonl")

    e = sub.add_parser("eval", help="run one or more models over the tasks")
    e.add_argument("--tasks", default="data/tasks.jsonl")
    e.add_argument("--model", action="append", required=True,
                   help="sim:strong | sim:weak | anthropic:<model> | openai:<model> (repeatable)")
    e.add_argument("--samples", type=int, default=1)
    e.add_argument("--limit", type=int)
    e.add_argument("--out", default="results/results.jsonl")

    r = sub.add_parser("report", help="accuracy by model/topic and failure-mode counts")
    r.add_argument("--results", default="results/results.jsonl")

    p = sub.add_parser("pairs", help="build chosen/rejected preference pairs for RLHF / DPO")
    p.add_argument("--tasks", default="data/tasks.jsonl")
    p.add_argument("--results", default="results/results.jsonl")
    p.add_argument("--out", default="results/preference_pairs.jsonl")

    a = ap.parse_args(argv)
    if a.cmd == "build":
        tasks = build_tasks(a.seed, a.per_family)
        n = write_jsonl(a.out, (t.to_dict() for t in tasks))
        print(f"wrote {n} verified tasks to {a.out}")
    elif a.cmd == "eval":
        tasks = load_tasks(a.tasks)[: a.limit]
        rows = []
        for spec in a.model:
            m = get_model(spec)
            res = evaluate(m, tasks, a.samples)
            acc = sum(x["correct"] for x in res) / len(res)
            print(f"{m.name}: accuracy {acc:.3f} over {len(res)} responses")
            rows += res
        write_jsonl(a.out, rows)
    elif a.cmd == "report":
        print(json.dumps(summarize(read_jsonl(a.results)), indent=2))
    elif a.cmd == "pairs":
        pairs = preference_pairs(read_jsonl(a.results), load_tasks(a.tasks))
        n = write_jsonl(a.out, pairs)
        print(f"wrote {n} preference pairs to {a.out}")


if __name__ == "__main__":
    main()
