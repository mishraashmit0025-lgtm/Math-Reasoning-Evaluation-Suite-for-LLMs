"""Task schema shared by generators, evaluator and preference builder."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class Task:
    id: str
    topic: str
    difficulty: str  # "medium" | "hard" | "expert"
    question: str
    answer: str  # canonical answer as a SymPy-parsable string
    solution: str  # verified step-by-step reference solution
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(**d)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tasks(path: str | Path) -> list[Task]:
    return [Task.from_dict(d) for d in read_jsonl(path)]
