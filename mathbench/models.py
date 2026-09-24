"""Model backends.

Real backends (Anthropic, OpenAI) use plain HTTPS so the suite has no SDK
dependency. ``SimulatedModel`` is an offline stand-in that injects realistic
error types at a configurable rate so the whole pipeline can be exercised and
tested without API keys.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import urllib.request
from abc import ABC, abstractmethod

import sympy as sp

from .task import Task

SYSTEM_PROMPT = (
    "Solve the problem. Reason step by step, then give the final answer on its own "
    "line as \\boxed{...} using an exact value (integer, fraction or closed form)."
)


class Model(ABC):
    name: str

    @abstractmethod
    def solve(self, task: Task, sample: int = 0) -> str: ...


def _post(url: str, headers: dict, body: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class AnthropicModel(Model):
    def __init__(self, model: str = "claude-sonnet-5", temperature: float = 1.0):
        self.model, self.temperature = model, temperature
        self.name = f"anthropic/{model}"
        self.key = os.environ["ANTHROPIC_API_KEY"]

    def solve(self, task: Task, sample: int = 0) -> str:
        out = _post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            {
                "model": self.model,
                "max_tokens": 4096,
                "temperature": self.temperature,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": task.question}],
            },
        )
        return "".join(b.get("text", "") for b in out["content"])


class OpenAIModel(Model):
    def __init__(self, model: str = "gpt-4o", temperature: float = 1.0):
        self.model, self.temperature = model, temperature
        self.name = f"openai/{model}"
        self.key = os.environ["OPENAI_API_KEY"]

    def solve(self, task: Task, sample: int = 0) -> str:
        out = _post(
            "https://api.openai.com/v1/chat/completions",
            {"authorization": f"Bearer {self.key}", "content-type": "application/json"},
            {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task.question},
                ],
            },
        )
        return out["choices"][0]["message"]["content"]


class SimulatedModel(Model):
    """Deterministic offline model with a per-topic skill profile."""

    ERROR_KINDS = ("arithmetic_slip", "sign_error", "off_by_one", "no_final_answer", "wrong_method")

    def __init__(self, name: str = "sim-strong", skill: float = 0.8, topic_skill: dict | None = None):
        self.name, self.skill, self.topic_skill = name, skill, topic_skill or {}

    def _rng(self, task: Task, sample: int) -> random.Random:
        h = hashlib.sha256(f"{self.name}|{task.id}|{sample}".encode()).hexdigest()
        return random.Random(int(h[:16], 16))

    def solve(self, task: Task, sample: int = 0) -> str:
        rng = self._rng(task, sample)
        p = self.topic_skill.get(task.topic, self.skill)
        if task.difficulty == "hard" and p < 1:
            p *= 0.8
        ref = sp.sympify(task.answer)
        steps = "Let me work through this.\n" + task.solution.split("\n")[0] + "\n"
        if rng.random() < p:
            return steps + f"\\boxed{{{sp.latex(ref)}}}"
        kind = rng.choice(self.ERROR_KINDS)
        if kind == "no_final_answer":
            return steps + "This requires a longer computation that I will set up but not finish."
        if kind == "sign_error":
            wrong = -ref if ref != 0 else sp.Integer(1)
        elif kind == "off_by_one":
            wrong = ref + rng.choice([-1, 1])
        elif kind == "arithmetic_slip":
            wrong = ref * sp.Rational(rng.choice([99, 101, 98, 102]), 100)
        else:
            wrong = ref * rng.randint(2, 5) + rng.randint(1, 9)
        return steps + f"\\boxed{{{sp.latex(sp.nsimplify(wrong))}}}"


def get_model(spec: str) -> Model:
    """Build a model from a spec such as ``sim:strong``, ``anthropic:claude-sonnet-5`` or ``openai:gpt-4o``."""
    kind, _, arg = spec.partition(":")
    if kind == "sim":
        presets = {
            "strong": dict(skill=0.85, topic_skill={"probability": 0.7, "number_theory": 0.9}),
            "weak": dict(skill=0.5, topic_skill={"calculus": 0.65, "linear_algebra": 0.35}),
        }
        return SimulatedModel(f"sim-{arg or 'strong'}", **presets.get(arg or "strong", {}))
    if kind == "anthropic":
        return AnthropicModel(arg or "claude-sonnet-5")
    if kind == "openai":
        return OpenAIModel(arg or "gpt-4o")
    raise ValueError(f"unknown model spec {spec!r}")
