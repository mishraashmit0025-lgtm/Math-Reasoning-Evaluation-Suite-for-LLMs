# Math Reasoning Evaluation Suite for LLMs

A benchmark and toolkit for testing how well large language models handle multi-step math. It includes:

- **256 math tasks with verified answers** across 16 task families: combinatorics, number theory, probability, statistics, linear algebra, calculus, recurrences and optimization. Every task comes with a step-by-step reference solution. Each answer is computed in closed form with SymPy, then checked against a separate brute-force or numeric computation before the task is accepted.
- **An evaluation harness** for Claude (Anthropic API), GPT (OpenAI API) and offline simulated models. It pulls the final answer out of each response (`\boxed{}`, "Final answer:", or the last number) and grades it by symbolic equivalence, so `\frac{\pi}{4}`, `pi/4` and `0.785398...` all count as the same answer.
- **Failure-mode tagging**: wrong answers are sorted into `sign_error`, `off_by_one`, `arithmetic_slip`, `wrong_magnitude`, `no_final_answer`, `unparseable_answer` or `wrong_value`.
- **Preference pairs for RLHF / DPO**: `{prompt, chosen, rejected, meta}` JSONL records. Each pairs a verified-correct response with an incorrect one, ranked so that confident wrong answers are paired first.

## Quick start

```bash
pip install -r requirements.txt

python -m mathbench build                                   # regenerate + verify data/tasks.jsonl
python -m mathbench eval --model sim:strong --model sim:weak --samples 2
python -m mathbench report                                  # accuracy by topic + failure counts
python -m mathbench pairs                                   # results/preference_pairs.jsonl
python -m pytest -q
```

Real models (keys are read from the environment):

```bash
export ANTHROPIC_API_KEY=...   OPENAI_API_KEY=...
python -m mathbench eval --model anthropic:claude-sonnet-5 --model openai:gpt-4o --samples 3
```

`notebooks/failure_analysis.ipynb` breaks results down by model, topic and difficulty, and lists the tasks that every model failed.

## Layout

```
mathbench/
  generators/        task families; each returns (question, answer, solution, verify)
  task.py            Task schema + JSONL I/O
  grading.py         answer extraction, LaTeX -> SymPy parsing, equivalence check
  failures.py        failure-mode tagging
  models.py          Anthropic / OpenAI HTTP clients + deterministic SimulatedModel
  evaluate.py        parallel evaluation, summaries, preference-pair builder
data/tasks.jsonl     the 256-task benchmark (seed 2024)
tests/               pytest suite (generation, grading, tagging, end-to-end)
```

## Adding a task family

Write a function `gen(rng) -> (question, answer, solution, difficulty, tags, verify)` where `verify()` recomputes the answer by an independent method. Then register it in the module's `GENERATORS` dict. `build_tasks` will reject any task whose answer does not match its verifier.

## Note on the simulated models

`sim:strong` and `sim:weak` are offline stand-ins that answer correctly at a set rate per topic and inject realistic errors otherwise. They let CI exercise the whole pipeline without API keys. **Their accuracy numbers are not measurements of any real model.**
