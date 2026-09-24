import sympy as sp
import pytest

from mathbench.evaluate import evaluate, preference_pairs, summarize
from mathbench.failures import tag_failure
from mathbench.generators import GENERATORS, build_tasks
from mathbench.grading import extract_answer, grade
from mathbench.models import SimulatedModel, get_model
from mathbench.task import load_tasks


@pytest.fixture(scope="module")
def tasks():
    return build_tasks(seed=7, per_family=3)


def test_every_family_builds_and_verifies(tasks):
    families = {t.id.rsplit("-", 1)[0] for t in tasks}
    assert len(families) == len(GENERATORS)
    assert len(tasks) == 3 * len(GENERATORS)


def test_build_is_deterministic():
    a = [t.question for t in build_tasks(seed=11, per_family=2)]
    b = [t.question for t in build_tasks(seed=11, per_family=2)]
    assert a == b


def test_shipped_dataset_has_250_plus_unique_tasks():
    shipped = load_tasks("data/tasks.jsonl")
    assert len(shipped) >= 250
    assert len({t.question for t in shipped}) == len(shipped)
    for t in shipped:
        sp.sympify(t.answer)  # every answer parses


@pytest.mark.parametrize("text,expected", [
    ("so the result is \\boxed{\\frac{3}{4}}", "\\frac{3}{4}"),
    ("nested \\boxed{2^{10}} done", "2^{10}"),
    ("Final answer: 42.", "42"),
    ("I think it is 17, or maybe 19", "19"),
    ("no numbers here", None),
])
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


@pytest.mark.parametrize("resp,ref,ok", [
    ("\\boxed{\\frac{3}{4}}", "3/4", True),
    ("\\boxed{0.75}", "3/4", True),
    ("\\boxed{\\frac{\\pi}{4}}", "pi/4", True),
    ("\\boxed{2\\sqrt{2}}", "2*sqrt(2)", True),
    ("\\boxed{1/3}", "3/4", False),
])
def test_grade_equivalence(resp, ref, ok):
    assert grade(resp, ref).correct is ok


@pytest.mark.parametrize("resp,ref,tag", [
    ("no answer given", "10", "no_final_answer"),
    ("\\boxed{-10}", "10", "sign_error"),
    ("\\boxed{11}", "10", "off_by_one"),
    ("\\boxed{1010}", "1000", "arithmetic_slip"),
    ("\\boxed{50}", "10", "wrong_magnitude"),
])
def test_failure_tags(resp, ref, tag):
    assert tag_failure(grade(resp, ref), ref) == tag


def test_perfect_model_scores_100(tasks):
    res = evaluate(SimulatedModel("oracle", skill=1.0), tasks)
    assert all(r["correct"] for r in res)


def test_end_to_end_pairs(tasks):
    res = evaluate(get_model("sim:strong"), tasks, samples=2) + evaluate(get_model("sim:weak"), tasks, samples=2)
    summary = summarize(res)
    assert summary["sim-strong"]["accuracy"] > summary["sim-weak"]["accuracy"]
    pairs = preference_pairs(res, tasks)
    assert pairs
    for p in pairs:
        assert grade(p["chosen"], p["meta"]["reference_answer"]).correct
        assert not grade(p["rejected"], p["meta"]["reference_answer"]).correct
