"""Smoke test S10: oracle isolation must fail closed.

Report reference: PROJECT_REPORT.md §10.4 — "The tests/isolation job is
non-negotiable: it asserts that a learner-side call to true_cost raises.
If that test is ever skipped, every number the repository produces
becomes unfalsifiable."
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from safelie.envs.guards import LearnerAccessError
from safelie.envs.synthetic import SyntheticConstrainedMarlEnv

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "safelie"
LEARNER_MODULES = ["training", "algos"]


def test_bare_oracle_handle_raises_on_true_cost():
    env = SyntheticConstrainedMarlEnv(n_agents=3)
    env.reset(seed=0)
    env.step({aid: [0.1, 0.1] for aid in env.agent_ids})

    handle = env.oracle_handle()  # the public, learner-reachable method
    with pytest.raises(LearnerAccessError):
        handle.true_cost()


def test_dual_cost_step_has_no_true_cost_field():
    """Structural check: the object the learner receives every step simply
    does not carry a true_cost attribute, under any name, so there is no
    path (info dict included) through which it could leak."""
    env = SyntheticConstrainedMarlEnv(n_agents=2)
    step = env.reset(seed=0)
    assert not hasattr(step, "true_cost")
    assert "true_cost" not in step.info
    step2 = env.step({aid: [0.0, 0.0] for aid in env.agent_ids})
    assert not hasattr(step2, "true_cost")
    assert "true_cost" not in step2.info
    # exhaustive: no field of the learner-visible object contains "true_cost"
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(step2)}
    assert "true_cost" not in field_names


def test_privileged_handle_actually_works_for_the_evaluator():
    """The isolation boundary must fail closed for the learner but not be
    so aggressive that the legitimate oracle path is also broken."""
    env = SyntheticConstrainedMarlEnv(n_agents=2)
    env.reset(seed=0)
    env.step({aid: [0.5, 0.5] for aid in env.agent_ids})
    privileged = env._oracle_handle_privileged()
    true_cost = privileged.true_cost()
    assert set(true_cost) == set(env.agent_ids)
    assert all(v >= 0.0 for v in true_cost.values())  # C^i >= 0, Definition 1


@pytest.mark.parametrize("module_name", LEARNER_MODULES)
def test_no_learner_module_calls_the_privileged_oracle_path(module_name):
    """Grep-level test (S10's second clause): no training/algorithm module
    may *reference* `_oracle_handle_privileged` as code (an attribute
    access, e.g. `env._oracle_handle_privileged()`) or import
    `safelie.eval.oracle` — the privileged path is reserved for the
    evaluator alone. This checks the AST, not raw text, so a module's own
    docstring is free to *discuss* the isolation boundary (as
    `safelie.training.loop` does) without tripping the check — only actual
    code use is forbidden.
    """
    module_dir = SRC_ROOT / module_name
    if not module_dir.exists():
        pytest.skip(f"{module_name} module not present")
    for path in module_dir.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "_oracle_handle_privileged":
                pytest.fail(
                    f"{path} accesses ._oracle_handle_privileged as code; this "
                    f"must be reserved for safelie.eval.oracle.OracleEvaluator (S10)"
                )
            if isinstance(node, ast.ImportFrom) and node.module == "safelie.eval.oracle":
                pytest.fail(f"{path} imports safelie.eval.oracle directly (S10)")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "safelie.eval.oracle", (
                        f"{path} imports safelie.eval.oracle directly (S10)"
                    )
