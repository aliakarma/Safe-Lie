"""Regression test: a config naming an unimplemented environment must
fail immediately and loudly, never silently fall back to the synthetic
environment.

This guards against a real bug found while building this repository:
`ExperimentRun.__init__` originally hardcoded
`SyntheticConstrainedMarlEnv` regardless of `cfg.env.name`, so a
`manyagent_ant` config would silently train on the wrong environment
instead of raising -- see SMOKE_TEST_REPORT.md's "fixes applied" section.
"""

from __future__ import annotations

import pytest

from safelie.envs.factory import build_env
from safelie.envs.synthetic import SyntheticConstrainedMarlEnv
from safelie.sources.registry import default_m7_sources
from safelie.training.loop import ExperimentRun
from safelie.utils.config import EnvConfig, ExperimentConfig


def test_build_env_returns_synthetic_for_synthetic_name():
    cfg = EnvConfig(name="synthetic_constrained_marl", n_agents=3, budget=5.0)
    env = build_env(cfg, rollout_length=16)
    assert isinstance(env, SyntheticConstrainedMarlEnv)


@pytest.mark.parametrize("name", ["manyagent_ant", "halfcheetah_2x3", "safety_gym_nav"])
def test_build_env_raises_for_unimplemented_environments(name):
    cfg = EnvConfig(name=name, n_agents=6, budget=25.0)
    with pytest.raises(NotImplementedError, match="not implemented"):
        build_env(cfg, rollout_length=2000)


def test_experiment_run_construction_raises_immediately_for_manyagent_ant():
    """The actual regression: ExperimentRun(cfg) must raise here, not
    silently construct a SyntheticConstrainedMarlEnv and proceed to train
    on it for hours before anyone notices the mismatch."""
    cfg = ExperimentConfig(
        run_id="regression_test",
        env={"name": "manyagent_ant", "n_agents": 6, "budget": 25.0},
        topology={"name": "ring", "n_agents": 6},
        sources=default_m7_sources(),
        total_steps=500_000,
        rollout_length=2000,
    )
    with pytest.raises(NotImplementedError, match="manyagent_ant"):
        ExperimentRun(cfg)
