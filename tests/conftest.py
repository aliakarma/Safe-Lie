"""Shared tiny experiment config for the smoke suite."""

from __future__ import annotations

import pytest

from safelie.utils.config import ExperimentConfig


def make_tiny_config(tmp_path, run_id: str = "tiny", attack_name: str = "none", defense_name: str = "mean", f: int = 0, seed: int = 0) -> ExperimentConfig:
    return ExperimentConfig(
        run_id=run_id,
        seed=seed,
        env={
            "name": "synthetic_constrained_marl",
            "n_agents": 3,
            "budget": 25.0,
            "horizon": 16,
            "obs_dim": 6,
            "action_dim": 2,
        },
        topology={"name": "ring", "n_agents": 3},
        sources={
            "sources": [
                {"source_id": "own_critic", "source_type": "own_critic", "independence_class": "ic_own"},
                {"source_id": "peer_critic_1", "source_type": "peer_critic", "independence_class": "ic_peer1"},
                {"source_id": "monitor_1", "source_type": "monitor", "independence_class": "ic_mon1"},
            ]
        },
        attack={"name": attack_name, "f": f, "budget_ratio": 0.5},
        defense={"name": defense_name, "f": f, "beta": 1.5},
        ppo={"epochs": 1, "minibatches": 2, "hidden_dim": 16},
        total_steps=32,
        rollout_length=16,
        output_dir=str(tmp_path / "runs"),
    )


@pytest.fixture
def tiny_config_factory(tmp_path):
    def _make(**kwargs):
        return make_tiny_config(tmp_path, **kwargs)

    return _make
