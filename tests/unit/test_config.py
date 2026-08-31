"""Smoke tests S2 (config validation) and S5 (M <= 2f rejected)."""

from __future__ import annotations

import pytest
from safelie.sources.registry import default_m7_sources
from safelie.utils.config import ExperimentConfig, load_experiment_config
from pydantic import ValidationError


def _base_kwargs(**overrides):
    kwargs = dict(
        run_id="test",
        env={"name": "synthetic_constrained_marl", "n_agents": 6, "budget": 25.0},
        topology={"name": "ring", "n_agents": 6},
        sources=default_m7_sources(),
        total_steps=1000,
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_config_constructs():
    cfg = ExperimentConfig(**_base_kwargs())
    assert cfg.env.n_agents == 6


def test_malformed_config_rejected_before_any_environment_is_built():
    with pytest.raises(ValidationError):
        ExperimentConfig(**_base_kwargs(env={"name": "not_a_real_env", "n_agents": 6, "budget": 25.0}))


def test_negative_budget_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig(**_base_kwargs(env={"name": "synthetic_constrained_marl", "n_agents": 6, "budget": -1.0}))


def test_topology_n_agents_mismatch_rejected():
    with pytest.raises(ValueError, match="n_agents"):
        ExperimentConfig(**_base_kwargs(topology={"name": "ring", "n_agents": 4}))


def test_rce_m7_f4_rejected_at_config_time():
    """M=7, f=4: M - 2f = -1. Must raise at construction, matching Table
    4's void operating point (PROJECT_REPORT.md §13.2, W2)."""
    with pytest.raises(ValueError, match="effective_M"):
        ExperimentConfig(**_base_kwargs(defense={"name": "rce", "f": 4, "beta": 1.5}))


def test_rce_m7_f1_accepted_at_config_time():
    cfg = ExperimentConfig(**_base_kwargs(defense={"name": "rce", "f": 1, "beta": 1.5}))
    assert cfg.defense.f == 1


def test_erdos_renyi_topology_requires_p():
    with pytest.raises(ValidationError, match="topology.p"):
        ExperimentConfig(**_base_kwargs(topology={"name": "erdos_renyi", "n_agents": 6}))


def test_attack_f_cannot_exceed_source_count():
    with pytest.raises(ValueError, match="cannot exceed"):
        ExperimentConfig(**_base_kwargs(attack={"name": "primary", "f": 99}))


def test_load_from_yaml_file(tmp_path):
    yaml_content = """
run_id: yaml_test
seed: 1
env:
  name: synthetic_constrained_marl
  n_agents: 3
  budget: 25.0
topology:
  name: ring
  n_agents: 3
sources:
  sources:
    - source_id: own_critic
      source_type: own_critic
      independence_class: ic_own
    - source_id: peer_critic_1
      source_type: peer_critic
      independence_class: ic_peer1
    - source_id: monitor_1
      source_type: monitor
      independence_class: ic_mon1
total_steps: 1000
"""
    path = tmp_path / "config.yaml"
    path.write_text(yaml_content, encoding="utf-8")
    cfg = load_experiment_config(str(path))
    assert cfg.run_id == "yaml_test"
    assert cfg.env.n_agents == 3


def test_load_malformed_yaml_file_raises(tmp_path):
    path = tmp_path / "bad_config.yaml"
    path.write_text("run_id: bad\nenv:\n  name: not_real\n  n_agents: 3\n  budget: 25.0\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_experiment_config(str(path))
