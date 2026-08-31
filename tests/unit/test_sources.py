"""Smoke test S12: effective_M counts independence classes, not report count.

Report reference: PROJECT_REPORT.md §13.4 (W4), §R10.3 (D10).
"""

from __future__ import annotations

import pytest

from safelie.sources.registry import (
    SourceRegistry,
    default_m5_sources,
    default_m7_sources,
)
from safelie.utils.config import ExperimentConfig


def test_m7_primary_config_has_effective_m_equal_to_nominal_m():
    sources = default_m7_sources()
    registry = SourceRegistry(sources)
    assert registry.M == 7
    assert registry.effective_M == 7  # every source is its own independence class


def test_m5_two_agent_config_effective_m_is_reduced_by_correlated_replicas():
    """The paper's own N=2, M=5 configuration: 2 peer critics + 3 ensemble
    replicas sharing one process. effective_M must be 3 (2 peers + 1
    replica class), not the nominal 5 — this is W4's accounting made
    concrete."""
    sources = default_m5_sources()
    registry = SourceRegistry(sources)
    assert registry.M == 5
    assert registry.effective_M == 3


def test_config_rejects_rce_when_effective_m_is_too_small_even_if_nominal_m_is_fine():
    """A config claiming M=5 >= 2f+1=5 (f=2) on paper, but whose 5 reports
    span only 3 independence classes, must be rejected: 3 <= 2*2."""
    sources = default_m5_sources()  # nominal M=5, effective_M=3
    with pytest.raises(ValueError, match="effective_M"):
        ExperimentConfig(
            run_id="test",
            env={"name": "synthetic_constrained_marl", "n_agents": 2, "budget": 25.0},
            topology={"name": "ring", "n_agents": 2},
            sources=sources,
            defense={"name": "rce", "f": 2, "beta": 1.5},
            total_steps=1000,
        )
