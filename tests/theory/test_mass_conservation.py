"""Smoke tests S26 / Theory Test A: Theorem 1, open-loop mass conservation.

PROJECT_REPORT.md §R3.1: required topologies (complete, ring, star,
erdos_renyi x3 configs, identity, shared_constraint) and required delta
schedules (single-agent persistent, single-agent selective, multi-agent
mixed-sign, zero-sum). Tolerance 1e-10 at N<=24, K<=2000, eta=0.035.
"""

from __future__ import annotations

import numpy as np
import pytest

from safelie.consensus.topologies import build_topology
from safelie.theory.mass_conservation import verify_mass_conservation

N = 6
ETA = 0.035
K = 500
TOL = 1e-10


def _topologies(n: int) -> dict[str, np.ndarray]:
    return {
        "complete": build_topology("complete", n),
        "ring": build_topology("ring", n),
        "star": build_topology("star", n),
        "erdos_renyi_p03": build_topology("erdos_renyi", n, p=0.3, graph_seed=1),
        "erdos_renyi_p05": build_topology("erdos_renyi", n, p=0.5, graph_seed=2),
        "erdos_renyi_p08": build_topology("erdos_renyi", n, p=0.8, graph_seed=3),
        "identity": build_topology("identity", n),
        "shared_constraint": build_topology("shared_constraint", n),
    }


def _persistent_single_agent(n: int, k: int, delta: float = 1.5, agent: int = 0) -> list[np.ndarray]:
    v = np.zeros(n)
    v[agent] = delta
    return [v.copy() for _ in range(k)]


def _selective_single_agent(n: int, k: int, delta: float = 1.5, agent: int = 0) -> list[np.ndarray]:
    schedule = []
    for t in range(k):
        v = np.zeros(n)
        if t % 3 == 0:  # corrupts one in three rounds
            v[agent] = delta
        schedule.append(v)
    return schedule


def _multi_agent_mixed_sign(n: int, k: int) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    signs = rng.choice([-1.0, 1.0], size=n)
    magnitudes = rng.uniform(0.5, 2.0, size=n)
    v = signs * magnitudes
    return [v.copy() for _ in range(k)]


def _zero_sum_schedule(n: int, k: int) -> list[np.ndarray]:
    """Total injected mass is exactly zero — a sign-error trap."""
    rng = np.random.default_rng(7)
    schedule = []
    for _ in range(k):
        v = rng.normal(size=n)
        v -= v.mean()  # exactly zero-sum every round
        schedule.append(v)
    return schedule


@pytest.mark.parametrize(
    "schedule_fn",
    [_persistent_single_agent, _selective_single_agent, _multi_agent_mixed_sign, _zero_sum_schedule],
)
def test_mass_conservation_holds_to_machine_precision(schedule_fn):
    topologies = _topologies(N)
    delta_schedule = schedule_fn(N, K)
    result = verify_mass_conservation(topologies, delta_schedule, ETA, tol=TOL)
    assert result.passed, (
        f"Theorem 1 open-loop check failed for {schedule_fn.__name__}: "
        f"max_abs_deviation={result.max_abs_deviation:.3e}, "
        f"cross_topology_spread={result.cross_topology_spread:.3e}"
    )


def test_zero_sum_schedule_gives_exactly_zero_mass():
    """A sign-error trap: if total injected mass is zero, e_K.sum() must be zero
    for every topology, not merely 'small'."""
    topologies = _topologies(N)
    delta_schedule = _zero_sum_schedule(N, K)
    result = verify_mass_conservation(topologies, delta_schedule, ETA, tol=TOL)
    assert abs(result.expected) < TOL
    for name, mass in result.masses.items():
        assert abs(mass) < TOL, f"topology {name}: expected zero mass, got {mass:.3e}"


def test_large_n_and_k_still_holds():
    """Larger scale check: N=24, K=2000, per §R3.1's stated tolerance regime."""
    n, k = 24, 2000
    topologies = {
        "complete": build_topology("complete", n),
        "ring": build_topology("ring", n),
        "identity": build_topology("identity", n),
    }
    delta_schedule = _persistent_single_agent(n, k, delta=0.7, agent=3)
    result = verify_mass_conservation(topologies, delta_schedule, ETA, tol=TOL)
    assert result.passed
