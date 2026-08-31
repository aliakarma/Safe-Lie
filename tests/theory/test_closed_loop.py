"""Smoke test S28 / Theory Test C: the closed-loop synthetic diagnostic.

PROJECT_REPORT.md §R3.3: this test must *run and record a rho value* —
the gate does not require a particular outcome, since a large rho is a
finding about the paper's central theorem (W1), not a bug.
"""

from __future__ import annotations

import numpy as np
from safelie.consensus.topologies import build_topology
from safelie.theory.closed_loop import run_closed_loop_diagnostic

N = 6


def _topologies(n: int) -> dict[str, np.ndarray]:
    return {
        "complete": build_topology("complete", n),
        "ring": build_topology("ring", n),
        "star": build_topology("star", n),
    }


def test_homogeneous_gain_is_exactly_invariant():
    """With c_i = c for all i, phi commutes with W and 1^T e_K stays exactly
    invariant — the report's stated vacuous-pass case, included so the
    diagnostic's own claim ('heterogeneity is what breaks invariance') is
    itself falsifiable."""
    c = np.full(N, 0.02)
    result = run_closed_loop_diagnostic(
        _topologies(N), N, K=300, eta=0.035, c=c, d=25.0, delta=2.0, corrupted_agent=0
    )
    assert result.gain_heterogeneity == 0.0
    assert result.rho < 1e-6, f"homogeneous gain should give rho ~ 0, got {result.rho}"


def test_heterogeneous_gain_runs_and_records_rho():
    """The diagnostic must execute and produce a recorded rho — whatever it
    says. This is a measurement, not an assertion about a specific value."""
    rng = np.random.default_rng(0)
    c = rng.uniform(0.005, 0.05, size=N)
    result = run_closed_loop_diagnostic(
        _topologies(N), N, K=300, eta=0.035, c=c, d=25.0, delta=2.0, corrupted_agent=0
    )
    assert result.gain_heterogeneity > 0.0
    assert np.isfinite(result.rho)
    assert result.rho >= 0.0
    assert set(result.aggregate_bias) == set(_topologies(N))


def test_rho_is_reproducible_given_fixed_inputs():
    """Determinism: the diagnostic is a pure function of its inputs."""
    c = np.array([0.01, 0.02, 0.015, 0.03, 0.005, 0.025])
    kwargs = dict(K=300, eta=0.035, c=c, d=25.0, delta=2.0, corrupted_agent=1)
    r1 = run_closed_loop_diagnostic(_topologies(N), N, **kwargs)
    r2 = run_closed_loop_diagnostic(_topologies(N), N, **kwargs)
    assert r1.rho == r2.rho
    assert r1.aggregate_bias == r2.aggregate_bias
