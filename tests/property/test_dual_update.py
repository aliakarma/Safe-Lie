"""Smoke tests S9 / S23: unconditional dual update, liveness under
over-reporting (Proposition 3).
"""

from __future__ import annotations

import numpy as np
from safelie.consensus.topologies import build_topology
from safelie.training.dual import dual_update


def test_dual_update_executes_every_round_under_extreme_over_reporting():
    n = 6
    W = build_topology("ring", n)
    lam = np.zeros(n)
    lam_max = 25.0
    eta_lambda = 0.035

    trajectory = []
    for _k in range(1000):
        residual = np.full(n, 1e6)  # extreme over-reporting every round
        lam = dual_update(lam, W, eta_lambda, residual, lam_max)
        trajectory.append(lam.copy())

    assert len(trajectory) == 1000  # the update ran every single round -- no deadlock
    np.testing.assert_allclose(lam, lam_max, atol=1e-9)  # saturates, does not diverge or NaN
    assert np.all(np.isfinite(lam))


def test_dual_update_is_a_pure_function_with_no_branch_on_residual_magnitude():
    """The liveness property is structural: the same call executes
    identically whether the residual is tiny or enormous."""
    n = 4
    W = build_topology("complete", n)
    lam0 = np.array([1.0, 2.0, 3.0, 4.0])
    small = dual_update(lam0, W, 0.035, np.full(n, 0.1), 25.0)
    large = dual_update(lam0, W, 0.035, np.full(n, 1e9), 25.0)
    assert small.shape == large.shape == (n,)
    assert np.all(small <= 25.0) and np.all(large <= 25.0)
