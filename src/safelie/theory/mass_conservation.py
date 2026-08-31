"""Numerical verification of Theorem 1 (corruption mass conservation).

Report reference: PROJECT_REPORT.md §R3.1 (Test A), §6.4.

Theorem 1 (main_iclr.tex, Appendix A.1): under the open-loop primal-dual
recursion, the dual-bias error obeys

    e_{k+1} = W e_k + eta_lambda * delta_k,   e_0 = 0

and its aggregate satisfies

    1^T e_K = eta_lambda * sum_{k=0}^{K-1} 1^T delta_k

independent of W, for any doubly stochastic mixing matrix. This module
tests exactly that recursion, with no environment, policy, or critic
involved — it is a two-line consequence of 1^T W = 1^T and should hold to
machine precision. A failure here indicates a bug in W's construction
(see `safelie.consensus.mixing.assert_doubly_stochastic`), not a refutation
of the theorem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from safelie.consensus.mixing import assert_doubly_stochastic


def simulate_dual_bias(W: np.ndarray, delta_schedule: list[np.ndarray], eta: float) -> np.ndarray:
    """Unroll e_{k+1} = W e_k + eta * delta_k from e_0 = 0. Returns e_K."""
    n = W.shape[0]
    e = np.zeros(n)
    for delta_k in delta_schedule:
        e = W @ e + eta * delta_k
    return e


@dataclass
class MassConservationResult:
    masses: dict[str, float]
    expected: float
    max_abs_deviation: float
    cross_topology_spread: float
    passed: bool


def verify_mass_conservation(
    topologies: dict[str, np.ndarray],
    delta_schedule: list[np.ndarray],
    eta: float,
    tol: float = 1e-10,
) -> MassConservationResult:
    """Theorem 1, open loop. Two assertions, not one.

    (a) For every topology, the simulated aggregate mass matches the
        closed form ``eta * sum_k 1^T delta_k`` to `tol`.
    (b) The resulting mass is identical across *all* topologies to `tol` —
        this is the content of "independent of W".
    """
    expected = eta * sum(float(d.sum()) for d in delta_schedule)

    masses: dict[str, float] = {}
    max_abs_deviation = 0.0
    for name, W in topologies.items():
        assert_doubly_stochastic(W)
        e_K = simulate_dual_bias(W, delta_schedule, eta)
        mass = float(e_K.sum())
        masses[name] = mass
        max_abs_deviation = max(max_abs_deviation, abs(mass - expected))

    cross_topology_spread = max(masses.values()) - min(masses.values())
    passed = (max_abs_deviation < tol) and (cross_topology_spread < tol)

    return MassConservationResult(
        masses=masses,
        expected=expected,
        max_abs_deviation=max_abs_deviation,
        cross_topology_spread=cross_topology_spread,
        passed=passed,
    )
