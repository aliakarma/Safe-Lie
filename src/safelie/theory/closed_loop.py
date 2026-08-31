"""Closed-loop synthetic diagnostic for Theorem 1 (Test C).

Report reference: PROJECT_REPORT.md §R3.3, §13.1 (weakness W1).

Theorem 1's proof assumes both the corrupted and uncorrupted trajectories
share the same primal sequence {theta_k} and hence the same residuals g_k
("by construction"). The real closed-loop system violates this: a biased
multiplier changes the policy, which changes J_C, which changes g_k. This
module does **not** run RL — it replaces the policy response with the
cheapest model that has the right qualitative structure (a linear
feedback from the multiplier back onto the residual, with a physical
floor on cost) and measures how much topology-invariance degrades once
the loop is closed.

    g_k = g0_k + phi(lambda_k),      phi_i(lambda) = -c_i * lambda_i
    g_k >= -d                        (cost cannot go below zero)

With a *homogeneous* gain (c_i = c for all i), phi commutes with W and
1^T e_K stays exactly invariant — the test would pass vacuously. Breaking
that requires per-agent heterogeneous gains c_i, which is why `c` here is
a per-agent array, not a scalar.

This is reported as a **diagnostic**, not a finding about MAPPO-Lagrangian:
the response model is a caricature and no policy, environment, or critic
is involved. It answers a narrower, but still useful, question: is the
open-loop invariance of Theorem 1 fragile once *any* closed loop with the
right sign structure is introduced?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from safelie.consensus.mixing import assert_doubly_stochastic


def _project(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def _simulate_closed_loop(
    W: np.ndarray,
    g0_schedule: list[np.ndarray],
    delta_schedule: list[np.ndarray],
    c: np.ndarray,
    d: np.ndarray,
    eta: float,
    lambda_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the corrupted and uncorrupted closed-loop dual recursions.

    Returns (lambda_K, lambda_circ_K).
    """
    n = W.shape[0]
    lam = np.zeros(n)
    lam_circ = np.zeros(n)
    for g0_k, delta_k in zip(g0_schedule, delta_schedule, strict=False):
        g_k = np.maximum(g0_k - c * lam, -d)
        g_circ_k = np.maximum(g0_k - c * lam_circ, -d)

        lam = _project(W @ lam + eta * (g_k + delta_k), 0.0, lambda_max)
        lam_circ = _project(W @ lam_circ + eta * g_circ_k, 0.0, lambda_max)
    return lam, lam_circ


@dataclass
class ClosedLoopResult:
    aggregate_bias: dict[str, float]  # per topology: 1^T e_K
    rho: float  # relative spread across topologies
    mean_gain: float
    gain_heterogeneity: float
    K: int
    open_loop_reference: float = field(repr=False, default=0.0)


def run_closed_loop_diagnostic(
    topologies: dict[str, np.ndarray],
    n_agents: int,
    K: int,
    eta: float,
    c: np.ndarray,
    d: np.ndarray | float = 25.0,
    delta: float = 1.0,
    corrupted_agent: int = 0,
    g0_scale: float = 0.0,
    lambda_max: float = 25.0,
    rng_seed: int = 0,
) -> ClosedLoopResult:
    """Measure how much topology-invariance degrades once the loop closes.

    A persistent single-agent attack (matching Proposition 1's setup) is
    injected while a linear per-agent feedback `c` (heterogeneous across
    agents) and a cost floor at ``-d`` are active. `rho` is the relative
    spread of the aggregate bias ``|1^T e_K|`` across the supplied
    topologies: 0 means Theorem 1's open-loop invariance survived exactly;
    a large value means it did not.
    """
    c = np.asarray(c, dtype=float)
    assert c.shape == (n_agents,), f"c must have shape ({n_agents},), got {c.shape}"
    d_arr = np.full(n_agents, float(d)) if isinstance(d, int | float) else np.asarray(d, dtype=float)

    rng = np.random.default_rng(rng_seed)
    if g0_scale > 0:
        g0_schedule = [rng.normal(0.0, g0_scale, size=n_agents) for _ in range(K)]
    else:
        g0_schedule = [np.zeros(n_agents) for _ in range(K)]

    delta_vec = np.zeros(n_agents)
    delta_vec[corrupted_agent] = delta
    delta_schedule = [delta_vec for _ in range(K)]

    aggregate_bias: dict[str, float] = {}
    for name, W in topologies.items():
        assert_doubly_stochastic(W)
        lam_K, lam_circ_K = _simulate_closed_loop(
            W, g0_schedule, delta_schedule, c, d_arr, eta, lambda_max
        )
        e_K = lam_K - lam_circ_K
        aggregate_bias[name] = float(e_K.sum())

    abs_biases = [abs(v) for v in aggregate_bias.values()]
    lo, hi = min(abs_biases), max(abs_biases)
    denom = 0.5 * (hi + lo)
    rho = 0.0 if denom < 1e-12 else (hi - lo) / denom

    return ClosedLoopResult(
        aggregate_bias=aggregate_bias,
        rho=rho,
        mean_gain=float(c.mean()),
        gain_heterogeneity=float(c.std()),
        K=K,
        open_loop_reference=eta * delta * K,
    )
