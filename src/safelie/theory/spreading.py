"""Numerical verification of Proposition 1 (spreading and stealth).

Report reference: PROJECT_REPORT.md §R3.2 (Test B), §6.5.

For a single corrupted agent j injecting a persistent bias delta_k = delta
* e_j, Proposition 1 (main_iclr.tex, Appendix A.2) states

    e_K = eta_lambda * delta * [ (K/N) * 1 + r_K ],   ||r_K||_2 <= 1 / (1 - sigma_2(W))

i.e. every agent's multiplier converges to the *same* bias eta_lambda *
delta * K / N, uniformly, with a transient residual bounded independently
of K. Against a non-communicating learner (W = I, sigma_2 = 1) the same
attack biases only agent j, by eta_lambda * delta * K.

The "stealth" claim made operational: a median-referenced anomaly monitor
that flags any agent deviating from the fleet median by more than tau
catches the W = I case immediately, but under consensus every agent sits
at the same bias, so no agent deviates from the median at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from safelie.consensus.mixing import second_largest_singular_value
from safelie.theory.mass_conservation import simulate_dual_bias


@dataclass
class SpreadingResult:
    e_K: np.ndarray
    uniform_component: np.ndarray
    residual_norm: float
    residual_bound: float
    residual_bound_satisfied: bool
    sigma2: float
    median_deviation: float


def verify_spreading(
    W: np.ndarray,
    delta: float,
    j: int,
    n_agents: int,
    eta: float,
    K: int,
) -> SpreadingResult:
    """Simulate the persistent single-agent attack and check Eq. (spread)."""
    e_j = np.zeros(n_agents)
    e_j[j] = 1.0
    delta_schedule = [delta * e_j for _ in range(K)]

    e_K = simulate_dual_bias(W, delta_schedule, eta)

    uniform_component = (eta * delta * K / n_agents) * np.ones(n_agents)
    r_K = e_K - uniform_component
    residual_norm = float(np.linalg.norm(r_K))

    sigma2 = second_largest_singular_value(W)
    # 1 / (1 - sigma2) diverges as sigma2 -> 1 (W = I); treat that case as
    # an unbounded (vacuous) bound rather than raising a ZeroDivisionError.
    residual_bound = float(1.0 / (1.0 - sigma2)) if sigma2 < 1.0 - 1e-12 else float("inf")

    median_deviation = float(np.max(np.abs(e_K - np.median(e_K))))

    return SpreadingResult(
        e_K=e_K,
        uniform_component=uniform_component,
        residual_norm=residual_norm,
        residual_bound=residual_bound,
        residual_bound_satisfied=residual_norm <= residual_bound + 1e-9,
        sigma2=sigma2,
        median_deviation=median_deviation,
    )
