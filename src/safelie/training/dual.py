"""The unconditional, projected dual (multiplier) update.

Report reference: main_iclr.tex Eq. 2 / Algorithm 1 line 8;
PROJECT_REPORT.md §6.8, Proposition 3 (liveness) — "The dual update must
be unconditional. Any `if` statement guarding line 7 reintroduces the
deadlock." This function contains no gate of any kind, by construction:
it is a pure array computation with no branch that could skip it.
"""

from __future__ import annotations

import numpy as np


def dual_update(lam: np.ndarray, W: np.ndarray, eta_lambda: float, residual: np.ndarray, lam_max: float) -> np.ndarray:
    """lambda_{k+1} = Proj_[0, lam_max]( W @ lambda_k + eta_lambda * residual )

    `residual` is J_bar - d (or J_hat - d for undefended aggregators) per
    agent — the pessimistic, return-scale quantity produced by
    `safelie.defenses`. This function performs no attack-awareness, no
    gating, and no branching on the magnitude of `residual`: liveness
    (Proposition 3) is a structural property of this function having
    exactly one code path.
    """
    mixed = W @ lam
    updated = mixed + eta_lambda * residual
    return np.clip(updated, 0.0, lam_max)


def pid_dual_update(
    integral: np.ndarray,
    prev_residual: np.ndarray,
    W: np.ndarray,
    k_p: float,
    k_i: float,
    k_d: float,
    residual: np.ndarray,
    lam_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """PID-Lagrangian multiplier update (Stooke et al., 2020), decentralized.

        Delta_k   = residual                      (= J_bar - d, per agent)
        d_k       = max(0, Delta_k - Delta_{k-1})  one-sided, elementwise
        I_{k+1}   = Proj_[0, lam_max]( W @ I_k + k_i * Delta_k )
        lambda_k  = Proj_[0, lam_max]( k_p * Delta_k + I_{k+1} + k_d * d_k )

    Returns `(lambda_k, I_{k+1})`.

    **Consensus mixes the integral only.** `I` is the accumulating state
    that Eq. 2's consensus step is about, so `W` is applied there and the
    proportional and derivative terms are each agent's local response to
    its own current residual. Two consequences worth stating, because both
    are the reason this placement was chosen:

    * **It reduces exactly.** At `k_p = k_d = 0` and `k_i = eta_lambda`,
      `I` and `lambda` coincide and the recursion above is
      `dual_update` term for term, bit for bit. Every existing config --
      A1's, A2's, A3's twelve -- therefore runs an unchanged code path.
      `tests/unit/test_pid_dual.py::test_pid_reduces_to_lagrangian_bitwise`
      pins this against `dual_update` itself rather than against a
      transcription of it.
    * **Theorem 1 carries over to the integral path, and only to it.**
      `1^T e_K = k_i * sum_k 1^T delta_k` still holds for the `I`
      recursion, because `1^T W = 1^T` is all that proof uses. The
      proportional and derivative terms add a bias that does *not*
      accumulate in `K` -- which is precisely the transfer-function change
      that makes PID-Lagrangian a baseline worth running here
      (main_iclr.tex sec. 2: "by changing how cost error propagates into
      the multiplier it alters the attack's transfer function").

    The derivative is taken on the cost and one-sided, both as in Stooke et
    al. Because `d` is a constant, `J_C,k - J_C,k-1 == Delta_k -
    Delta_{k-1}`, so differencing the residual is the same quantity; the
    one-sided `max(0, .)` keeps a *falling* cost from discounting the
    multiplier, which is the behaviour the derivative term exists to avoid.

    Liveness (Proposition 3) is preserved structurally, on the same
    argument as `dual_update`: every operation here is elementwise
    arithmetic, `np.maximum` or `np.clip`. There is no branch of any kind,
    so no path through this function can skip the update, and none can
    depend on the magnitude of `residual`.
    """
    derivative = np.maximum(residual - prev_residual, 0.0)
    new_integral = np.clip(W @ integral + k_i * residual, 0.0, lam_max)
    lam = np.clip(k_p * residual + new_integral + k_d * derivative, 0.0, lam_max)
    return lam, new_integral
