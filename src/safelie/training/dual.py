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
