"""The three margin quantities, kept separate (decision D9).

Report reference: PROJECT_REPORT.md §R10.4 — beta*sigma (empirical
conservatism, measured every round), epsilon(M,f,alpha) (the theoretical
bound, uncomputable at runtime since it depends on the unknown
sub-Gaussian parameter varsigma), and an offline-calibrated
epsilon_hat_offline used as a *reference*, never a proof, that the
condition beta*sigma >= epsilon holds.

**No runtime safety guarantee is claimed by this module.**
`guarantee_in_force` is an estimate against an offline calibration, not a
verification of Theorem 2's precondition -- the paper's own analysis
(§13.3, W3) explains why that precondition cannot be checked without
ground truth the deployment lacks by construction.
"""

from __future__ import annotations

import numpy as np


def calibrate_epsilon_offline(clean_run_disagreements: list[float], alpha: float = 0.05) -> float:
    """Offline calibration of an epsilon_hat reference from a clean
    (no-attack) run's observed source disagreement (MAD values). This is
    a `[INFERRED]` addition the paper does not specify a procedure for;
    §13.3 recommends exactly this pattern (use a benign pre-deployment
    phase). We take a high quantile of the clean-run MAD as the
    reference, which is deliberately conservative rather than a precise
    estimate of the unknown sub-Gaussian parameter varsigma.
    """
    if not clean_run_disagreements:
        return 0.0
    quantile = 1.0 - alpha
    return float(np.quantile(clean_run_disagreements, quantile))


def compute_guarantee_in_force(applied_margin: float, epsilon_offline: float) -> bool:
    """Whether the *empirical* margin (beta*sigma, this round) meets or
    exceeds the *offline-calibrated reference* epsilon_hat_offline.

    This is NOT a proof that Theorem 2's precondition holds -- it is the
    best available runtime estimate of it, logged per round so a
    deployment can alarm when it is violated (PROJECT_REPORT.md §15.2,
    `/v1/health/margin`).
    """
    return applied_margin >= epsilon_offline
