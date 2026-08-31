"""The adaptive stealth attack.

Report reference: main_iclr.tex §7.4 App. B (quoted formula);
PROJECT_REPORT.md §7.2 — kappa and nu are `[GAP]` G7, no values given
anywhere in the paper. This repository sweeps
kappa in {0.5, 1.0, 2.0}, nu/d in {0.02, 0.05, 0.10} as an `[INFERRED]`
starting grid (recommended by the report) and defaults to kappa=1.0,
nu/d=0.05.

    delta^i_k = -min(B, kappa * max(0, J_hat^i_C - d^i + nu))

holds the *reported* residual just below the budget by a margin nu,
maximizing stealth subject to the per-round budget B. This attack is
adaptive: it is a function of the learner's current cost estimate, not a
fixed schedule.

Not part of the Stage-2 compact matrix (§R2.1: persistent static attack
only); implemented here so it drops in as a config change for Stage-3
item 11.
"""

from __future__ import annotations


def stealth_attack(
    residuals: dict[str, float],
    corrupted: set[str],
    k: int,
    B: float,
    kappa: float,
    nu: float,
    d: float,
) -> dict[str, float]:
    out = dict(residuals)
    for sid in corrupted:
        if sid not in out:
            continue
        g = out[sid]
        j_hat = g + d  # residual back to return scale
        delta = -min(B, kappa * max(0.0, j_hat - d + nu))
        out[sid] = g + delta
    return out
