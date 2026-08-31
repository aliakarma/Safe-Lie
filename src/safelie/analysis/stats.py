"""Welch's t-test with Holm correction across attack conditions.

Report reference: main_iclr.tex §5.1 — "Five seeds per cell, mean +/- SD,
with Welch's t-test for the primary comparison and Holm correction across
the attack conditions." PROJECT_REPORT.md §R2.3 / decision D6: at n=3
(the compact pilot) this procedure manufactures false precision and must
NOT be used; it applies from >= 5 seeds (Stage 3) onward. Callers are
responsible for enforcing that minimum — this module does not silently
run on too few samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

MIN_SEEDS_FOR_INFERENCE = 5  # [SPEC] the paper's own seed count; decision D6


@dataclass(frozen=True)
class WelchResult:
    statistic: float
    p_value: float
    mean_a: float
    mean_b: float
    n_a: int
    n_b: int


def welch_t_test(sample_a: np.ndarray, sample_b: np.ndarray) -> WelchResult:
    sample_a, sample_b = np.asarray(sample_a, dtype=float), np.asarray(sample_b, dtype=float)
    if len(sample_a) < MIN_SEEDS_FOR_INFERENCE or len(sample_b) < MIN_SEEDS_FOR_INFERENCE:
        raise ValueError(
            f"Welch's t-test requires >= {MIN_SEEDS_FOR_INFERENCE} seeds per "
            f"arm (the paper's own count); got {len(sample_a)} and "
            f"{len(sample_b)}. At fewer seeds (e.g. the 3-seed compact "
            f"pilot), report per-seed sign/ordering instead "
            f"(safelie.eval.protocol.pilot_seed_summary) -- "
            f"PROJECT_REPORT.md §R2.3, decision D6."
        )
    t_stat, p_value = scipy_stats.ttest_ind(sample_a, sample_b, equal_var=False)
    return WelchResult(
        statistic=float(t_stat),
        p_value=float(p_value),
        mean_a=float(sample_a.mean()),
        mean_b=float(sample_b.mean()),
        n_a=len(sample_a),
        n_b=len(sample_b),
    )


def holm_correction(p_values: list[float]) -> list[float]:
    """Holm step-down correction, one adjusted p-value per input, in the
    same order as the input (not re-sorted)."""
    if not p_values:
        return []
    _, adjusted, _, _ = multipletests(p_values, method="holm")
    return list(adjusted)
