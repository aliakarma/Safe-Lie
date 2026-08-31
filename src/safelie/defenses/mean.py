"""Plain mean aggregation — the unprotected control (Table 4 'Mean' row).

Report reference: PROJECT_REPORT.md §9.3 — "plain averaging offers
essentially no protection ... since averaging is exactly the operation
conserving corruption mass" (Theorem 1). Smoke test S21 asserts this has
no breakdown point: a single corrupted value of unbounded magnitude
corrupts the mean without limit.
"""

from __future__ import annotations

import numpy as np

from safelie.defenses.base import AggregateResult, mad


def mean_aggregator(values: np.ndarray, f: int = 0) -> AggregateResult:
    """Unweighted mean of all M values. `f` is accepted for interface
    compatibility and ignored — mean rejects nothing."""
    values = np.asarray(values, dtype=float)
    return AggregateResult(
        point_estimate=float(values.mean()),
        retained_values=values,
        spread=mad(values),
        degenerate=False,
    )
