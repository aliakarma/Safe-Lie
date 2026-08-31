"""Coordinate-wise median aggregation (Table 4 comparison aggregator).

Report reference: PROJECT_REPORT.md §7 (Yin et al. 2018 / Blanchard et al.
2017 aggregation primitives, borrowed explicitly rather than claimed as
novel). For a scalar constraint this reduces to the plain median.
"""

from __future__ import annotations

import numpy as np

from safelie.defenses.base import AggregateResult, mad


def coordinate_median_aggregator(values: np.ndarray, f: int = 0) -> AggregateResult:
    """Scalar median of all M values. `f` is accepted for interface
    compatibility; the median's breakdown point is 50% regardless of `f`."""
    values = np.asarray(values, dtype=float)
    return AggregateResult(
        point_estimate=float(np.median(values)),
        retained_values=values,
        spread=mad(values),
        degenerate=False,
    )
