"""Krum aggregation (Table 4 comparison aggregator), scalar case.

Report reference: PROJECT_REPORT.md §7 — borrowed from Blanchard et al.
(2017), explicit debt acknowledged rather than presented as new.

Krum selects the single candidate value whose sum of squared distances to
its closest `M - f - 2` neighbours is smallest — the most "clustered"
value, which is robust to up to `f` arbitrary outliers as long as
`M >= f + 3` (at least one neighbour must remain in the scoring set).
"""

from __future__ import annotations

import numpy as np

from safelie.defenses.base import AggregateResult, mad


def krum_aggregator(values: np.ndarray, f: int) -> AggregateResult:
    values = np.asarray(values, dtype=float)
    m = len(values)
    if m < f + 3:
        raise ValueError(
            f"Krum requires M >= f + 3 (at least one neighbour in the "
            f"scoring set), got M={m}, f={f}"
        )
    n_neighbors = m - f - 2

    scores = np.zeros(m)
    for i in range(m):
        dists_sq = (values - values[i]) ** 2
        dists_sq[i] = np.inf  # exclude self
        nearest = np.sort(dists_sq)[:n_neighbors]
        scores[i] = nearest.sum()

    selected = int(np.argmin(scores))
    order = np.argsort(np.abs(values - values[selected]))
    neighborhood = values[order[: n_neighbors + 1]]  # selected value + its neighbours

    return AggregateResult(
        point_estimate=float(values[selected]),
        retained_values=neighborhood,
        spread=mad(neighborhood),
        degenerate=(n_neighbors < 2),
    )
