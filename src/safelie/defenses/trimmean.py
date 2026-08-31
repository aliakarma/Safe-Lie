"""Trimmed-mean aggregation (Algorithm 1, lines 5-6, without the margin).

Report reference: main_iclr.tex Algorithm 1 lines 5-6; PROJECT_REPORT.md
§13.2 (W2), §R10.2 (D3). This is the "TrimMean" row of Table 4 — RCE
(`safelie.defenses.rce`) adds the pessimism margin on top of this.

Two defects the paper's own Table 4 reports without resolving are fixed
here by raising rather than silently degrading (decision D3):

  - M <= 2f: discarding f largest and f smallest would remove all or more
    than all values. This is not "an edge case to tune around" — it is
    the algorithm being asked to average over a negative-size set. Raise
    at call time, matching the config-time raise in
    `safelie.utils.config.ExperimentConfig`.
  - |T| < 3 (e.g. M=7, f=2 leaves 3; M=7, f=3 leaves 1): the retained set
    is technically non-empty but its MAD is either extremely noisy (3
    points) or identically zero (1 point). The trimmed mean itself is
    still well-defined and returned; callers that need a *margin* on top
    of it (`safelie.defenses.rce`) are responsible for flooring/warning,
    since the raw trimmed mean alone makes no dispersion claim.
"""

from __future__ import annotations

import numpy as np

from safelie.defenses.base import AggregateResult, mad


def trimmed_mean_aggregator(values: np.ndarray, f: int) -> AggregateResult:
    values = np.asarray(values, dtype=float)
    m = len(values)
    if m <= 2 * f:
        raise ValueError(
            f"Trimmed mean requires M > 2f, got M={m}, f={f} (M - 2f = "
            f"{m - 2 * f} <= 0). Discarding f largest and f smallest would "
            f"remove all values or more; this is mathematically undefined, "
            f"not a degenerate-but-valid case (PROJECT_REPORT.md §13.2, W2)."
        )
    order = np.argsort(values)
    retained_idx = order[f : m - f]
    retained = values[retained_idx]
    return AggregateResult(
        point_estimate=float(retained.mean()),
        retained_values=retained,
        spread=mad(retained),
        degenerate=(len(retained) < 3),
    )
