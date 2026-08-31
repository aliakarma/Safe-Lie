"""Shared aggregator interface and result type.

Report reference: Phase 6, PROJECT_REPORT.md §7.1; smoke tests S6-S8, S21-S25.

All quantities here are on the **return scale**, commensurate with the
budget d^i — never on the per-step cost scale (§4.2 of the paper, quoted
in the report as the single most likely reimplementation bug). Callers
are responsible for feeding return-scale estimates in; this module does
not know about steps or discounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


def mad(values: np.ndarray) -> float:
    """Median absolute deviation, unscaled.

    `[DECISION]` The paper specifies "MAD" without a consistency-constant
    convention. We use the raw (unscaled) MAD rather than the
    normal-consistent 1.4826*MAD, documented in docs/assumptions.md. This
    only rescales the margin's effective beta; it does not change any
    qualitative conclusion in this repository.
    """
    if len(values) == 0:
        return 0.0
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


@dataclass
class AggregateResult:
    """Output of one aggregator call for one constraint, one round.

    point_estimate: the robust location estimate (hat J_C for RCE/TrimMean).
    retained_values: the values that survived aggregation (all of them for
        mean/median; the trimmed set for TrimMean/RCE; the selected value's
        neighbourhood for Krum).
    retained_n: len(retained_values).
    spread: MAD over `retained_values` (informational for mean/median,
        load-bearing for RCE).
    degenerate: True if `retained_n` was too small for `spread` to be a
        meaningful dispersion estimate (S8). Never silently coerced to a
        confident-looking number without this flag being set.
    """

    point_estimate: float
    retained_values: np.ndarray
    spread: float
    degenerate: bool = False
    retained_n: int = field(init=False)

    def __post_init__(self) -> None:
        self.retained_n = len(self.retained_values)


class Aggregator(Protocol):
    def __call__(self, values: np.ndarray, f: int) -> AggregateResult: ...
