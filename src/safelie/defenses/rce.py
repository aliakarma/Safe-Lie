"""Robust Constraint Estimation (RCE) — Algorithm 1.

Report reference: main_iclr.tex Algorithm 1; PROJECT_REPORT.md §7.1
(line-by-line notes), §R10.2 (D3), §13.2 (W2), §13.3 (W3).

RCE = TrimMean_f, then MAD over the retained set, then a pessimistic
margin: J_bar = J_hat + beta * sigma. Three load-bearing properties,
verbatim from the paper (§4.2):

  1. Everything here is on the RETURN scale (the caller's responsibility;
     see `safelie.defenses.base`).
  2. The multiplier update this feeds is UNCONDITIONAL — RCE never gates.
     That is enforced in `safelie.training.dual`, not here.
  3. Beta (conservatism knob) and per-source reliability weights are kept
     as SEPARATE roles. Reliability weights (Algorithm 1 lines 2, 10) are
     declared in the paper but never consumed by any line of the
     pseudocode (`[GAP]` G1); this implementation ships them disabled by
     default (decision D12) and does not implement an invented update
     rule as if it were specified.

The single most dangerous bug this module could contain, per the report:
silently returning `spread=0.0` for a degenerate retained set, which turns
RCE into plain trimmed mean while logs still say "RCE". We instead floor
the spread at a configured `sigma_min`, set `degenerate=True`, and require
callers to log that explicitly (S8, S24).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np

from safelie.defenses.base import AggregateResult
from safelie.defenses.trimmean import trimmed_mean_aggregator

logger = logging.getLogger(__name__)


@dataclass
class RceResult(AggregateResult):
    """Extends AggregateResult with the pessimistic estimate.

    `pessimistic_estimate` is J_bar = J_hat + beta * spread — the quantity
    actually fed to the dual update (Algorithm 1 line 7). `applied_margin`
    is beta * spread on its own, logged separately per decision D9 so it
    is never conflated with the theoretical bound epsilon(M,f,alpha)
    (see `safelie.eval.margin.compute_guarantee_in_force`).
    """

    beta: float = 0.0
    applied_margin: float = 0.0
    pessimistic_estimate: float = 0.0


def rce_aggregate(
    values: np.ndarray,
    f: int,
    beta: float,
    sigma_min: float = 1e-3,
    min_retained: int = 3,
) -> RceResult:
    """Algorithm 1, lines 4-7, for one constraint, one round.

    Raises if M <= 2f (delegated to `trimmed_mean_aggregator`; decision
    D3). Floors the spread at `sigma_min` and flags `degenerate=True`
    with a warning if the retained set has fewer than `min_retained`
    points, per decision D3 / smoke test S8. Never silently returns a
    zero spread on a degenerate retained set.
    """
    base = trimmed_mean_aggregator(values, f)

    spread = base.spread
    degenerate = base.degenerate or (base.retained_n < min_retained)
    if base.retained_n < min_retained:
        if spread < sigma_min:
            msg = (
                f"RCE: retained set size {base.retained_n} < min_retained="
                f"{min_retained}; MAD={spread:.6g} is unreliable or exactly "
                f"zero. Flooring spread at sigma_min={sigma_min} and "
                f"marking guarantee_in_force=False for this round rather "
                f"than silently returning a confident-looking margin "
                f"(PROJECT_REPORT.md §13.2/§R10.2, smoke test S8)."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            logger.warning(msg)
            spread = sigma_min

    applied_margin = beta * spread
    pessimistic_estimate = base.point_estimate + applied_margin

    return RceResult(
        point_estimate=base.point_estimate,
        retained_values=base.retained_values,
        spread=spread,
        degenerate=degenerate,
        beta=beta,
        applied_margin=applied_margin,
        pessimistic_estimate=pessimistic_estimate,
    )
