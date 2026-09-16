"""The reporting-residual / median-deviation monitor.

Report reference: main_iclr.tex Section 2, the sentence following
Proposition `cor:spread` -- "an operator flagging any agent whose
multiplier deviates from the fleet median catches the concentrated attack
exactly, whereas under consensus the same total bias is uniform, the
median moves with it, and no agent deviates at all"; and
`safelie.theory.spreading`, which already computes this statistic on the
*theoretical* dual bias `e_K`.

This module is the operational counterpart of that sentence, applied to
the quantity an operator can actually see: the realized multiplier vector
`lambda_k`, read from a completed run's `rounds.jsonl`.

**This is not an adversarial-intent detector and must not be described as
one.** It is a dispersion statistic on the multiplier vector. It cannot
distinguish an attack from any other cause of multiplier dispersion --
heterogeneous agents, a hard constraint for one agent, or an unlucky
seed. Under `W = I` the multipliers are *intrinsically* dispersed even
with no adversary present, because each coordinate integrates its own
residual with no mixing (Proposition `prop:enforced`), so a raised
statistic under `W = I` is not by itself evidence of corruption. That is
exactly why the campaign this module serves runs a clean reference under
each topology rather than comparing a statistic against an absolute bar.

**The threshold is deliberately not fixed here.** main_iclr.tex specifies
the statistic ("deviates from the fleet median") but names no `tau`. This
module therefore reports the statistic and evaluates detection over a
caller-supplied grid of thresholds; choosing one is a pre-declaration
decision belonging to a gates document, not a default hidden in a
library. See docs/p1_concentrated_attack_gates.md Section 8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MonitorTrace:
    """Per-round output of the median-deviation monitor over a whole run.

    `deviation` is `|lambda_k^i - median_j(lambda_k^j)|`, shape (K, N).
    `fleet_statistic` is `max_i deviation[k, i]`, shape (K,) -- the single
    number an operator watching the whole fleet would track.
    """

    deviation: np.ndarray
    fleet_statistic: np.ndarray
    median: np.ndarray

    @property
    def n_rounds(self) -> int:
        return int(self.deviation.shape[0])

    @property
    def n_agents(self) -> int:
        return int(self.deviation.shape[1])


def median_deviation_trace(lam: np.ndarray) -> MonitorTrace:
    """Run the monitor over a multiplier trajectory `lam` of shape (K, N).

    The median is taken across agents *within* a round, never across
    rounds: the monitor is a snapshot dispersion test, and pooling rounds
    would let a slow drift common to the whole fleet register as a
    per-agent anomaly.
    """
    lam = np.asarray(lam, dtype=float)
    if lam.ndim != 2:
        raise ValueError(f"lam must be (K, N); got shape {lam.shape}")
    med = np.median(lam, axis=1)
    dev = np.abs(lam - med[:, None])
    return MonitorTrace(deviation=dev, fleet_statistic=dev.max(axis=1), median=med)


def detection_frequency(trace: MonitorTrace, tau: float) -> float:
    """Fraction of rounds at which ANY agent exceeds `tau`."""
    return float((trace.fleet_statistic > tau).mean())


def per_agent_detection_frequency(trace: MonitorTrace, tau: float) -> np.ndarray:
    """Fraction of rounds at which EACH agent individually exceeds `tau`."""
    return (trace.deviation > tau).mean(axis=0)


def threshold_sweep(trace: MonitorTrace, taus: list[float] | np.ndarray) -> dict[float, float]:
    """Detection frequency at each threshold in `taus`.

    Reported instead of a single number because main_iclr.tex fixes the
    statistic but not the threshold: a conclusion that survives the whole
    sweep does not depend on a `tau` chosen after the fact.
    """
    return {float(t): detection_frequency(trace, float(t)) for t in taus}


def calibrate_threshold(clean_trace: MonitorTrace, false_positive_rate: float) -> float:
    """The `tau` at which a CLEAN reference run fires on `false_positive_rate`
    of its rounds, i.e. the `(1 - fpr)` quantile of the clean fleet statistic.

    The clean reference must come from a run under the SAME topology. A
    threshold calibrated on a ring run and applied to a `W = I` run measures
    the topology's intrinsic dispersion, not the adversary -- the confound
    this function's docstring exists to make hard to commit by accident.
    """
    if not 0.0 < false_positive_rate < 1.0:
        raise ValueError(f"false_positive_rate must be in (0,1); got {false_positive_rate}")
    return float(np.quantile(clean_trace.fleet_statistic, 1.0 - false_positive_rate))


def concentration_ratio(vec: np.ndarray) -> np.ndarray:
    """`max_i |v_i| / sum_i |v_i|` per row -- the localization statistic.

    Equals 1 when all the mass sits on one coordinate (the `W = I`
    signature of Proposition `cor:spread`) and `1/N` when it is spread
    perfectly uniformly (the connected-consensus signature). Rows whose
    total mass is numerically zero return NaN rather than 0 or 1: a
    perturbation that has vanished has no meaningful concentration, and
    silently reporting either extreme would invent a signature.
    """
    v = np.abs(np.asarray(vec, dtype=float))
    if v.ndim == 1:
        v = v[None, :]
    total = v.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(total > 1e-12, v.max(axis=1) / total, np.nan)
    return out
