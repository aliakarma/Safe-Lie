"""Static corruption operators: the primary attack, over-reporting, and the
benign (non-adversarial) control.

Report reference: main_iclr.tex App. B; PROJECT_REPORT.md §7.2.

Injection point (`[SPEC]`, load-bearing): corruption is applied to the
return-scale residual g^i_k = J_hat^i_C - d^i, **after** critic evaluation
and **before** consensus — the point corresponding to a compromised
channel. Applying it anywhere else (e.g. to per-step costs inside the
environment) changes the threat model from channel compromise to sensor
compromise and invalidates comparison with the paper. This module only
ever transforms a `residuals: dict[str, float]` that the caller has
already computed post-critic, pre-consensus.
"""

from __future__ import annotations

import numpy as np

from safelie.attacks.taxonomy import Direction, Support


def _sign(direction: Direction, source_id: str) -> float:
    if direction == "negative":
        return -1.0
    if direction == "positive":
        return 1.0
    # "mixed": deterministic per-source sign so the schedule is reproducible
    # without extra RNG state.
    return -1.0 if (hash(source_id) % 2 == 0) else 1.0


def static_attack(
    residuals: dict[str, float],
    corrupted: set[str],
    k: int,
    B: float,
    direction: Direction = "negative",
    support: Support = "persistent",
    selective_period: int = 3,
) -> dict[str, float]:
    """The primary attack, generalized over the direction and support axes.

    Primary attack (`[SPEC]`, App. B): direction="negative",
    support="persistent" gives delta^i_k = -B every round for i in
    `corrupted`. `direction="positive"` gives the over-reporting attack
    used in Proposition 3's liveness argument. `support="selective"`
    corrupts only every `selective_period`-th round.
    """
    if support == "selective" and (k % selective_period != 0):
        return dict(residuals)
    out = dict(residuals)
    for sid in corrupted:
        if sid in out:
            out[sid] = out[sid] + _sign(direction, sid) * B
    return out


def benign_control(
    residuals: dict[str, float],
    corrupted: set[str],
    k: int,
    sigma_equiv: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """The falsification test (`[SPEC]`, §5.1): unbiased zero-mean noise of
    magnitude matched to the attack's budget, replacing the adversary.

    `sigma_equiv` should be chosen so the noise magnitude matches the
    attack's per-round budget B (§R2.2 condition D); smoke test S20
    checks the resulting sample mean is indistinguishable from zero over
    many rounds — if this control is silently biased, condition D is
    worthless."""
    out = dict(residuals)
    for sid in corrupted:
        if sid in out:
            out[sid] = out[sid] + float(rng.normal(0.0, sigma_equiv))
    return out
