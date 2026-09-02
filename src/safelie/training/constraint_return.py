"""Monte-Carlo estimators of the constraint objective J_C^i(theta).

Report reference: main_iclr.tex Eq. 1 --
`J_C^i(theta) = E_{pi_theta}[ sum_t gamma^t C_t^i ]`; PROJECT_REPORT.md
Section 6.1 ("the SAME gamma must be used for cost returns").

**Why this module exists (the G0 conditional-pass finding).** Before it,
the value that reached the dual update was `ret_c[0]` -- the GAE(lambda)
bootstrap target at the round's first step, produced by
`safelie.training.gae.compute_gae`. That quantity is a perfectly valid
*value-regression target* for the cost critic and a perfectly valid input
to policy-gradient advantage estimation. It is **not** a direct estimator
of `J_C^i(theta)`, because a lambda-return is a geometric blend of n-step
returns whose weight on genuine sampled cost decays as
`(gamma*lambda)^n`. At gamma=0.99, lambda=0.95 that blend keeps only
about `1/(1 - gamma*lambda) ~ 16.8` steps of real Monte-Carlo evidence
and hands the remaining horizon -- roughly `1/(1-gamma) = 100` steps
worth of discounted mass -- to the cost critic's own bootstrap. Whatever
bias the critic carries is therefore inherited by the number the dual
update compares against the budget `d`, and the G0 clean baseline
measured exactly that: a sustained mean bias of about -7 against the
withheld oracle's true discounted cost return.

The repair is a separation of concerns, not a replacement of GAE:

  * **Policy-gradient advantage estimation** keeps GAE(gamma=0.99,
    lambda=0.95) unchanged -- `adv_r`, `adv_c`, and the two critics'
    regression targets `ret_r`/`ret_c` all still come from
    `safelie.training.gae`. Nothing in `safelie.training.ppo` changes.
  * **Constraint-objective estimation** -- the quantity the dual update
    needs -- is computed here, independently, as an explicit discounted
    Monte-Carlo sum over sampled cost, with no critic in the path.

## Which discounted sum, exactly

There are two defensible empirical readings of `sum_t gamma^t C_t`, and
they are NOT interchangeable, so this module computes both and names them
apart rather than picking one silently.

``discounted_window_return`` -- **one global discount clock over the
round's fixed `rollout_length`-step window**, ignoring any episode
boundary the backend auto-resets through. This is the definition the rest
of this repository already uses: `safelie.eval.oracle.OracleEvaluator`
accumulates `gamma**self._t * c` with a `_t` that starts at 0 on
`env.reset()` and never resets, and
`safelie.eval.harness.evaluate_true_cost` discounts task return and
reported cost with the identical `gamma**t`. `OracleEpisodeResult`'s
docstring states it outright: "'episode' always denotes this fixed
window, never a termination-to-termination segment". It is also the
definition `docs/g0_gates.md` gates on.

**This is the estimator that feeds the dual update**, for one reason that
is methodological rather than aesthetic: it is the only choice under
which "estimator bias against the oracle" is a comparison of two
measurements of the *same* functional. Feeding the dual a per-episode
quantity while grading it against a fixed-window oracle would fold a
definitional mismatch into the bias number and make the G1 comparison
uninterpretable. Redefining J_C is a separate scientific change and is
not made here.

``episodic_mc_returns`` -- the textbook estimator: each episode gets its
own clock starting at its own first step, and complete episodes are
averaged. Computed and logged as a **diagnostic**, so the size of the
definitional difference is measured rather than assumed. Two honest
caveats, both recorded in the returned dataclass rather than buried:
episodes still running when the window ends are *censored* and excluded
(including them would under-read), and excluding them biases the mean
toward short episodes, which under a cost that scales with velocity are
not a random subsample. That censoring bias is precisely why this is the
diagnostic and the window sum is the signal.

At gamma=0.99 the two agree closely whenever the round's first episode
runs long relative to the ~100-step effective horizon
(`gamma**1000 = 4.3e-5`), and diverge when episodes are short. The G1
logs record both every round, so the divergence is observable.

## What this module must not be confused with

Not an undiscounted sum. Not a per-step average (`cost_rate_mean` in the
rollout diagnostics is that, and it is *not* on the return scale -- a
uniform-in-time assumption relating the two is false here, and the G0
data says so: a late-training round with mean rate 0.105 has a true
discounted return of ~21, not ~10.5, because cost concentrates early in
an episode). Not a GAE advantage. Not a normalized return
(`safelie.algos.normalization` operates strictly inside
`safelie.training.ppo`; every value here is raw return scale, the same
scale `safelie.defenses` and `safelie.eval` speak). Not a critic
prediction.

## Isolation

Every function here consumes the learner-visible `DualCostStep`
`reported_cost` stream. None of them touches
`env._oracle_handle_privileged()` or true cost. Under `attack.name ==
"none"` the environment's reported and true per-step cost happen to be
identical, so the clean numbers coincide -- that is a useful correctness
check, not a channel. Under attack the corruption is applied downstream
of this module, to the *communicated source reports*
(`safelie.attacks.apply_attack` in `safelie.training.loop`), exactly as
before: the attacker still corrupts the report, never the oracle, and the
dual update still consumes only corrupted-then-aggregated values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def discounted_cost_to_go(costs: np.ndarray, gamma: float, tail_bootstrap: float = 0.0) -> np.ndarray:
    """`G_t = sum_{s=t}^{T-1} gamma^(s-t) * c_s + gamma^(T-t) * tail_bootstrap`.

    One global clock, no reset at episode boundaries -- see the module
    docstring for why that matches this repository's definition of J_C.
    `tail_bootstrap` defaults to 0.0, which makes `G_t` literally "the
    discounted cost remaining inside this window", self-consistent with
    the window definition and free of any critic dependence. Shape (T,).
    """
    costs = np.asarray(costs, dtype=np.float64)
    out = np.zeros(len(costs), dtype=np.float64)
    running = float(tail_bootstrap)
    for t in reversed(range(len(costs))):
        running = float(costs[t]) + gamma * running
        out[t] = running
    return out


def discounted_window_return(costs: np.ndarray, gamma: float) -> float:
    """`sum_{t=0}^{T-1} gamma^t * c_t` -- the empirical estimate of
    `J_C^i(theta)` this repository's oracle also computes. Identical to
    `discounted_cost_to_go(costs, gamma)[0]`; kept as its own name because
    it is the quantity the dual update consumes and it should be greppable
    as such."""
    costs = np.asarray(costs, dtype=np.float64)
    if len(costs) == 0:
        return 0.0
    return float(discounted_cost_to_go(costs, gamma)[0])


def complete_target_horizon(gamma: float, tol: float = 0.01) -> int:
    """Smallest `h` with `gamma**h <= tol`: the number of trailing steps
    of a window whose Monte-Carlo cost-to-go is censored by more than
    `tol` of its own discounted mass.

    Used to mask the tail when fitting a regression head against MC
    cost-to-go targets. Without the mask, rows near `t = T` carry targets
    that are systematically shrunk (at gamma=0.99, `G_{T-100}` is missing
    37% of its mass, `G_{T-10}` 90%), and a head fit on them predicts low
    at the query point -- reintroducing, through the back door, the same
    downward bias this module exists to remove. At gamma=0.99, tol=0.01
    this is 459 steps, leaving 1541 of a 2000-step window's rows usable.

    Note this does NOT affect `discounted_window_return`, whose own
    censoring is `gamma**T` (1.9e-9 at T=2000) and therefore nil.
    """
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must be in (0, 1); got {gamma!r}")
    if not 0.0 < tol < 1.0:
        raise ValueError(f"tol must be in (0, 1); got {tol!r}")
    return int(math.ceil(math.log(tol) / math.log(gamma)))


def complete_target_count(window_length: int, gamma: float, tol: float = 0.01) -> int:
    """How many leading rows of a `window_length`-step window have MC
    cost-to-go complete to within `tol`.

    When the window is shorter than `complete_target_horizon` no masking
    is possible and every row is returned: every target is then censored
    to the same degree the window definition of J_C is itself censored,
    which is inherent rather than a defect introduced here. Always at
    least 1.
    """
    h = complete_target_horizon(gamma, tol)
    if window_length <= h:
        return max(1, int(window_length))
    return int(window_length - h)


@dataclass(frozen=True)
class EpisodicMCResult:
    """Diagnostic companion to the window estimator (module docstring).

    `mean_return` averages only COMPLETE episodes -- ones whose end fell
    inside the window. `censored_length` is how many trailing steps
    belonged to an episode still running when the window closed; those
    steps are excluded, and `n_complete == 0` means the whole window was
    one unfinished episode, in which case `mean_return` falls back to the
    window sum (the best available estimate) and `fell_back` says so.
    """

    mean_return: float
    n_complete: int
    episode_returns: tuple[float, ...]
    episode_lengths: tuple[int, ...]
    censored_length: int
    fell_back: bool


def episodic_mc_returns(
    costs: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
) -> EpisodicMCResult:
    """Per-episode discounted cost return, each episode on its own clock.

    An episode ends at any `t` with `terminated[t] or truncated[t]` (the
    backend auto-resets there -- see
    `safelie.envs.mamujoco.MaMuJoCoDualCostEnv.step` -- so step `t+1` is
    already the next episode's first step). This is the textbook
    Monte-Carlo estimator of `J_C`; it is a diagnostic here, not the dual
    signal, for the definitional reason in the module docstring plus the
    short-episode censoring bias recorded on `EpisodicMCResult`.
    """
    costs = np.asarray(costs, dtype=np.float64)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    T = len(costs)
    if T == 0:
        return EpisodicMCResult(0.0, 0, (), (), 0, True)

    returns: list[float] = []
    lengths: list[int] = []
    start = 0
    for t in range(T):
        if terminated[t] or truncated[t]:
            seg = costs[start : t + 1]
            disc = gamma ** np.arange(len(seg), dtype=np.float64)
            returns.append(float(np.dot(disc, seg)))
            lengths.append(len(seg))
            start = t + 1
    censored = T - start
    if not returns:
        return EpisodicMCResult(discounted_window_return(costs, gamma), 0, (), (), censored, True)
    return EpisodicMCResult(
        mean_return=float(np.mean(returns)),
        n_complete=len(returns),
        episode_returns=tuple(returns),
        episode_lengths=tuple(lengths),
        censored_length=censored,
        fell_back=False,
    )
