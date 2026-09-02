"""Generalized Advantage Estimation, shared between reward and cost.

Report reference: PROJECT_REPORT.md §7.4 — GAE lambda = 0.95 `[SPEC]`;
§6.1 — "the SAME gamma must be used for cost returns; using undiscounted
cost sums with discounted rewards is a common and silent bug." This
module takes gamma as an explicit argument every call so that bug is
structurally hard to introduce (there is no default that could be reused
inconsistently between the reward and cost computations).

**Terminated vs. truncated (bug fix, not part of `[SPEC]`).** A prior
version of this function took one combined `dones` array and treated
every done step identically: zero bootstrap, and the lambda-recursion cut
at that boundary. That is correct for a genuine *termination* (the MDP
truly ends; cost-to-go after it is exactly 0) but wrong for a time-limit
*truncation* (an artificial cutoff; the underlying MDP does not end, and
the standard, well-documented fix -- see Pardo et al. 2018, "Time Limits
in Reinforcement Learning" -- is to bootstrap through the cutoff using the
critic's own value estimate at the true final observation, not zero).

This matters concretely here: `safelie.envs.mamujoco`'s backend applies
its own ~1000-step time limit, well below the pilot configs'
`rollout_length=2000`, and auto-resets internally on either signal
(`MaMuJoCoDualCostEnv.step`). Collapsing that mid-rollout truncation into
a hard zero-bootstrap boundary (as the combined-`dones` version did)
silently caps `ret_c[0]`'s effective horizon at whatever the time limit
allows and introduces exactly the kind of systematic downward bias P0 #1
already documents from other sources -- an additional, compounding
mechanism, not the same one. The synthetic environment never triggers
this distinction (its own "truncation" always falls exactly on the last
buffer index, where there is nothing left to bootstrap into either way),
so this fix changes MaMuJoCo-backed runs' cost-critic targets and leaves
synthetic-environment numbers unchanged.
"""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
    gae_lambda: float,
    last_value: float = 0.0,
    truncation_bootstrap_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """rewards, values, terminated, truncated: shape (T,). values[t] is
    V(s_t); last_value is V(s_T) (bootstrap for the buffer's own final
    step, when neither terminated nor truncated fires there -- e.g. a
    round that ends mid-episode with more real trajectory beyond the
    buffer). `truncation_bootstrap_values[t]`: the critic's value
    estimate at the TRUE final observation of a truncated-at-t episode
    (see `safelie.envs.mamujoco.MaMuJoCoDualCostEnv`'s
    `info["final_observation"]`); required wherever `truncated[t] and not
    terminated[t]`, ignored elsewhere. Defaults to 0 everywhere if not
    given, reproducing the (incorrect, but harmless for an environment
    that never truncates mid-buffer) old behaviour.

    A step with `terminated[t]=True` always bootstraps with 0, regardless
    of `truncated[t]` -- a genuine terminal takes precedence.

    Returns (advantages, returns), both shape (T,).
    """
    T = len(rewards)
    if truncation_bootstrap_values is None:
        truncation_bootstrap_values = np.zeros(T)
    advantages = np.zeros(T, dtype=np.float64)
    last_gae = 0.0
    next_value = last_value
    for t in reversed(range(T)):
        term_t = bool(terminated[t])
        trunc_t = bool(truncated[t]) and not term_t
        done_t = term_t or trunc_t
        mask = 0.0 if done_t else 1.0  # cuts the lambda-recursion at any episode boundary
        if term_t:
            bootstrap = 0.0
        elif trunc_t:
            bootstrap = float(truncation_bootstrap_values[t])
        else:
            bootstrap = next_value
        delta = rewards[t] + gamma * bootstrap - values[t]
        last_gae = delta + gamma * gae_lambda * mask * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns
