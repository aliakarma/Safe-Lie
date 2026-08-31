"""Generalized Advantage Estimation, shared between reward and cost.

Report reference: PROJECT_REPORT.md §7.4 — GAE lambda = 0.95 `[SPEC]`;
§6.1 — "the SAME gamma must be used for cost returns; using undiscounted
cost sums with discounted rewards is a common and silent bug." This
module takes gamma as an explicit argument every call so that bug is
structurally hard to introduce (there is no default that could be reused
inconsistently between the reward and cost computations).
"""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
    last_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """rewards, values, dones: shape (T,). values[t] is V(s_t); last_value
    is V(s_T) (bootstrap). Returns (advantages, returns), both shape (T,).
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    last_gae = 0.0
    next_value = last_value
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * gae_lambda * mask * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns
