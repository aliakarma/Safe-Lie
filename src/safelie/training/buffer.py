"""Per-agent on-policy rollout buffer.

Report reference: PROJECT_REPORT.md §4.5 (storage layers), Phase 7.

Collects `rollout_length` steps of (obs, raw_action, logprob, reward,
reported_cost, value, cost_value, done) per agent, then computes GAE for
both the reward and the cost stream using the *same* discount gamma
(§6.1). Cost returns are on the return scale by construction: they are
exactly the quantity `safelie.defenses` and `safelie.eval` expect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from safelie.training.gae import compute_gae


@dataclass
class AgentRollout:
    obs: list[np.ndarray] = field(default_factory=list)
    raw_actions: list[np.ndarray] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    cost_values: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        raw_action: np.ndarray,
        logprob: float,
        reward: float,
        cost: float,
        value: float,
        cost_value: float,
        done: bool,
    ) -> None:
        self.obs.append(obs)
        self.raw_actions.append(raw_action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.costs.append(cost)
        self.values.append(value)
        self.cost_values.append(cost_value)
        self.dones.append(done)

    def finalize(self, gamma: float, gae_lambda: float, last_value: float, last_cost_value: float) -> dict:
        rewards = np.array(self.rewards)
        costs = np.array(self.costs)
        values = np.array(self.values)
        cost_values = np.array(self.cost_values)
        dones = np.array(self.dones)

        adv_r, ret_r = compute_gae(rewards, values, dones, gamma, gae_lambda, last_value)
        adv_c, ret_c = compute_gae(costs, cost_values, dones, gamma, gae_lambda, last_cost_value)

        return {
            "obs": np.stack(self.obs),
            "raw_actions": np.stack(self.raw_actions),
            "logprobs": np.array(self.logprobs),
            "adv_r": adv_r,
            "ret_r": ret_r,
            "adv_c": adv_c,
            "ret_c": ret_c,
            "cost_return_estimate": float(ret_c[0]) if len(ret_c) else 0.0,
        }

    def __len__(self) -> int:
        return len(self.rewards)
