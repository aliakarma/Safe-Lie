"""Per-agent on-policy rollout buffer.

Report reference: PROJECT_REPORT.md §4.5 (storage layers), Phase 7.

Collects `rollout_length` steps of (obs, raw_action, logprob, reward,
reported_cost, value, cost_value, terminated, truncated) per agent, then
computes GAE for both the reward and the cost stream using the *same*
discount gamma (§6.1). Cost returns are on the return scale by
construction: they are exactly the quantity `safelie.defenses` and
`safelie.eval` expect.

`terminated`/`truncated` are kept SEPARATE (not collapsed into one `done`
flag) so `safelie.training.gae.compute_gae` can bootstrap correctly
through a time-limit truncation instead of treating it as a hard,
zero-cost-to-go terminal -- see that module's docstring for why this
matters concretely for `safelie.envs.mamujoco`.
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
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    # Bootstrap value at the TRUE final observation of a truncated-not-
    # terminated step (safelie.envs.mamujoco's `info["final_observation"]`),
    # one pair (reward-value, cost-value) per step; 0.0 wherever unused.
    truncation_value_bootstrap: list[float] = field(default_factory=list)
    truncation_cost_value_bootstrap: list[float] = field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        raw_action: np.ndarray,
        logprob: float,
        reward: float,
        cost: float,
        value: float,
        cost_value: float,
        terminated: bool,
        truncated: bool,
        truncation_value_bootstrap: float = 0.0,
        truncation_cost_value_bootstrap: float = 0.0,
    ) -> None:
        self.obs.append(obs)
        self.raw_actions.append(raw_action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.costs.append(cost)
        self.values.append(value)
        self.cost_values.append(cost_value)
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        self.truncation_value_bootstrap.append(truncation_value_bootstrap)
        self.truncation_cost_value_bootstrap.append(truncation_cost_value_bootstrap)

    def finalize(self, gamma: float, gae_lambda: float, last_value: float, last_cost_value: float) -> dict:
        rewards = np.array(self.rewards)
        costs = np.array(self.costs)
        values = np.array(self.values)
        cost_values = np.array(self.cost_values)
        terminated = np.array(self.terminated)
        truncated = np.array(self.truncated)
        trunc_v = np.array(self.truncation_value_bootstrap)
        trunc_cv = np.array(self.truncation_cost_value_bootstrap)

        adv_r, ret_r = compute_gae(
            rewards, values, terminated, truncated, gamma, gae_lambda, last_value,
            truncation_bootstrap_values=trunc_v,
        )
        adv_c, ret_c = compute_gae(
            costs, cost_values, terminated, truncated, gamma, gae_lambda, last_cost_value,
            truncation_bootstrap_values=trunc_cv,
        )

        return {
            "obs": np.stack(self.obs),
            "raw_actions": np.stack(self.raw_actions),
            "logprobs": np.array(self.logprobs),
            "adv_r": adv_r,
            "ret_r": ret_r,
            "adv_c": adv_c,
            "ret_c": ret_c,
            "cost_return_estimate": float(ret_c[0]) if len(ret_c) else 0.0,
            # Raw per-step critic predictions (the network's own V(s_t)
            # forward pass, *not* the GAE-lambda value target `ret_c`/
            # `ret_r`). Kept distinct so callers can log "critic
            # prediction" and "value target" as the two separate learner
            # training quantities they are (P0 #6) rather than
            # conflating a bootstrap target with what the critic itself
            # currently outputs.
            "values": values,
            "cost_values": cost_values,
            # Raw, undiscounted per-step means over this round's own
            # rollout. Pure diagnostics for the G0 validation stage
            # (docs/g0_gates.md): "training reward" and the realized
            # per-step cost rate, neither of which is derivable from the
            # GAE targets above and neither of which is an evaluation
            # metric -- those are the oracle's episodic_* quantities.
            "reward_mean": float(rewards.mean()) if len(rewards) else 0.0,
            "cost_rate_mean": float(costs.mean()) if len(costs) else 0.0,
            "n_terminated": int(terminated.sum()),
            "n_truncated": int(truncated.sum()),
        }

    def __len__(self) -> int:
        return len(self.rewards)
