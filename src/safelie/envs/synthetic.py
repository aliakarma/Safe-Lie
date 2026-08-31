"""A lightweight synthetic constrained multi-agent environment.

Report reference: PROJECT_REPORT.md §9 (system-prompt engineering
guidance) — "provide a small synthetic/demo dataset ONLY when useful for
smoke testing, clearly labeled as non-paper data."

**This is NOT Safe Multi-Agent MuJoCo.** The paper's actual environments
(`ManyAgent Ant`, `HalfCheetah` 2x3, Safety-Gymnasium multi-agent
navigation) require the MuJoCo physics engine and multi-agent wrapper
packages that are heavy, GPU-stage dependencies the report itself assigns
to Google Colab, not to local CPU verification (§R7.1: "the local machine
never runs a full-scale training job"). This module is a `[DECISION]`/
approximation classified (C) in docs/assumptions.md: a hand-built,
CPU-only environment satisfying the exact `DualCostEnvWrapper` contract,
used to run the Stage-1 local smoke suite (S15-S20) and a genuine tiny
end-to-end training loop. No number this environment produces should be
compared to anything in main_iclr.tex Table 3/4. Real Safe MAMuJoCo
integration is the documented Stage-2 requirement — see
`safelie.envs.mamujoco`.

Dynamics (deliberately simple, chosen only so that (a) a random policy's
discounted cost return clears the budget and (b) a policy that drives
actions toward zero can satisfy it, so the constraint is genuinely
binding and testable):

    x_i <- clip(x_i + 0.1 * a_i[0] + noise, -5, 5)     per agent i
    reward = -0.1 * sum_i x_i^2 - 0.01 * sum_i ||a_i||^2   (SHARED, Definition 1)
    true_cost_i = sum(a_i^2)                                (per-agent, Definition 1, C^i >= 0)
"""

from __future__ import annotations

import numpy as np

from safelie.envs.dual_cost import AgentID, DualCostStep
from safelie.envs.guards import OracleReadOnlyView, _PrivilegedOracleView


class SyntheticConstrainedMarlEnv:
    """CPU-only stand-in for Safe MAMuJoCo, used for local verification only."""

    def __init__(
        self,
        n_agents: int,
        budget: float = 25.0,
        obs_dim: int = 8,
        action_dim: int = 2,
        horizon: int = 200,
        process_noise: float = 0.02,
    ):
        self.n_agents = n_agents
        self.budget = budget
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.process_noise = process_noise

        self.agent_ids: list[AgentID] = [f"agent_{i}" for i in range(n_agents)]
        self._x: np.ndarray = np.zeros(n_agents)
        self._t = 0
        self._rng = np.random.default_rng(0)
        self._last_true_cost: dict[AgentID, float] = {aid: 0.0 for aid in self.agent_ids}

    def reset(self, seed: int | None = None) -> DualCostStep:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._x = self._rng.normal(0.0, 0.5, size=self.n_agents)
        self._t = 0
        obs = self._make_obs()
        self._last_true_cost = {aid: 0.0 for aid in self.agent_ids}
        return DualCostStep(
            obs=obs,
            reward=0.0,
            reported_cost=dict(self._last_true_cost),
            terminated={aid: False for aid in self.agent_ids},
            truncated={aid: False for aid in self.agent_ids},
            info={},
        )

    def _make_obs(self) -> dict[AgentID, np.ndarray]:
        obs = {}
        for i, aid in enumerate(self.agent_ids):
            base = np.zeros(self.obs_dim)
            base[0] = self._x[i]
            base[1] = self._t / max(self.horizon, 1)
            if self.obs_dim > 2:
                base[2:] = self._rng.normal(0.0, 0.01, size=self.obs_dim - 2)
            obs[aid] = base.astype(np.float32)
        return obs

    def step(self, actions: dict[AgentID, np.ndarray]) -> DualCostStep:
        action_arr = np.stack(
            [np.clip(np.asarray(actions[aid], dtype=float), -1.0, 1.0) for aid in self.agent_ids]
        )
        drive = action_arr[:, 0] if self.action_dim >= 1 else np.zeros(self.n_agents)
        noise = self._rng.normal(0.0, self.process_noise, size=self.n_agents)
        self._x = np.clip(self._x + 0.1 * drive + noise, -5.0, 5.0)

        true_cost = {
            aid: float(np.sum(action_arr[i] ** 2)) for i, aid in enumerate(self.agent_ids)
        }
        self._last_true_cost = true_cost

        reward = float(-0.1 * np.sum(self._x**2) - 0.01 * np.sum(action_arr**2))

        self._t += 1
        truncated = self._t >= self.horizon
        obs = self._make_obs()

        return DualCostStep(
            obs=obs,
            reward=reward,
            reported_cost=dict(true_cost),  # faithful at the environment level; see module docstring
            terminated={aid: False for aid in self.agent_ids},
            truncated={aid: truncated for aid in self.agent_ids},
            info={"t": self._t},
        )

    def oracle_handle(self) -> OracleReadOnlyView:
        """Public method any code can call. Always returns a sealed
        (raising) view; see `_oracle_handle_privileged`."""
        return OracleReadOnlyView(get_true_cost=lambda: self._last_true_cost)

    def _oracle_handle_privileged(self) -> _PrivilegedOracleView:
        """Not part of `DualCostEnvWrapper`. Only
        `safelie.eval.oracle.OracleEvaluator` calls this."""
        return _PrivilegedOracleView(get_true_cost=lambda: self._last_true_cost)
