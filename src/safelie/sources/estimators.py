"""Return-scale cost estimators backing each source type.

Report reference: PROJECT_REPORT.md Phase 2 (`[GAP]` G5, ensemble/monitor
diversification) and Phase 1 (`[GAP]` G4, peer observability of C^i).

  - `own_critic`: the constraint owner's own cost-value network,
    bootstrapped at the round's initial observation via GAE — exactly
    `J_hat^i_C(theta_k)` in the paper's notation.
  - `peer_critic`: a *different* agent's cost-value network, evaluated on
    the constraint owner's initial observation. This resolves G4 by
    restricting peer observability to state the peer's own network can
    read (the owner's observation vector), rather than inventing
    cross-agent private-state access.
  - `ensemble_replica` / `monitor`: small, independently-initialized
    regression heads, refit every round on a bootstrap resample of the
    owner's own rollout (obs, cost-to-go) pairs. This is the diversification
    mechanism the report recommends (independent init + bootstrap-resampled
    minibatches) and is genuinely different data + genuinely different
    weights each round, not a relabeled copy of the same network.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class DiversifiedReplica:
    """One independently-initialized small regression head, refit each
    round on a bootstrap resample of (obs, cost_to_go) pairs."""

    def __init__(self, obs_dim: int, hidden_dim: int = 16, seed: int = 0, lr: float = 1e-2, steps: int = 20):
        gen = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        for p in self.net.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, generator=gen)
            else:
                nn.init.zeros_(p)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.steps = steps
        self.rng = np.random.default_rng(seed)

    def refit_and_predict(self, obs: np.ndarray, cost_to_go: np.ndarray, query_obs: np.ndarray) -> float:
        n = len(obs)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        target_t = torch.as_tensor(cost_to_go, dtype=torch.float32)
        for _ in range(self.steps):
            idx = self.rng.integers(0, n, size=n)  # bootstrap resample
            pred = self.net(obs_t[idx]).squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred, target_t[idx])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        with torch.no_grad():
            query = torch.as_tensor(query_obs, dtype=torch.float32).unsqueeze(0)
            return float(self.net(query).item())

    def state_dict(self) -> dict:
        """Report reference / smoke test S14: a checkpoint that omits this
        state would restore a replica to its *initial* random weights
        rather than its trained-so-far weights, silently breaking
        bitwise-identical continuation from exactly the round after
        restore."""
        return {"net": self.net.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.net.load_state_dict(state["net"])
        self.optimizer.load_state_dict(state["optimizer"])
