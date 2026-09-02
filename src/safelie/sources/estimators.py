"""Return-scale cost estimators backing each source type.

Report reference: PROJECT_REPORT.md Phase 2 (`[GAP]` G5, ensemble/monitor
diversification) and Phase 1 (`[GAP]` G4, peer observability of C^i).

  - `own_critic`: the constraint owner's own discounted Monte-Carlo
    constraint return over the round's rollout (see
    `safelie.training.constraint_return`) — exactly `J_hat^i_C(theta_k)`
    in the paper's notation, with no critic and no function-approximation
    bias in the path.
  - `peer_critic`: a *different* agent's **constraint-report head**
    (`DiversifiedReplica`, see below) — not that agent's PPO cost-value
    critic — evaluated on the constraint owner's initial observation. This
    resolves G4 by restricting peer observability to state the peer's own
    network can read (the owner's observation vector), rather than
    inventing cross-agent private-state access. `docs/g2_gates.md`
    records why this must be a dedicated head rather than
    `AgentBundle.cost_value`: the PPO cost critic is trained against
    `ret_c`, a GAE(lambda) bootstrap target, for a purpose (advantage
    estimation) that has nothing to do with the peer's OWN constraint
    return being queried out of distribution by another agent, and
    G1 measured the result of that mismatch at corr(peer_critic, true)
    ~= 0 (uncorrelated with the truth) across all three G1 seeds.
  - `ensemble_replica` / `monitor`: small, independently-initialized
    regression heads, refit every round on a bootstrap resample of the
    owner's own rollout (obs, cost-to-go) pairs. This is the diversification
    mechanism the report recommends (independent init + bootstrap-resampled
    minibatches) and is genuinely different data + genuinely different
    weights each round, not a relabeled copy of the same network.

G2-peer (`docs/g2_gates.md`) reuses the same `DiversifiedReplica` class
for a second purpose: one **constraint-report head per physical agent**,
refit once per round on that agent's own masked MC cost-to-go targets
(`safelie.training.loop.ExperimentRun.constraint_report_heads`), then
queried — without refitting — once for every owner that has this agent as
a `peer_critic` source this round. This is why `refit()` and `predict()`
are exposed as separate methods below: a `monitor`/`ensemble_replica`
source still calls `refit_and_predict()` once per owner per round (fit
and query the SAME owner's data, unchanged since G1), while the new
peer-critic head must be fit exactly once per round and then queried by
several different owners against several different observations —
refitting on every query would silently re-bias the head toward whichever
owner queried it most recently in that round.
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

    def refit(self, obs: np.ndarray, cost_to_go: np.ndarray) -> None:
        """Fit this head's weights to `(obs, cost_to_go)`, in place, via
        `self.steps` gradient steps on bootstrap resamples. Does not
        predict; call `predict` (possibly several times, for several
        different query points) afterward."""
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

    def predict(self, query_obs: np.ndarray) -> float:
        """This head's current prediction at `query_obs`, with no refit."""
        with torch.no_grad():
            query = torch.as_tensor(query_obs, dtype=torch.float32).unsqueeze(0)
            return float(self.net(query).item())

    def refit_and_predict(self, obs: np.ndarray, cost_to_go: np.ndarray, query_obs: np.ndarray) -> float:
        """`refit(obs, cost_to_go)` then `predict(query_obs)` — the
        `ensemble_replica`/`monitor` sources' calling convention,
        unchanged since G1: fit and query the same owner's data every
        call."""
        self.refit(obs, cost_to_go)
        return self.predict(query_obs)

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
