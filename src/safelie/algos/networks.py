"""Per-agent actor / reward-critic / cost-critic networks.

Report reference: PROJECT_REPORT.md Phase 3 (compact scope: MAPPO-
Lagrangian only, decision noted in §R2.1); §7.4 hyperparameters
(hidden_dim pinned per decision D14, §12 gap G12: "network architectures
... deferred to the MACPO reference implementation by reference only" —
this repository pins one small, documented architecture instead, since
the MACPO reference is not vendored here).

Each agent holds an independent policy and independent reward/cost
critics conditioned on its own local observation — the decentralized
primal-dual setting of Eq. 2, not a centralized-critic variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def _mlp(in_dim: int, out_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, out_dim),
    )


class GaussianPolicy(nn.Module):
    """Diagonal-Gaussian continuous-action policy, actions squashed to
    [-1, 1] via tanh at sampling time (not inside the distribution, to
    keep the log-prob computation exact)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mean_net = _mlp(obs_dim, action_dim, hidden_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.mean_net(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        raw_action = dist.sample()
        logprob = dist.log_prob(raw_action).sum(-1)
        action = torch.tanh(raw_action)
        return action, logprob

    def evaluate(self, obs: torch.Tensor, raw_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        logprob = dist.log_prob(raw_action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logprob, entropy


class ValueCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = _mlp(obs_dim, 1, hidden_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class AgentBundle:
    """One agent's policy + reward critic + cost critic, with their
    optimizers. Deliberately a plain container, not an nn.Module, since
    the three networks are optimized with (potentially) different
    schedules and the dual variable lives outside all of them."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, lr: float):
        self.policy = GaussianPolicy(obs_dim, action_dim, hidden_dim)
        self.value = ValueCritic(obs_dim, hidden_dim)
        self.cost_value = ValueCritic(obs_dim, hidden_dim)
        params = list(self.policy.parameters()) + list(self.value.parameters()) + list(
            self.cost_value.parameters()
        )
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def state_dict(self) -> dict:
        return {
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "cost_value": self.cost_value.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.policy.load_state_dict(state["policy"])
        self.value.load_state_dict(state["value"])
        self.cost_value.load_state_dict(state["cost_value"])
        self.optimizer.load_state_dict(state["optimizer"])
