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

P0 #2/#3/#5 (implementation repair, not a change to Eq. 2 itself):

  - Observation normalization. Every network's input observation is
    normalized by a per-agent, per-dimension running mean/std
    (`safelie.algos.normalization.RunningMeanStd`) before it reaches the
    policy or either critic. There was previously no normalization
    anywhere in the pipeline.
  - Return normalization. `value_net`/`cost_value_net` are trained to
    predict a *normalized* target; `AgentBundle.value`/`.cost_value`
    (the only entry points the rest of the codebase uses --
    `safelie.training.loop`, `safelie.sources.estimators` via the peer-
    critic query, `scripts/calibrate_cost.py`) denormalize back to return
    scale before returning, so every caller outside
    `safelie.training.ppo`'s own loss computation keeps receiving
    return-scale values exactly as before.
  - Separate optimizers. The policy, the reward critic, and the cost
    critic each get their own `torch.optim.Adam` instance and their own
    `clip_grad_norm_` call (`safelie.training.ppo`), rather than one
    optimizer and one joint clip across all three networks' concatenated
    parameters. A single shared clip means one network's gradient norm
    (e.g. an under-normalized cost critic's, before this fix) sets the
    scaling factor applied to all three networks' gradients, including
    the policy's -- a coupling that has no justification in Eq. 2, where
    the three updates are independent minimizations/maximizations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from safelie.algos.normalization import RunningMeanStd


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
    """One agent's policy + reward critic + cost critic, each with its
    own optimizer (P0 #5) plus observation/return normalization (P0
    #2/#3). Deliberately a plain container, not an nn.Module, since the
    three networks are optimized independently and the dual variable
    lives outside all of them.

    `value`/`cost_value` are the ONLY methods the rest of the codebase
    should call to query a critic (`safelie.training.loop`, the peer-
    critic source in `safelie.sources.estimators` via
    `_collect_source_value`, `scripts/calibrate_cost.py`): they normalize
    the input observation and denormalize the output back to return
    scale, so every caller keeps receiving exactly the same kind of
    quantity as before this fix. Only `safelie.training.ppo`'s own
    training step reaches past them to `value_net`/`cost_value_net`
    directly, because it needs the *normalized* prediction to compute a
    loss against a normalized target.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        lr: float,
        critic_lr: float | None = None,
        normalize_obs: bool = True,
        normalize_returns: bool = True,
    ):
        self.policy = GaussianPolicy(obs_dim, action_dim, hidden_dim)
        self.value_net = ValueCritic(obs_dim, hidden_dim)
        self.cost_value_net = ValueCritic(obs_dim, hidden_dim)

        self.normalize_obs = normalize_obs
        self.normalize_returns = normalize_returns
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.ret_rms = RunningMeanStd(shape=())
        self.cost_ret_rms = RunningMeanStd(shape=())

        effective_critic_lr = lr if critic_lr is None else critic_lr
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=effective_critic_lr)
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value_net.parameters(), lr=effective_critic_lr)

    def normalize_obs_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        return self.obs_rms.normalize(obs) if self.normalize_obs else obs

    def update_normalization_stats(self, obs, ret_r, ret_c) -> None:
        """Called once per round, before that round's PPO update, from
        that round's own on-policy rollout (`safelie.training.ppo`).
        Batched (Chan et al.) update, not per-step online, so
        normalization statistics are deterministic and checkpointable
        the same way everything else in this pipeline is."""
        if self.normalize_obs:
            self.obs_rms.update(obs)
        if self.normalize_returns:
            self.ret_rms.update(ret_r)
            self.cost_ret_rms.update(ret_c)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return-scale reward-value prediction (denormalized)."""
        obs_n = self.normalize_obs_tensor(obs)
        raw = self.value_net(obs_n)
        return self.ret_rms.denormalize(raw) if self.normalize_returns else raw

    def cost_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return-scale cost-value prediction (denormalized) -- exactly
        J_hat^i_C(theta_k) at `obs`, on the same scale as the budget
        d^i. This is what `safelie.training.loop._collect_source_value`
        uses for both the `own_critic` and `peer_critic` sources."""
        obs_n = self.normalize_obs_tensor(obs)
        raw = self.cost_value_net(obs_n)
        return self.cost_ret_rms.denormalize(raw) if self.normalize_returns else raw

    def state_dict(self) -> dict:
        return {
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
            "cost_value_net": self.cost_value_net.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "cost_value_optimizer": self.cost_value_optimizer.state_dict(),
            "obs_rms": self.obs_rms.state_dict(),
            "ret_rms": self.ret_rms.state_dict(),
            "cost_ret_rms": self.cost_ret_rms.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.policy.load_state_dict(state["policy"])
        self.value_net.load_state_dict(state["value_net"])
        self.cost_value_net.load_state_dict(state["cost_value_net"])
        self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        self.value_optimizer.load_state_dict(state["value_optimizer"])
        self.cost_value_optimizer.load_state_dict(state["cost_value_optimizer"])
        self.obs_rms.load_state_dict(state["obs_rms"])
        self.ret_rms.load_state_dict(state["ret_rms"])
        self.cost_ret_rms.load_state_dict(state["cost_ret_rms"])
