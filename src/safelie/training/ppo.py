"""The PPO-Lagrangian policy/critic update.

Report reference: main_iclr.tex Eq. 2 — theta^i_{k+1} = theta^i_k +
eta_theta grad_{theta^i}[J_R(theta_k) - lambda^i_k J^i_C(theta_k)]; PPO
clip 0.2, GAE lambda 0.95 `[SPEC]`.

The Lagrangian combination is realized the standard way for clipped PPO:
a single combined advantage `adv_R - lambda^i * adv_C` drives one clipped
surrogate objective, so the same trust-region mechanics apply to the
safety term as to the reward term. Reward-critic and cost-critic value
losses are trained alongside with independent MSE terms, using their own
GAE-derived returns (`safelie.training.gae`, same gamma for both, §6.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from safelie.algos.networks import AgentBundle
from safelie.utils.config import PPOConfig


@dataclass
class PPOUpdateStats:
    policy_loss: float
    value_loss: float
    cost_value_loss: float
    entropy: float
    approx_kl: float


def ppo_lagrangian_update(
    agent: AgentBundle,
    rollout: dict,
    lam: float,
    cfg: PPOConfig,
) -> PPOUpdateStats:
    obs = torch.as_tensor(rollout["obs"], dtype=torch.float32)
    raw_actions = torch.as_tensor(rollout["raw_actions"], dtype=torch.float32)
    old_logprobs = torch.as_tensor(rollout["logprobs"], dtype=torch.float32)
    adv_r = torch.as_tensor(rollout["adv_r"], dtype=torch.float32)
    adv_c = torch.as_tensor(rollout["adv_c"], dtype=torch.float32)
    ret_r = torch.as_tensor(rollout["ret_r"], dtype=torch.float32)
    ret_c = torch.as_tensor(rollout["ret_c"], dtype=torch.float32)

    combined_adv = adv_r - lam * adv_c
    if combined_adv.std() > 1e-6:
        combined_adv = (combined_adv - combined_adv.mean()) / (combined_adv.std() + 1e-8)

    n = len(obs)
    minibatch_size = max(1, n // cfg.minibatches)
    stats: dict[str, list[float]] = {
        "policy_loss": [], "value_loss": [], "cost_value_loss": [], "entropy": [], "approx_kl": [],
    }

    for _ in range(cfg.epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, minibatch_size):
            idx = perm[start : start + minibatch_size]
            if len(idx) == 0:
                continue
            batch_obs = obs[idx]
            batch_actions = raw_actions[idx]
            batch_old_logprob = old_logprobs[idx]
            batch_adv = combined_adv[idx]
            batch_ret_r = ret_r[idx]
            batch_ret_c = ret_c[idx]

            new_logprob, entropy = agent.policy.evaluate(batch_obs, batch_actions)
            ratio = torch.exp(new_logprob - batch_old_logprob)
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * batch_adv
            policy_loss = -torch.min(surr1, surr2).mean() - cfg.entropy_coef * entropy.mean()

            value_pred = agent.value(batch_obs)
            cost_value_pred = agent.cost_value(batch_obs)
            value_loss = torch.nn.functional.mse_loss(value_pred, batch_ret_r)
            cost_value_loss = torch.nn.functional.mse_loss(cost_value_pred, batch_ret_c)

            loss = policy_loss + cfg.value_coef * (value_loss + cost_value_loss)

            agent.optimizer.zero_grad()
            loss.backward()
            params = list(agent.policy.parameters()) + list(agent.value.parameters()) + list(
                agent.cost_value.parameters()
            )
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            agent.optimizer.step()

            with torch.no_grad():
                approx_kl = float((batch_old_logprob - new_logprob).mean().item())

            stats["policy_loss"].append(float(policy_loss.item()))
            stats["value_loss"].append(float(value_loss.item()))
            stats["cost_value_loss"].append(float(cost_value_loss.item()))
            stats["entropy"].append(float(entropy.mean().item()))
            stats["approx_kl"].append(approx_kl)

    return PPOUpdateStats(
        policy_loss=float(np.mean(stats["policy_loss"])) if stats["policy_loss"] else 0.0,
        value_loss=float(np.mean(stats["value_loss"])) if stats["value_loss"] else 0.0,
        cost_value_loss=float(np.mean(stats["cost_value_loss"])) if stats["cost_value_loss"] else 0.0,
        entropy=float(np.mean(stats["entropy"])) if stats["entropy"] else 0.0,
        approx_kl=float(np.mean(stats["approx_kl"])) if stats["approx_kl"] else 0.0,
    )
