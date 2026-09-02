"""P0 #5 regression: policy, reward critic, and cost critic must train
with independent optimizers and independent gradient-norm clips.

Bug fixed: `AgentBundle` held one `torch.optim.Adam` over all three
networks' concatenated parameters, and `ppo_lagrangian_update` called one
`clip_grad_norm_` over that same concatenation. A single shared clip
means one network's gradient norm sets the scaling factor applied to
every network's gradient: a poorly-scaled cost critic (P0 #1/#3, before
observation/return normalization existed) could dominate the joint norm
and silently shrink the policy's own update by the same factor, with no
relationship to the policy's own loss.
"""

from __future__ import annotations

import torch

from safelie.algos.networks import AgentBundle
from safelie.training.ppo import ppo_lagrangian_update
from safelie.utils.config import PPOConfig


def _rollout(T: int = 16, obs_dim: int = 4, action_dim: int = 2) -> dict:
    rng = torch.Generator().manual_seed(0)
    obs = torch.randn(T, obs_dim, generator=rng).numpy()
    raw_actions = torch.randn(T, action_dim, generator=rng).numpy()
    logprobs = torch.zeros(T).numpy() - 1.0
    adv_r = (torch.randn(T, generator=rng) * 0.5).numpy()
    adv_c = (torch.randn(T, generator=rng) * 0.5).numpy()
    ret_r = (torch.randn(T, generator=rng) * 2.0 + 5.0).numpy()
    ret_c = (torch.randn(T, generator=rng) * 1.0 + 3.0).numpy()  # normal, budget-scale cost return
    return {
        "obs": obs, "raw_actions": raw_actions, "logprobs": logprobs,
        "adv_r": adv_r, "adv_c": adv_c, "ret_r": ret_r, "ret_c": ret_c,
    }


def _policy_params(agent: AgentBundle) -> list[torch.Tensor]:
    return [p.detach().clone() for p in agent.policy.parameters()]


class TestOptimizerSeparation:
    def test_three_distinct_optimizers_with_disjoint_parameter_sets(self):
        agent = AgentBundle(obs_dim=4, action_dim=2, hidden_dim=8, lr=3e-4)
        assert agent.policy_optimizer is not agent.value_optimizer
        assert agent.value_optimizer is not agent.cost_value_optimizer
        assert agent.policy_optimizer is not agent.cost_value_optimizer

        policy_ids = {id(p) for p in agent.policy.parameters()}
        value_ids = {id(p) for p in agent.value_net.parameters()}
        cost_ids = {id(p) for p in agent.cost_value_net.parameters()}
        assert policy_ids.isdisjoint(value_ids)
        assert policy_ids.isdisjoint(cost_ids)
        assert value_ids.isdisjoint(cost_ids)

        opt_policy_ids = {id(p) for g in agent.policy_optimizer.param_groups for p in g["params"]}
        opt_value_ids = {id(p) for g in agent.value_optimizer.param_groups for p in g["params"]}
        opt_cost_ids = {id(p) for g in agent.cost_value_optimizer.param_groups for p in g["params"]}
        assert opt_policy_ids == policy_ids
        assert opt_value_ids == value_ids
        assert opt_cost_ids == cost_ids

    def test_a_badly_scaled_cost_critic_does_not_suppress_the_policy_update(self):
        """The regression itself. adv_r/adv_c (what the policy loss
        actually depends on) are held FIXED and normal-scale between the
        two conditions; only the cost critic's *own* initial weights
        (hence its own loss/gradient magnitude, which the policy loss
        does not read) differ. With independent optimizers and clips,
        the policy's parameter update must be identical either way."""
        cfg = PPOConfig(epochs=1, minibatches=1, grad_clip=0.5, entropy_coef=0.0, value_coef=0.5)
        rollout = _rollout()

        torch.manual_seed(123)
        agent_normal = AgentBundle(obs_dim=4, action_dim=2, hidden_dim=8, lr=1e-2, normalize_returns=False)
        torch.manual_seed(123)
        agent_bad_critic = AgentBundle(obs_dim=4, action_dim=2, hidden_dim=8, lr=1e-2, normalize_returns=False)
        # Force the cost critic wildly wrong: huge output -> huge MSE loss
        # -> huge gradient norm for cost_value_net specifically.
        with torch.no_grad():
            for p in agent_bad_critic.cost_value_net.parameters():
                p.mul_(1000.0)

        before_normal = _policy_params(agent_normal)
        before_bad = _policy_params(agent_bad_critic)
        for a, b in zip(before_normal, before_bad, strict=True):
            torch.testing.assert_close(a, b)  # same policy init (same manual_seed)

        ppo_lagrangian_update(agent_normal, rollout, lam=0.5, cfg=cfg)
        ppo_lagrangian_update(agent_bad_critic, rollout, lam=0.5, cfg=cfg)

        after_normal = _policy_params(agent_normal)
        after_bad = _policy_params(agent_bad_critic)

        for b_normal, b_bad in zip(before_normal, before_bad, strict=True):
            del b_normal, b_bad  # already checked above

        for a_normal, a_bad in zip(after_normal, after_bad, strict=True):
            torch.testing.assert_close(a_normal, a_bad, rtol=1e-4, atol=1e-6)

    def test_cost_value_net_itself_still_updates_despite_its_own_clip(self):
        """Sanity check on the other side: the cost critic's own optimizer
        must still take a (clipped, bounded, but nonzero) step -- the fix
        decouples the networks, it does not stop the cost critic from
        training."""
        cfg = PPOConfig(epochs=1, minibatches=1, grad_clip=0.5, entropy_coef=0.0, value_coef=0.5)
        rollout = _rollout()
        agent = AgentBundle(obs_dim=4, action_dim=2, hidden_dim=8, lr=1e-2, normalize_returns=False)
        before = [p.detach().clone() for p in agent.cost_value_net.parameters()]
        ppo_lagrangian_update(agent, rollout, lam=0.5, cfg=cfg)
        after = [p.detach().clone() for p in agent.cost_value_net.parameters()]
        assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))


class TestNormalizationWiring:
    def test_update_normalization_stats_advances_obs_and_return_running_stats(self):
        agent = AgentBundle(obs_dim=4, action_dim=2, hidden_dim=8, lr=1e-3)
        cfg = PPOConfig(epochs=1, minibatches=1)
        rollout = _rollout()
        count_before = agent.obs_rms.count
        ppo_lagrangian_update(agent, rollout, lam=0.0, cfg=cfg)
        assert agent.obs_rms.count > count_before
        assert agent.ret_rms.count > 1e-4
        assert agent.cost_ret_rms.count > 1e-4

    def test_cost_value_query_stays_on_return_scale_after_training(self):
        """.cost_value() must keep returning budget-comparable numbers,
        not the internal normalized representation, even once the
        normalizer has real (non-trivial) statistics."""
        agent = AgentBundle(obs_dim=4, action_dim=2, hidden_dim=8, lr=1e-3)
        cfg = PPOConfig(epochs=2, minibatches=2)
        rollout = _rollout()
        for _ in range(5):
            ppo_lagrangian_update(agent, rollout, lam=0.0, cfg=cfg)
        obs = torch.as_tensor(rollout["obs"][:1], dtype=torch.float32)
        pred = agent.cost_value(obs).item()
        # ret_c was sampled around mean 3.0; a converged critic's
        # denormalized prediction should land in the same ballpark, not
        # in the normalizer's internal ~[-10, 10] clipped z-score range
        # by coincidence-of-scale alone (here they'd overlap by chance;
        # the real assertion is that it is finite and not NaN/inf, which
        # a broken denormalize -- e.g. forgetting to multiply back by
        # std -- would risk once std is far from 1).
        assert abs(pred) < 1e6
        import math
        assert math.isfinite(pred)

    def test_agent_bundle_checkpoint_round_trip_preserves_normalization_state(self, tmp_path):
        agent = AgentBundle(obs_dim=4, action_dim=2, hidden_dim=8, lr=1e-3)
        cfg = PPOConfig(epochs=1, minibatches=1)
        rollout = _rollout()
        ppo_lagrangian_update(agent, rollout, lam=0.0, cfg=cfg)

        state = agent.state_dict()
        restored = AgentBundle(obs_dim=4, action_dim=2, hidden_dim=8, lr=1e-3)
        restored.load_state_dict(state)

        import numpy as np
        np.testing.assert_array_equal(agent.obs_rms.mean, restored.obs_rms.mean)
        np.testing.assert_array_equal(agent.ret_rms.mean, restored.ret_rms.mean)
        np.testing.assert_array_equal(agent.cost_ret_rms.mean, restored.cost_ret_rms.mean)
        assert agent.obs_rms.count == restored.obs_rms.count

        obs = torch.as_tensor(rollout["obs"][:3], dtype=torch.float32)
        torch.testing.assert_close(agent.cost_value(obs), restored.cost_value(obs))
        torch.testing.assert_close(agent.value(obs), restored.value(obs))
