"""The withheld-evaluator harness: runs an independent rollout of the
current (frozen) policies and measures true cost.

Report reference: PROJECT_REPORT.md §4.1 (C11 "Oracle evaluator ... code
path the learner provably cannot influence", listed as a component
distinct from C12 "Experiment orchestrator" and C1-C10 the learner
proper); smoke test S10.

This module — not `safelie.training` — is the only place in the pipeline
allowed to import and use `OracleEvaluator`. It performs its own,
separate environment rollout (a fresh instance, a fresh seed) using the
current policy weights with gradients disabled; it never touches the
learner's rollout buffer, residuals, aggregator, or dual state. The
top-level orchestrator (`safelie.experiment`) calls this once per round
*after* `safelie.training.loop.ExperimentRun.run_round()` has already
updated the policies, and merges the two independently-produced records
before logging — exactly the "oracle block ... written by the evaluator,
never the learner" schema rule (PROJECT_REPORT.md §8.2).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import torch

from safelie.algos.networks import AgentBundle
from safelie.envs.dual_cost import AgentID
from safelie.eval.oracle import OracleCapableEnv, OracleEpisodeResult, OracleEvaluator


def evaluate_true_cost(
    env_factory: Callable[[], OracleCapableEnv],
    agents: dict[AgentID, AgentBundle],
    gamma: float,
    budget: float,
    rollout_length: int,
    seed: int,
) -> OracleEpisodeResult:
    """Roll out the current (frozen) policies for one fresh episode and
    return the oracle's full evaluation accounting for it: true cost
    (privileged), plus task return and reported-cost return (both P0 #6
    additions, computed here rather than in `OracleEvaluator` because they
    read only the ordinary, learner-visible `DualCostStep` fields, not the
    privileged handle). All three are proper discounted Monte-Carlo sums
    over this one fresh episode -- never a GAE(lambda) training target,
    and never mixed with anything from the learner's own rollout.
    Deterministic given `seed` and the current policy weights.
    """
    env = env_factory()
    step = env.reset(seed=seed)
    oracle = OracleEvaluator(env=env, gamma=gamma, budget=budget)

    discounted_task_return = 0.0
    discounted_reported_cost = dict.fromkeys(env.agent_ids, 0.0)

    for t in range(rollout_length):
        actions: dict[AgentID, np.ndarray] = {}
        for aid in env.agent_ids:
            obs_t = torch.as_tensor(step.obs[aid], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                # P0 #2: the policy was trained on normalized
                # observations (safelie.training.ppo) -- evaluating it on
                # raw ones here would silently feed it an out-of-
                # distribution input and invalidate this oracle rollout.
                obs_n = agents[aid].normalize_obs_tensor(obs_t)
                action, _ = agents[aid].policy.act(obs_n)
            actions[aid] = action.squeeze(0).numpy()
        step = env.step(actions)
        oracle.record()
        discount = gamma**t
        discounted_task_return += discount * step.reward
        for aid in env.agent_ids:
            discounted_reported_cost[aid] += discount * step.reported_cost[aid]

    return dataclasses.replace(
        oracle.episode_result(),
        episodic_task_return=float(discounted_task_return),
        episodic_reported_cost_return={aid: float(v) for aid, v in discounted_reported_cost.items()},
    )
