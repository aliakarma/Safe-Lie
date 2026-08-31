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
    return the oracle's true-cost accounting for it. Deterministic given
    `seed` and the current policy weights."""
    env = env_factory()
    step = env.reset(seed=seed)
    oracle = OracleEvaluator(env=env, gamma=gamma, budget=budget)

    for _ in range(rollout_length):
        actions: dict[AgentID, np.ndarray] = {}
        for aid in env.agent_ids:
            obs_t = torch.as_tensor(step.obs[aid], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _ = agents[aid].policy.act(obs_t)
            actions[aid] = action.squeeze(0).numpy()
        step = env.step(actions)
        oracle.record()

    return oracle.episode_result()
