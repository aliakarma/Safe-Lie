"""Single dispatch point from `EnvConfig.name` to a concrete environment.

Report reference: PROJECT_REPORT.md Phase 1. This exists so that a config
naming a real environment (`manyagent_ant`, `halfcheetah_2x3`,
`safety_gym_nav`) fails loudly and immediately with `NotImplementedError`
rather than silently falling back to the synthetic stand-in — which is
exactly the kind of silent substitution that would make a Stage-2 pilot
run's results meaningless without anyone noticing.
"""

from __future__ import annotations

from safelie.envs.dual_cost import DualCostEnvWrapper
from safelie.envs.synthetic import SyntheticConstrainedMarlEnv
from safelie.utils.config import EnvConfig


def build_env(env_cfg: EnvConfig, rollout_length: int) -> DualCostEnvWrapper:
    if env_cfg.name == "synthetic_constrained_marl":
        return SyntheticConstrainedMarlEnv(
            n_agents=env_cfg.n_agents,
            budget=env_cfg.budget,
            obs_dim=env_cfg.obs_dim,
            action_dim=env_cfg.action_dim,
            horizon=rollout_length,
        )
    raise NotImplementedError(
        f"env.name='{env_cfg.name}' requires an environment adapter not "
        f"implemented in this repository. See safelie/envs/mamujoco.py for "
        f"what is needed and why it was left unimplemented (a heavy, "
        f"GPU-stage dependency out of scope for local CPU verification, "
        f"per PROJECT_REPORT.md §R7.1)."
    )
