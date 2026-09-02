"""Single dispatch point from `EnvConfig.name` to a concrete environment.

Report reference: PROJECT_REPORT.md Phase 1. This exists so that a config
naming a real environment never silently falls back to the synthetic
stand-in -- which is exactly the kind of silent substitution that would
make a Stage-2 pilot run's results meaningless without anyone noticing.

The MuJoCo-backed names (`manyagent_ant`, `halfcheetah_2x3`,
`halfcheetah_6x1`, `ant_4x2`) are served by `safelie.envs.mamujoco`, which
needs an optional dependency. A missing dependency raises `ImportError`
with install instructions; it never degrades to the synthetic environment.
`safety_gym_nav` remains unimplemented: Safety-Gymnasium's navigation
tasks are goal-conditioned with a different agent/observation structure,
not a MuJoCo factorization, so nothing here would fit them.
"""

from __future__ import annotations

from safelie.envs.dual_cost import DualCostEnvWrapper
from safelie.envs.synthetic import SyntheticConstrainedMarlEnv
from safelie.utils.config import EnvConfig

MAMUJOCO_ENV_NAMES = frozenset(
    {"manyagent_ant", "halfcheetah_2x3", "halfcheetah_6x1", "ant_4x2"}
)


def build_env(env_cfg: EnvConfig, rollout_length: int) -> DualCostEnvWrapper:
    if env_cfg.name == "synthetic_constrained_marl":
        return SyntheticConstrainedMarlEnv(
            n_agents=env_cfg.n_agents,
            budget=env_cfg.budget,
            obs_dim=env_cfg.obs_dim,
            action_dim=env_cfg.action_dim,
            horizon=rollout_length,
        )
    if env_cfg.name in MAMUJOCO_ENV_NAMES:
        from safelie.envs.mamujoco import build_mamujoco_env

        return build_mamujoco_env(env_cfg, rollout_length=rollout_length)
    raise NotImplementedError(
        f"env.name='{env_cfg.name}' requires an environment adapter not implemented in "
        f"this repository. Safety-Gymnasium's navigation tasks are goal-conditioned "
        f"rather than a MuJoCo factorization, so the Safe MAMuJoCo adapter "
        f"(safelie/envs/mamujoco.py) does not cover them."
    )
