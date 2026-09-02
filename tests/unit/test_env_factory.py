"""The environment factory must never silently substitute environments.

This guards a real bug found while building this repository:
`ExperimentRun.__init__` originally hardcoded
`SyntheticConstrainedMarlEnv` regardless of `cfg.env.name`, so a
`manyagent_ant` config would train on the wrong environment instead of
raising -- see SMOKE_TEST_REPORT.md's "fixes applied" section.

`manyagent_ant` is now implemented (`safelie.envs.mamujoco`), so the guard
is no longer "it raises". It is the stronger and more durable property:
whatever `build_env` returns for a MuJoCo config, it is *not* the
synthetic stand-in, and if the optional dependency is missing the call
fails loudly rather than falling back. Those assertions hold whether or
not MuJoCo is installed, which is why they are not skipped.
"""

from __future__ import annotations

import pytest

from safelie.envs.factory import MAMUJOCO_ENV_NAMES, build_env
from safelie.envs.synthetic import SyntheticConstrainedMarlEnv
from safelie.sources.registry import default_m7_sources
from safelie.utils.config import EnvConfig, ExperimentConfig


def _backend_available() -> bool:
    from safelie.envs.mamujoco import available_backend

    try:
        available_backend()
        return True
    except ImportError:
        return False


requires_backend = pytest.mark.skipif(
    not _backend_available(), reason="no Safe MAMuJoCo backend installed"
)


def test_build_env_returns_synthetic_for_synthetic_name():
    cfg = EnvConfig(name="synthetic_constrained_marl", n_agents=3, budget=5.0)
    env = build_env(cfg, rollout_length=16)
    assert isinstance(env, SyntheticConstrainedMarlEnv)


def test_build_env_raises_for_unimplemented_environment():
    """Safety-Gymnasium navigation is goal-conditioned, not a MuJoCo
    factorization; the adapter does not cover it and must say so."""
    cfg = EnvConfig(name="safety_gym_nav", n_agents=6, budget=25.0)
    with pytest.raises(NotImplementedError, match="not implemented"):
        build_env(cfg, rollout_length=2000)


@pytest.mark.parametrize("name", sorted(MAMUJOCO_ENV_NAMES))
def test_mujoco_configs_never_fall_back_to_synthetic(name):
    """The regression itself. Either a real MuJoCo environment comes back,
    or the missing dependency is reported -- never a silent stand-in that
    would train for hours on the wrong dynamics."""
    n_agents = {"halfcheetah_2x3": 2, "ant_4x2": 4}.get(name, 6)
    cfg = EnvConfig(name=name, n_agents=n_agents, budget=25.0, velocity_threshold=1.0)
    try:
        env = build_env(cfg, rollout_length=64)
    except ImportError as exc:
        assert "install" in str(exc).lower()
        return
    assert not isinstance(env, SyntheticConstrainedMarlEnv)


def test_agent_count_mismatch_raises_rather_than_refactorizing():
    """N is what Theorem 1's spreading argument is stated over, so a
    config whose agent count disagrees with its factorization is a
    scientific error, not something to quietly coerce."""
    cfg = EnvConfig(name="halfcheetah_2x3", n_agents=6, budget=25.0)
    with pytest.raises((ValueError, ImportError)):
        build_env(cfg, rollout_length=64)


@requires_backend
def test_manyagent_ant_satisfies_the_dual_cost_contract():
    cfg = EnvConfig(
        name="manyagent_ant", n_agents=6, budget=25.0, velocity_threshold=1.0
    )
    env = build_env(cfg, rollout_length=32)

    assert env.n_agents == 6
    assert len(env.agent_ids) == 6
    assert env.obs_dim > 0 and env.action_dim > 0

    step = env.reset(seed=0)
    # Uniform observation width matters downstream: `training.loop`
    # evaluates a peer's cost critic on the owner's observation.
    assert {o.shape for o in step.obs.values()} == {(env.obs_dim,)}

    import numpy as np

    step = env.step({aid: np.zeros(env.action_dim, dtype=np.float32) for aid in env.agent_ids})
    assert isinstance(step.reward, float)
    assert set(step.reported_cost) == set(env.agent_ids)
    assert all(c >= 0.0 for c in step.reported_cost.values())


@requires_backend
def test_experiment_run_builds_networks_from_the_environment_not_the_config(tmp_path):
    """The dimension bug: `EnvConfig.obs_dim`/`action_dim` default to 8/2
    and the pilot configs never set them, so sizing networks from the
    config produced 8-dim policies for 63-dim observations."""
    from safelie.training.loop import ExperimentRun

    cfg = ExperimentConfig(
        run_id="dim_regression_test",
        env={
            "name": "manyagent_ant",
            "n_agents": 6,
            "budget": 25.0,
            "velocity_threshold": 1.0,
        },
        topology={"name": "ring", "n_agents": 6},
        sources=default_m7_sources(),
        total_steps=64,
        rollout_length=32,
        output_dir=str(tmp_path / "runs"),
    )
    run = ExperimentRun(cfg)

    assert run.obs_dim == run.env.obs_dim
    assert run.action_dim == run.env.action_dim
    assert run.obs_dim != cfg.env.obs_dim, "test is vacuous if the config default matches"

    bundle = next(iter(run.agents.values()))
    assert bundle.policy.mean_net[0].in_features == run.env.obs_dim
    assert bundle.policy.mean_net[-1].out_features == run.env.action_dim


@requires_backend
def test_mamujoco_rollouts_are_reproducible_across_episode_boundaries():
    """MaMuJoCo terminates on an unhealthy state while the learner steps a
    fixed rollout length, so the adapter restarts episodes internally. Those
    restarts draw their own seeds; if they came from anywhere but this
    object's own generator, two same-seed rollouts would diverge the first
    time an episode ended and the repository's determinism guarantee would
    hold only for runs that never fell over."""
    import numpy as np

    cfg = EnvConfig(name="manyagent_ant", n_agents=6, budget=25.0, velocity_threshold=1.0)

    def rollout():
        env = build_env(cfg, rollout_length=64)
        step = env.reset(seed=7)
        actions = np.full(env.action_dim, 0.9, dtype=np.float32)  # drives terminations
        trace = []
        for _ in range(200):
            step = env.step(dict.fromkeys(env.agent_ids, actions))
            trace.append(
                (
                    step.reward,
                    tuple(step.reported_cost[a] for a in env.agent_ids),
                    tuple(float(step.obs[a][0]) for a in env.agent_ids),
                )
            )
        return trace

    first, second = rollout(), rollout()
    assert first == second
    assert len({t[0] for t in first}) > 1, "test is vacuous if the episode never advances"


@requires_backend
def test_concurrent_construction_does_not_race_on_the_shared_asset_file():
    """MaMuJoCo generates ManySegmentAnt's model XML to a FIXED path inside
    its own package, loads it, then deletes it. Two processes building the
    same scenario race on that file and the loser sees

        ValueError: ParseXML: empty file '...6_segments.auto.xml'

    This bit for real: launching four pilot runs at once killed three of
    them at startup. It is not a startup-only hazard either -- the oracle
    builds a fresh environment every round, so a 250-round run touches that
    path 250 times and any concurrency makes a mid-run crash a matter of
    time. `_asset_generation_lock` serializes construction across
    processes; this test fails without it.
    """
    import subprocess
    import sys
    import textwrap

    worker = textwrap.dedent(
        """
        from safelie.envs.mamujoco import MaMuJoCoDualCostEnv
        for i in range(4):
            env = MaMuJoCoDualCostEnv(
                "manyagent_ant", 6, 25.0, 32, velocity_threshold=0.75
            )
            env.reset(seed=i)
        print("OK")
        """
    )
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", worker],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]
    results = [p.communicate() for p in procs]

    for i, ((out, err), proc) in enumerate(zip(results, procs, strict=True)):
        assert proc.returncode == 0, f"worker {i} died:\n{err[-2000:]}"
        assert "OK" in out
