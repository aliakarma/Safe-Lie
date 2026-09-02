"""Safe Multi-Agent MuJoCo / Safety-Gymnasium adapter.

Report reference: main_iclr.tex §5.1 ("Environments. Safe Multi-Agent
MuJoCo, primarily ManyAgent Ant (N=6) and HalfCheetah 2x3 (N=2), plus the
multi-agent tasks of Safety-Gymnasium"); PROJECT_REPORT.md §R7.1.

**Status: implemented, with three documented deviations forced by the
reference implementation itself.** Earlier builds of this repository left
this module unimplemented and classified the gap (D) "not implementable
from the materials available". That classification was about dependency
weight, and it was wrong about the science: installing the dependencies
and reading the reference implementation turns up three facts that no
amount of GPU time resolves, and that a silent "just wrap it" adapter
would have buried. They are recorded here because they change what the
pilot is entitled to claim.

Two backends satisfy `safelie.envs.dual_cost.DualCostEnvWrapper`:

  ``safety_gymnasium``  (backend "safe_mamujoco", preferred)
      The reference Safe MAMuJoCo of Gu et al., vendored inside
      `safety_gymnasium.tasks.safe_multi_agent.safe_mujoco_multi`. Its
      cost signal is [SPEC]-faithful because it *is* the spec.

  ``gymnasium_robotics``  (backend "mamujoco", portable fallback)
      Farama's maintained MaMuJoCo. Supplies the factorized physics but
      **no cost signal at all**; this module supplies one (see
      `cost_mode` below), which is a `[DECISION]`, not a `[SPEC]`.

The two cannot be installed side by side: `safety-gymnasium==1.0.0` pins
`gymnasium==0.28.1`, `gymnasium-robotics==1.2.2`, `mujoco==2.3.3`, all of
which conflict with the modern stack the fallback needs. Backend
selection is therefore an install-time fact, not a runtime switch, and
every run records which backend produced its numbers in
`DualCostStep.info["backend"]` and in the round log.

Deviation 1 -- **ManyAgent Ant is not a Safe MAMuJoCo environment.**
    `safe_mujoco_multi.TASK_VELCITY_THRESHOLD` covers exactly
    Ant {2x4, 4x2}, HalfCheetah {6x1, 2x3}, Hopper 3x1, Humanoid 9|8,
    Swimmer 2x1, Walker2d 2x3 -- and `SafeMAEnv.__init__` opens with
    `assert scenario in TASK_VELCITY_THRESHOLD`. `ManySegmentAnt`, the
    "ManyAgent Ant" of §5.1, is absent, so the paper's headline N=6
    environment **cannot run on the reference Safe MAMuJoCo at all**. It
    runs here only on the `mamujoco` backend, with this module's own cost
    function. `halfcheetah_6x1` is offered alongside it as the one
    genuinely N=6 configuration the reference implementation does
    support, so the pilot has a [SPEC]-faithful N=6 option.

Deviation 2 -- **Safe MAMuJoCo's cost is shared, not per-agent.** Its
    `step` computes one global speed indicator from the *torso* velocity
    and assigns the identical scalar to every agent::

        velocity = sqrt(x_velocity**2 + y_velocity**2)
        cost_n   = float(velocity > threshold)
        costs    = {agent: cost_n for agent in agents}

    The paper's Definition 1 constrains a per-agent C^i, and `[GAP]` G4
    asks how one agent's cost could be observable to a peer. Under the
    reference cost function the question is vacuous: every agent's cost
    is the same number, so peer observation is trivial and the per-agent
    constraint structure collapses to a single shared constraint. Both
    behaviours are available here via `cost_mode`, defaulting to
    whichever is faithful for the backend in use.

Deviation 3 -- **The reference thresholds do not bind at this pilot's
    scale.** Safe MAMuJoCo's thresholds are calibrated to roughly half
    the terminal speed of a *converged* unconstrained PPO agent. Measured
    directly on this repository's own policies (see
    `scripts/calibrate_cost.py`, and docs/assumptions.md for the
    numbers), a randomly-initialized ManySegmentAnt 6x1 policy exceeds
    Ant's 2.418 threshold on 0.75% of steps, giving a discounted cost
    return near 0.75 against the paper's budget d=25. lambda never lifts
    off zero, the constraint never binds, and all five pilot conditions
    become indistinguishable for reasons that have nothing to do with the
    hypothesis -- the precondition failure PROJECT_REPORT.md §R6.1 warns
    about, and the same one that forced `local_demo_clean.yaml`'s budget
    down to d=5 on the synthetic environment. Because §5.1 specifies no
    threshold for ManyAgent Ant, the threshold (not the paper's d=25) is
    the free parameter here, and it is calibrated by direct measurement.
    Run `scripts/calibrate_cost.py` before trusting any pilot number.

Everything downstream of `DualCostEnvWrapper` -- attacks, defenses,
source accounting, dual update, oracle isolation -- is unchanged and
environment-agnostic, exactly as the previous docstring promised.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np

from safelie.envs.dual_cost import AgentID, DualCostStep
from safelie.envs.guards import OracleReadOnlyView, _PrivilegedOracleView

CostMode = Literal["auto", "safe_mamujoco_shared", "per_agent_velocity"]
Backend = Literal["auto", "safe_mamujoco", "mamujoco"]

# Verbatim from safety_gymnasium.tasks.safe_multi_agent.safe_mujoco_multi
# .TASK_VELCITY_THRESHOLD (safety-gymnasium 1.0.0). [SPEC] for every pair
# listed; note the absence of any ManySegmentAnt entry -- Deviation 1.
SAFE_MAMUJOCO_VELOCITY_THRESHOLD: dict[tuple[str, str], float] = {
    ("Ant", "2x4"): 2.522,
    ("Ant", "4x2"): 2.418,
    ("HalfCheetah", "6x1"): 2.932,
    ("HalfCheetah", "2x3"): 3.227,
    ("Hopper", "3x1"): 0.9613,
    ("Humanoid", "9|8"): 0.58,
    ("Swimmer", "2x1"): 0.04891,
    ("Walker2d", "2x3"): 1.641,
}

# `[DECISION]` Calibrated by direct measurement rather than inherited from
# the table above, because §5.1 specifies no ManyAgent Ant threshold and
# the nearest table entry (Ant 4x2 = 2.418) leaves the constraint
# non-binding at d=25 -- Deviation 3. See scripts/calibrate_cost.py.
#
# Set to 0.75 after a completed 250-round condition-A run at 1.0 showed
# that clearing the initial-calibration bar is not sufficient: at 1.0 the
# true cost binds from round 0 (0.80x budget) but the learner's estimate
# does not cross d=25 until round ~159 of 250, leaving lambda positive in
# only 14% of (round, agent) cells and peaking at 6.7% of lambda_max. At
# 0.75 the true cost is 1.66x budget and the initial estimate 0.42x rather
# than 0.23x, so the estimate crosses the budget far earlier. See
# docs/assumptions.md for the full measurement.
CALIBRATED_VELOCITY_THRESHOLD: dict[tuple[str, str], float] = {
    ("ManySegmentAnt", "6x1"): 0.75,
}

# `EnvConfig.name` -> (scenario, agent_conf). `manyagent_ant` is the only
# entry the reference Safe MAMuJoCo cannot serve (Deviation 1).
_SCENARIO_MAP: dict[str, tuple[str, str | None, int | None]] = {
    "manyagent_ant": ("ManySegmentAnt", None, None),  # agent_conf built from n_agents
    "halfcheetah_2x3": ("HalfCheetah", "2x3", 2),
    "halfcheetah_6x1": ("HalfCheetah", "6x1", 6),
    "ant_4x2": ("Ant", "4x2", 4),
}


def scenario_for(env_name: str, n_agents: int) -> tuple[str, str]:
    """Map an `EnvConfig.name` onto a (scenario, agent_conf) pair."""
    if env_name not in _SCENARIO_MAP:
        raise NotImplementedError(
            f"env.name={env_name!r} has no Safe MAMuJoCo scenario mapping. Supported: "
            f"{', '.join(sorted(_SCENARIO_MAP))}."
        )
    scenario, agent_conf, required_n = _SCENARIO_MAP[env_name]
    if agent_conf is None:
        agent_conf = f"{n_agents}x1"
    elif required_n is not None and n_agents != required_n:
        raise ValueError(
            f"env.name={env_name!r} is an N={required_n} factorization, but "
            f"env.n_agents={n_agents}. Fix the config rather than the factorization: "
            f"the agent count is what Theorem 1's spreading argument is stated over."
        )
    return scenario, agent_conf


def resolve_velocity_threshold(
    scenario: str, agent_conf: str, override: float | None = None
) -> float:
    """The speed above which a step incurs unit cost.

    Prefers an explicit config override, then Safe MAMuJoCo's own [SPEC]
    table, then this repository's calibrated values. Raises rather than
    guessing: an uncalibrated threshold silently produces a non-binding
    constraint, which looks like a clean null result instead of the
    precondition failure it actually is (§R6.1).
    """
    if override is not None:
        return float(override)
    key = (scenario, agent_conf)
    if key in SAFE_MAMUJOCO_VELOCITY_THRESHOLD:
        return SAFE_MAMUJOCO_VELOCITY_THRESHOLD[key]
    if key in CALIBRATED_VELOCITY_THRESHOLD:
        return CALIBRATED_VELOCITY_THRESHOLD[key]
    raise ValueError(
        f"No velocity threshold known for scenario={scenario!r} agent_conf={agent_conf!r}. "
        f"Safe MAMuJoCo specifies none and this repository has not calibrated one. Set "
        f"env.velocity_threshold explicitly, after running scripts/calibrate_cost.py to "
        f"confirm the constraint actually binds at your budget (PROJECT_REPORT.md §R6.1)."
    )


def available_backend(preference: Backend = "auto") -> str:
    """Which backend this interpreter can actually construct.

    Import-tested, not merely name-checked: `safety_gymnasium` is
    importable only on Linux/Colab in practice (it depends on `pygame`,
    which has no Windows wheel at its pinned version).
    """
    def _has(module: str) -> bool:
        import importlib.util

        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False

    if preference == "safe_mamujoco":
        if not _has("safety_gymnasium"):
            raise ImportError(
                "backend='safe_mamujoco' requires safety-gymnasium, which is not "
                "installed. `pip install safety-gymnasium==1.0.0` (Linux/Colab only; "
                "it pins gymnasium==0.28.1 and will conflict with a modern "
                "gymnasium-robotics install)."
            )
        return "safe_mamujoco"
    if preference == "mamujoco":
        if not _has("gymnasium_robotics"):
            raise ImportError(
                "backend='mamujoco' requires gymnasium-robotics. "
                'Install it with `pip install "safelie[mujoco]"`.'
            )
        return "mamujoco"
    if _has("safety_gymnasium"):
        return "safe_mamujoco"
    if _has("gymnasium_robotics"):
        return "mamujoco"
    raise ImportError(
        "No Safe MAMuJoCo backend is installed. Install either "
        "`safety-gymnasium==1.0.0` (the reference implementation; Linux/Colab only) "
        'or `pip install "safelie[mujoco]"` (gymnasium-robotics; portable, but '
        "supplies no cost signal of its own -- see safelie.envs.mamujoco's docstring)."
    )


@contextmanager
def _asset_generation_lock(scenario: str, timeout: float = 300.0) -> Iterator[None]:
    """Cross-process mutex around backend environment construction.

    Guards the fixed-filename model-XML generation described in
    `MaMuJoCoDualCostEnv._build_backend_env`. The lock lives in the system
    temp directory rather than beside the asset, so it works when
    site-packages is read-only.

`filelock` ships with torch, so it is present in practice; if it is
    ever absent this degrades to no locking, and the retry in
    `_build_backend_env` is what keeps construction correct.
    """
    lock_path = Path(tempfile.gettempdir()) / f"safelie-mamujoco-{scenario}.lock"
    try:
        from filelock import FileLock
    except ImportError:
        yield
        return
    with FileLock(str(lock_path), timeout=timeout):
        yield


def _unwrap_to_mujoco(obj: Any) -> Any:
    """Walk down to the raw `MujocoEnv` holding `.model` / `.data`."""
    base = obj
    for attr in ("single_agent_env",):
        nxt = getattr(base, attr, None)
        if nxt is None and hasattr(base, "unwrapped"):
            nxt = getattr(base.unwrapped, attr, None)
        if nxt is not None:
            base = nxt
            break
    seen: set[int] = set()
    while hasattr(base, "unwrapped") and base.unwrapped is not base and id(base) not in seen:
        seen.add(id(base))
        base = base.unwrapped
    if not (hasattr(base, "model") and hasattr(base, "data")):
        raise AttributeError(
            f"Could not reach the MuJoCo model/data from {type(obj).__name__}; the "
            f"per-agent cost function needs them (see cost_mode='per_agent_velocity')."
        )
    return base


class MaMuJoCoDualCostEnv:
    """Safe MAMuJoCo behind the `DualCostEnvWrapper` contract.

    Two structural adaptations, both forced by the underlying env and both
    invisible to everything downstream:

    *Observation padding.* MaMuJoCo hands each agent a differently-sized
    observation (ManySegmentAnt 6x1: 61 for agent_0, 63 for the rest,
    because the head segment has no predecessor). The pipeline sizes one
    network shape per run and, in `safelie.training.loop`, evaluates a
    *peer's* cost critic on the *owner's* observation -- which is
    dimensionally impossible unless every agent's observation has the same
    width. Observations are therefore zero-padded to `max_i obs_dim_i`.
    `[DECISION]`: padding, not truncation, so no agent loses information.

    *Auto-reset.* MaMuJoCo terminates on an unhealthy state; the learner
    steps a fixed `rollout_length` regardless. On termination the episode
    is restarted internally and the step is flagged `terminated`, so GAE
    cuts the bootstrap at the boundary (`safelie.training.gae`) exactly as
    it would at a real episode end. Reset seeds are drawn from this
    object's own generator, seeded from `reset(seed=...)`, so a rollout
    stays reproducible across mid-episode restarts.
    """

    def __init__(
        self,
        env_name: str,
        n_agents: int,
        budget: float,
        rollout_length: int,
        agent_obsk: int = 1,
        cost_mode: CostMode = "auto",
        velocity_threshold: float | None = None,
        backend: Backend = "auto",
    ):
        self.scenario, self.agent_conf = scenario_for(env_name, n_agents)
        self.backend = available_backend(backend)
        self.velocity_threshold = resolve_velocity_threshold(
            self.scenario, self.agent_conf, velocity_threshold
        )

        self.agent_obsk = int(agent_obsk)

        if cost_mode == "auto":
            # Faithful-by-default: the reference cost on the reference
            # backend, the per-agent [DECISION] only where no reference
            # cost exists to be faithful to.
            cost_mode = (
                "safe_mamujoco_shared" if self.backend == "safe_mamujoco" else "per_agent_velocity"
            )
        self.cost_mode: str = cost_mode

        self._env = self._build_backend_env()
        self._backend_agents: list[str] = list(self._env.possible_agents)
        if len(self._backend_agents) != n_agents:
            raise ValueError(
                f"{self.scenario} {self.agent_conf} factorizes into "
                f"{len(self._backend_agents)} agents, but env.n_agents={n_agents}."
            )

        self.n_agents = n_agents
        self.budget = budget
        self.horizon = rollout_length
        self.agent_ids: list[AgentID] = [f"agent_{i}" for i in range(n_agents)]
        self._id_to_backend = dict(zip(self.agent_ids, self._backend_agents, strict=True))

        obs_dims = [int(self._env.observation_space(a).shape[0]) for a in self._backend_agents]
        act_dims = [int(self._env.action_space(a).shape[0]) for a in self._backend_agents]
        self.obs_dim = max(obs_dims)
        self.action_dim = max(act_dims)
        self._obs_dims = dict(zip(self.agent_ids, obs_dims, strict=True))
        self._act_dims = dict(zip(self.agent_ids, act_dims, strict=True))

        self._mj = _unwrap_to_mujoco(self._env)
        self._torso_body_ids = self._find_torso_bodies()

        self._rng = np.random.default_rng(0)
        self._t = 0
        self._last_true_cost: dict[AgentID, float] = dict.fromkeys(self.agent_ids, 0.0)

    # -- construction -------------------------------------------------

    def _build_backend_env(self) -> Any:
        # Serialized: MaMuJoCo's ManySegmentAnt/ManySegmentSwimmer path
        # generates its model XML to a FIXED filename inside its own
        # package directory, loads it, then deletes it
        # (`mujoco_multi._create_base_gym_env`). Two processes building the
        # same scenario race on that one file, and the loser reads a
        # half-written or already-deleted XML:
        #
        #     ValueError: ParseXML: empty file '...many_segment_ant_6_segments.auto.xml'
        #
        # This is not a startup-only hazard. `safelie.experiment` builds a
        # fresh environment for every oracle evaluation episode, so a
        # 250-round run touches that path 250 times and any concurrency --
        # several seeds or conditions sharing a machine, which is the
        # obvious way to use the pilot matrix -- makes a mid-run crash a
        # matter of time. Construction takes about a second, so serializing
        # it costs nothing measurable.
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                with _asset_generation_lock(self.scenario):
                    return self._construct()
            except (ValueError, OSError) as exc:
                # A lost race surfaces as a parse error on a file that
                # exists again moments later, so a bounded retry recovers.
                # Narrow on purpose: a genuinely malformed scenario raises
                # the same way every time and still fails after 5 tries.
                last_exc = exc
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(
            f"Could not construct {self.scenario} {self.agent_conf} after 5 attempts. "
            f"If this is a fixed-asset race it should have cleared; see "
            f"MaMuJoCoDualCostEnv._build_backend_env. Last error: {last_exc}"
        ) from last_exc

    def _construct(self) -> Any:
        if self.backend == "safe_mamujoco":
            from safety_gymnasium.tasks.safe_multi_agent.safe_mujoco_multi import make_ma

            return make_ma(
                scenario=self.scenario,
                agent_conf=self.agent_conf,
                agent_obsk=self.agent_obsk,
            )
        from gymnasium_robotics import mamujoco_v1

        return mamujoco_v1.parallel_env(
            scenario=self.scenario,
            agent_conf=self.agent_conf,
            agent_obsk=self.agent_obsk,
        )

    def _find_torso_bodies(self) -> list[int]:
        """Body ids of each agent's own torso segment, for the per-agent
        cost. Empty when the model has no per-agent torso (e.g. HalfCheetah
        is one body chain), in which case `per_agent_velocity` is refused
        rather than silently degraded to the shared cost."""
        import mujoco

        model = self._mj.model
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)
        ]
        ids = [i for i, n in enumerate(names) if n and n.startswith("torso_")]
        if len(ids) != self.n_agents:
            return []
        return ids

    # -- DualCostEnvWrapper -------------------------------------------

    def reset(self, seed: int | None = None) -> DualCostStep:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, _info = self._env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))
        self._t = 0
        self._last_true_cost = dict.fromkeys(self.agent_ids, 0.0)
        return DualCostStep(
            obs=self._pad_obs(obs),
            reward=0.0,
            reported_cost=dict(self._last_true_cost),
            terminated=dict.fromkeys(self.agent_ids, False),
            truncated=dict.fromkeys(self.agent_ids, False),
            info={"backend": self.backend, "cost_mode": self.cost_mode},
        )

    def step(self, actions: dict[AgentID, np.ndarray]) -> DualCostStep:
        backend_actions = {
            self._id_to_backend[aid]: np.clip(
                np.asarray(actions[aid], dtype=np.float32)[: self._act_dims[aid]], -1.0, 1.0
            )
            for aid in self.agent_ids
        }

        stepped = self._env.step(backend_actions)
        if len(stepped) == 6:  # safety_gymnasium: (obs, rew, cost, term, trunc, info)
            obs, rewards, native_costs, terminated, truncated, info = stepped
        else:  # gymnasium_robotics: (obs, rew, term, trunc, info)
            obs, rewards, terminated, truncated, info = stepped
            native_costs = None

        reward = float(np.mean([float(v) for v in rewards.values()]))
        true_cost = self._compute_cost(info, native_costs)
        self._last_true_cost = true_cost

        term = {aid: bool(terminated[self._id_to_backend[aid]]) for aid in self.agent_ids}
        trunc = {aid: bool(truncated[self._id_to_backend[aid]]) for aid in self.agent_ids}

        self._t += 1
        info: dict[str, Any] = {"t": self._t, "backend": self.backend, "cost_mode": self.cost_mode}
        if any(term.values()) or any(trunc.values()):
            # Gymnasium's own convention for auto-resetting vector envs
            # (`info["final_observation"]`): expose the TRUE pre-reset
            # observation before swapping in the new episode's, so a
            # caller can bootstrap a truncated (not terminated) episode's
            # value from where it actually ended rather than from the
            # unrelated post-reset state -- see
            # `safelie.training.gae.compute_gae`'s docstring for why this
            # is not optional for a time-limit truncation.
            info["final_observation"] = self._pad_obs(obs)
            obs, _ = self._env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))

        return DualCostStep(
            obs=self._pad_obs(obs),
            reward=reward,
            reported_cost=dict(true_cost),
            terminated=term,
            truncated=trunc,
            info=info,
        )

    def oracle_handle(self) -> OracleReadOnlyView:
        """Public method any code can call. Always returns a sealed
        (raising) view; see `_oracle_handle_privileged`."""
        return OracleReadOnlyView(get_true_cost=lambda: self._last_true_cost)

    def _oracle_handle_privileged(self) -> _PrivilegedOracleView:
        """Not part of `DualCostEnvWrapper`. Only
        `safelie.eval.oracle.OracleEvaluator` calls this."""
        return _PrivilegedOracleView(get_true_cost=lambda: self._last_true_cost)

    # -- internals ----------------------------------------------------

    def _pad_obs(self, obs: dict[str, np.ndarray]) -> dict[AgentID, np.ndarray]:
        padded: dict[AgentID, np.ndarray] = {}
        for aid in self.agent_ids:
            raw = np.asarray(obs[self._id_to_backend[aid]], dtype=np.float32)
            if raw.shape[0] < self.obs_dim:
                buf = np.zeros(self.obs_dim, dtype=np.float32)
                buf[: raw.shape[0]] = raw
                raw = buf
            padded[aid] = raw
        return padded

    def _compute_cost(
        self, info: dict, native_costs: dict[str, float] | None
    ) -> dict[AgentID, float]:
        if self.cost_mode == "safe_mamujoco_shared":
            if native_costs is not None:
                return {
                    aid: float(native_costs[self._id_to_backend[aid]]) for aid in self.agent_ids
                }
            # Reproduce safe_mujoco_multi.SafeMAEnv.step's formula exactly
            # on the fallback backend, so the two are comparable.
            single = info[self._backend_agents[0]]
            velocity = float(
                np.hypot(single["x_velocity"], single.get("y_velocity", 0.0))
                if self.scenario != "Swimmer"
                else single["x_velocity"]
            )
            shared = float(velocity > self.velocity_threshold)
            return dict.fromkeys(self.agent_ids, shared)

        # per_agent_velocity: the `[DECISION]` resolving [GAP] G4 -- each
        # agent's cost is the speed of its own torso segment, read from
        # shared simulator state (so a peer could in principle measure it,
        # which is exactly what G4 asks for), thresholded identically.
        if not self._torso_body_ids:
            raise ValueError(
                f"cost_mode='per_agent_velocity' needs one 'torso_<i>' body per agent, but "
                f"{self.scenario} {self.agent_conf} has none. Use "
                f"cost_mode='safe_mamujoco_shared' for this scenario."
            )
        import mujoco

        mujoco.mj_subtreeVel(self._mj.model, self._mj.data)
        return {
            aid: float(
                np.linalg.norm(self._mj.data.subtree_linvel[body_id]) > self.velocity_threshold
            )
            for aid, body_id in zip(self.agent_ids, self._torso_body_ids, strict=True)
        }


def build_mamujoco_env(
    env_cfg: Any, rollout_length: int, backend: Backend = "auto"
) -> MaMuJoCoDualCostEnv:
    """Factory used by `safelie.envs.factory.build_env`."""
    return MaMuJoCoDualCostEnv(
        env_name=env_cfg.name,
        n_agents=env_cfg.n_agents,
        budget=env_cfg.budget,
        rollout_length=rollout_length,
        agent_obsk=getattr(env_cfg, "agent_obsk", 1),
        cost_mode=getattr(env_cfg, "cost_mode", "auto"),
        velocity_threshold=getattr(env_cfg, "velocity_threshold", None),
        backend=backend,
    )
