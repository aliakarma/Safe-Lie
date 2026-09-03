"""Parallel trajectory-batch constraint sources: M independent rollout
batches under one pinned policy, M scalars, no neural estimator anywhere.

Report reference: `docs/g9_gates.md` (the pre-declared gates this module
is built to satisfy); `docs/g8_gates.md` Step 8 (the derivation that
forces parallel rather than sequential collection); main_iclr.tex Eq. 1.

## Why this module exists

Through G2 every constraint source was a *neural* estimator: the owner's
own critic, a peer's constraint-report head, or a small regression
replica refit each round (`safelie.sources.estimators`). G1 measured
corr(peer_critic, true) ~= 0. G2 replaced the peer's PPO cost critic with
a dedicated Monte-Carlo-trained head and the clean false-safe rate barely
moved (0.3810 -> 0.3819). G3-G7 established why: a regression head
queried at one observation estimates a *state-conditional* quantity, and
`J_C^i(theta)` is a policy-level expectation over the whole trajectory
distribution. Those are different objects, and no amount of repairing the
head makes the first into the second.

G7 established the estimator that *is* the right object:

    Jhat_{C,m}^i = (1/R_m) sum_{r in B_m} G_r^i,
    G_r^i        = sum_t gamma^t C_{r,t}^i

over a batch `B_m` of `R_m` trajectories drawn under `theta`. Disjoint
batches are uncorrelated by construction, not merely empirically.

G8 then established the *schedule* constraint, which is what this module's
shape is dictated by. Under sequential collection -- one rollout per
training round, so an `R_m = 30` batch spans 30 rounds -- the policy drifts
between source 1 and source 3 by 5-8 standard errors, and the three
sources end up estimating three different numbers. Independence is not
the binding constraint; *target drift* is. G8's Step-8 derivation shows
`R_drift` grows as `R_m^(3/2)` under a sequential schedule, so buying
precision by enlarging the batch makes drift worse faster than it makes
sampling error better. Collecting the `M` batches **in parallel under a
pinned theta_k** sets the drift term to exactly zero and is the only
schedule at which `R_m = 30` is admissible at all.

Hence this module: `M` replicas, each with its own environment instance
and its own RNG stream, all stepping the SAME frozen `theta_k`, each
returning one scalar per owner.

## What is deliberately absent

No cost critic. No GAE. No regression head. No peer critic. No monitor.
No identity-conditioned predictor. The only computation applied to a
trajectory is `discounted_window_return` on the learner-visible
`reported_cost` stream -- the same validated functional G3-G8 used and
the same one `safelie.eval.oracle.OracleEvaluator` computes.

## Isolation

This module is learner-side. It reads `DualCostStep.obs`,
`.reported_cost`, `.terminated`, `.truncated` and nothing else; it never
constructs an oracle handle, never imports `safelie.eval.oracle`, and
never reads privileged state. `tests/isolation/test_oracle_isolation.py`
enforces that at AST level over this whole package.

## Determinism, and why worker count is not a scientific parameter

Every trajectory's `(env_seed, torch_seed)` pair is drawn **in the main
process**, from `M` PCG64 streams spawned once from a single
`numpy.random.SeedSequence` -- one stream per replica, independent by
construction rather than by choice of offsets. A worker is then a pure
function of `(policy payload, env_seed, torch_seed)`: it reseeds the
environment via `env.reset(seed=...)` and the torch global generator via
`torch.manual_seed(...)` at the start of each trajectory, so nothing
carries over between trajectories and nothing depends on which process
ran which trajectory or in what order.

The consequence, which `tests/unit/test_source_batch.py` asserts rather
than assumes: `workers=1` and `workers=3` produce bitwise-identical
source values. `SourceCollectionConfig.workers` is a compute knob, and a
worker count that changed a reported number would be a defect.

## Policy pinning, verified rather than asserted

`policy_checksum` hashes every agent's policy parameters together with
its observation-normalization statistics -- both, because the trajectory
distribution depends on both (`AgentBundle.normalize_obs_tensor` is
applied before every forward pass, so a changed `obs_rms` is a changed
policy in every sense that matters here). The main process computes the
checksum of `theta_k`, each worker recomputes it from the parameters it
actually loaded, and the collector asserts equality for every replica
every round; it also re-checksums `theta_k` after collection returns, so
"collection mutated nothing" is checked rather than hoped for.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from safelie.algos.networks import AgentBundle
from safelie.envs.factory import build_env
from safelie.training.constraint_return import discounted_window_return
from safelie.utils.config import ExperimentConfig

# Drawn once, from `secrets`, and pinned here so every G9 run uses the
# same source-seed family and that family is disjoint from every seed
# used anywhere in G0-G8 (training: 0 and derived offsets 1000/4000/6000/
# 21000-24000/90000s; G3-G6: 777; G7: 8801/8802/8803; G8: 970000+). A
# 128-bit entropy value cannot collide with any of those by accident.
DEFAULT_SOURCE_ENTROPY = 286_314_957_402_113_664_887_331_205_920_951_063_913


def _tensor_bytes(x: torch.Tensor | np.ndarray) -> bytes:
    """Full-precision, byte-exact serialization for checksumming. float64
    everywhere so a float32 parameter and its float64 promotion hash the
    same way regardless of which side computed it, and C-contiguous so
    the byte order is the memory layout and not an accident of striding."""
    a = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    return np.ascontiguousarray(a, dtype=np.float64).tobytes()


def policy_checksum(agents: dict[str, AgentBundle], agent_ids: list[str]) -> str:
    """SHA-256 over the quantities that determine the trajectory
    distribution: every agent's policy parameters and its observation
    normalization statistics, in a fixed (sorted) key order.

    The critics are deliberately **excluded**. They are not consulted
    during source collection and they do not enter the action
    distribution, so including them would make the checksum change for
    reasons that have nothing to do with whether the sampling policy was
    pinned -- and a pinning check that fires on irrelevant changes is a
    pinning check nobody trusts.
    """
    h = hashlib.sha256()
    for aid in agent_ids:
        b = agents[aid]
        h.update(aid.encode("utf-8"))
        for key in sorted(b.policy.state_dict()):
            h.update(key.encode("utf-8"))
            h.update(_tensor_bytes(b.policy.state_dict()[key]))
        h.update(b"obs_rms")
        h.update(_tensor_bytes(b.obs_rms.mean))
        h.update(_tensor_bytes(b.obs_rms.var))
        h.update(_tensor_bytes(np.array([b.obs_rms.count])))
    return h.hexdigest()


def policy_payload(agents: dict[str, AgentBundle], agent_ids: list[str]) -> dict[str, Any]:
    """The minimal picklable description of `theta_k` a worker needs.

    Policy parameters plus `obs_rms`; no critics, no optimizer state. A
    worker that received optimizer state could in principle step it, so
    not sending it is a structural guarantee rather than a size
    optimization."""
    return {
        aid: {
            "policy": {k: v.detach().cpu().clone() for k, v in agents[aid].policy.state_dict().items()},
            "obs_rms_mean": np.array(agents[aid].obs_rms.mean, dtype=np.float64),
            "obs_rms_var": np.array(agents[aid].obs_rms.var, dtype=np.float64),
            "obs_rms_count": float(agents[aid].obs_rms.count),
        }
        for aid in agent_ids
    }


def _load_payload(agents: dict[str, AgentBundle], payload: dict[str, Any]) -> None:
    for aid, blob in payload.items():
        agents[aid].policy.load_state_dict(blob["policy"])
        agents[aid].obs_rms.mean = np.array(blob["obs_rms_mean"], dtype=np.float64)
        agents[aid].obs_rms.var = np.array(blob["obs_rms_var"], dtype=np.float64)
        agents[aid].obs_rms.count = float(blob["obs_rms_count"])


def collect_one_trajectory(
    env,
    agents: dict[str, AgentBundle],
    agent_ids: list[str],
    rollout_length: int,
    gamma: float,
    env_seed: int,
    torch_seed: int,
) -> dict[str, float]:
    """One frozen-policy trajectory; returns `G_r^i` per owner.

    The action-selection block is byte-identical to
    `scripts/g7_collect_estimand_dataset.py::collect_one_round` and to
    `safelie.training.loop.ExperimentRun.run_round`'s own rollout: the
    observation is normalized, the diagonal-Gaussian distribution is
    built, `.sample()` is drawn, and the sample is tanh-squashed. Keeping
    the identical path (rather than an arithmetically equivalent faster
    one) is deliberate -- G7 and G8 validated *this* estimator, and a
    rewritten forward pass would make G9 a test of new code rather than
    of the architecture G8 recommended.

    `torch.manual_seed` is called once per trajectory rather than an
    explicit `Generator` being threaded through, because `Normal.sample`
    reads the global generator and the goal is to keep `.sample()` in the
    path. Each trajectory runs to completion inside one process without
    interleaving, so the whole trajectory is a deterministic function of
    `torch_seed`, and nothing carries over to the next one.
    """
    torch.manual_seed(int(torch_seed))
    step = env.reset(seed=int(env_seed))
    costs = {aid: np.empty(rollout_length, dtype=np.float64) for aid in agent_ids}

    for t in range(rollout_length):
        actions: dict[str, np.ndarray] = {}
        for aid in agent_ids:
            obs_t = torch.as_tensor(step.obs[aid], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                obs_n = agents[aid].normalize_obs_tensor(obs_t)
                dist = agents[aid].policy.distribution(obs_n)
                raw_action = dist.sample()
                action = torch.tanh(raw_action)
            actions[aid] = action.squeeze(0).numpy()
        step = env.step(actions)
        for aid in agent_ids:
            costs[aid][t] = float(step.reported_cost[aid])

    return {aid: float(discounted_window_return(costs[aid], gamma)) for aid in agent_ids}


# --------------------------------------------------------------------------
# Worker process state. Built once per worker by `_init_worker` and reused
# for the whole run: constructing a MuJoCo environment costs ~2 s the first
# time, which at 22,500 trajectories would dominate everything else.
# --------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _init_worker(cfg_json: str) -> None:
    torch.set_num_threads(1)
    cfg = ExperimentConfig.model_validate_json(cfg_json)
    env = build_env(cfg.env, rollout_length=cfg.rollout_length)
    agent_ids = list(env.agent_ids)
    agents = {
        aid: AgentBundle(int(env.obs_dim), int(env.action_dim), cfg.ppo.hidden_dim, cfg.ppo.lr)
        for aid in agent_ids
    }
    _WORKER.update(cfg=cfg, env=env, agents=agents, agent_ids=agent_ids)


def _run_chunk(job: tuple) -> tuple[str, list]:
    """Run one contiguous chunk of trajectories under one pinned policy.

    Returns the checksum recomputed **from the loaded network objects**
    (not from the payload dict), so what is verified is what the worker
    actually rolled out with.
    """
    payload, items = job
    cfg = _WORKER["cfg"]
    env = _WORKER["env"]
    agents = _WORKER["agents"]
    agent_ids = _WORKER["agent_ids"]

    _load_payload(agents, payload)
    checksum = policy_checksum(agents, agent_ids)

    out = []
    for replica_id, traj_idx, env_seed, torch_seed in items:
        g = collect_one_trajectory(
            env, agents, agent_ids, cfg.rollout_length, cfg.ppo.gamma, env_seed, torch_seed
        )
        out.append((replica_id, traj_idx, g))
    return checksum, out


@dataclass
class BatchSourceResult:
    """One round's source collection, in full, ready to be logged."""

    round_k: int
    policy_checksum: str
    worker_checksums: dict[str, str]
    # replica_id -> owner -> Jhat_{C,m}^i
    source_means: dict[str, dict[str, float]]
    # replica_id -> owner -> the R_m raw G_r^i values
    per_trajectory: dict[str, dict[str, list[float]]]
    # replica_id -> [(env_seed, torch_seed), ...]
    seeds: dict[str, list[tuple[int, int]]]
    n_trajectories: int
    env_steps: int
    wall_clock_s: float
    # Populated only on validation rounds.
    reference_mean: dict[str, float] | None = None
    reference_per_trajectory: dict[str, list[float]] | None = None
    reference_seeds: list[tuple[int, int]] = field(default_factory=list)
    reference_n: int = 0
    reference_wall_clock_s: float = 0.0

    def aggregate(self, agent_ids: list[str]) -> dict[str, float]:
        return {
            aid: float(np.mean([self.source_means[m][aid] for m in sorted(self.source_means)]))
            for aid in agent_ids
        }


class ParallelBatchSourceCollector:
    """Owns the worker pool, the `M` replica RNG streams, and the
    seed-collision bookkeeping for one experiment run.

    One instance per `ExperimentRun`. The pool is created lazily on first
    use (so constructing an `ExperimentRun` in a test does not spawn
    processes) and lives for the rest of the run.
    """

    def __init__(self, cfg: ExperimentConfig, agent_ids: list[str]):
        sc = cfg.source_collection
        if sc is None:
            raise ValueError("ParallelBatchSourceCollector requires cfg.source_collection")
        self.cfg = cfg
        self.sc = sc
        self.agent_ids = list(agent_ids)
        self.replica_ids = [s.source_id for s in cfg.sources.sources]
        self.M = len(self.replica_ids)
        if self.M != sc.M:
            raise ValueError(
                f"cfg.sources declares {self.M} sources but source_collection.M is {sc.M}; "
                "the number of trajectory-batch replicas and the number of configured "
                "sources are the same quantity and must not be allowed to disagree."
            )

        # G9g-i: one SeedSequence, M+1 spawned children -- M replica
        # streams plus one reference stream used only on validation
        # rounds. Spawning (rather than seeding from M hand-chosen
        # integers) is what makes the streams independent by construction.
        self._root_ss = np.random.SeedSequence(sc.seed_entropy)
        children = self._root_ss.spawn(self.M + 1)
        self._replica_rngs = [np.random.default_rng(ss) for ss in children[: self.M]]
        self._reference_rng = np.random.default_rng(children[self.M])
        self.spawn_keys = [list(map(int, ss.entropy if isinstance(ss.entropy, (list, tuple)) else [ss.entropy]))
                           + list(map(int, ss.spawn_key)) for ss in children]

        # G9g-ii: every source seed ever issued, so duplicates are caught
        # rather than assumed away.
        self._issued_env_seeds: set[int] = set()
        self._issued_torch_seeds: set[int] = set()
        self.seed_collisions: list[dict] = []

        self._pool: mp.pool.Pool | None = None
        self._cfg_json = cfg.model_dump_json()

    # -- pool lifecycle -----------------------------------------------

    def _ensure_pool(self) -> mp.pool.Pool:
        if self._pool is None:
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(
                processes=self.sc.workers,
                initializer=_init_worker,
                initargs=(self._cfg_json,),
            )
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    # -- seeds ---------------------------------------------------------

    def _draw_seeds(self, rng: np.random.Generator, n: int) -> list[tuple[int, int]]:
        """`n` (env_seed, torch_seed) pairs from one replica's own stream.

        Both are drawn from the same PCG64 stream rather than from two
        streams, because the pair is consumed as a unit and a single
        well-mixed stream already gives independent draws; using two
        streams would add bookkeeping without adding independence.
        """
        raw = rng.integers(0, 2**31 - 1, size=(n, 2), dtype=np.int64)
        pairs = [(int(a), int(b)) for a, b in raw]
        for env_seed, torch_seed in pairs:
            if env_seed in self._issued_env_seeds:
                self.seed_collisions.append({"kind": "env_seed", "value": env_seed})
            if torch_seed in self._issued_torch_seeds:
                self.seed_collisions.append({"kind": "torch_seed", "value": torch_seed})
            self._issued_env_seeds.add(env_seed)
            self._issued_torch_seeds.add(torch_seed)
        return pairs

    # -- collection ----------------------------------------------------

    def _run_jobs(self, payload: dict, items: list[tuple]) -> tuple[list[str], list[tuple]]:
        """Farm `items` out to the pool in contiguous chunks.

        Chunking is a scheduling decision only: every item already carries
        its own seeds, so the partition affects wall clock and nothing
        else. With `workers == 1` the pool is bypassed entirely and the
        chunk runs in-process, which is what makes the worker-count
        invariance test cheap enough to run in CI.
        """
        if self.sc.workers <= 1:
            if not _WORKER:
                _init_worker(self._cfg_json)
            checksum, out = _run_chunk((payload, items))
            return [checksum], out

        pool = self._ensure_pool()
        # More chunks than workers so `Pool.map` load-balances: with
        # exactly one chunk per worker the round ends when the SLOWEST
        # worker ends, and a 10-trajectory chunk makes that straggler cost
        # up to ten trajectories. Finer chunks cost one extra policy
        # payload each (~210 KB, pickled once per chunk) and cut the
        # straggler to roughly one trajectory. Purely a scheduling choice:
        # every trajectory already carries its own seeds, so the partition
        # cannot change a value -- which is what the worker-count
        # invariance test in tests/unit/test_source_batch.py pins down.
        n_chunks = max(1, min(self.sc.workers * self.sc.chunks_per_worker, len(items)))
        bounds = np.linspace(0, len(items), n_chunks + 1).astype(int)
        jobs = [(payload, items[bounds[c] : bounds[c + 1]]) for c in range(n_chunks) if bounds[c + 1] > bounds[c]]
        results = pool.map(_run_chunk, jobs)
        checksums = [c for c, _ in results]
        out: list[tuple] = []
        for _, chunk_out in results:
            out.extend(chunk_out)
        return checksums, out

    def collect(
        self,
        agents: dict[str, AgentBundle],
        round_k: int,
        collect_reference: bool = False,
    ) -> BatchSourceResult:
        """The whole of step 3 (and 3b) of `docs/g9_gates.md`'s algorithm.

        `agents` is the live `ExperimentRun.agents` dict at `theta_k`.
        Nothing here mutates it: `policy_payload` clones every tensor it
        reads, and the rollouts happen against separate `AgentBundle`
        objects in separate processes.
        """
        t0 = time.time()
        expected = policy_checksum(agents, self.agent_ids)
        payload = policy_payload(agents, self.agent_ids)

        items: list[tuple] = []
        seeds: dict[str, list[tuple[int, int]]] = {}
        for m, rid in enumerate(self.replica_ids):
            pairs = self._draw_seeds(self._replica_rngs[m], self.sc.R_m)
            seeds[rid] = pairs
            for r, (env_seed, torch_seed) in enumerate(pairs):
                items.append((rid, r, env_seed, torch_seed))

        checksums, out = self._run_jobs(payload, items)

        per_traj: dict[str, dict[str, list[float]]] = {
            rid: {aid: [float("nan")] * self.sc.R_m for aid in self.agent_ids} for rid in self.replica_ids
        }
        for rid, r, g in out:
            for aid in self.agent_ids:
                per_traj[rid][aid][r] = g[aid]

        source_means = {
            rid: {aid: float(np.mean(per_traj[rid][aid])) for aid in self.agent_ids}
            for rid in self.replica_ids
        }

        # G9h-ii / G9h-iii: the workers ran what we think they ran, and
        # running them changed nothing on this side.
        worker_checksums = {f"chunk_{i}": c for i, c in enumerate(checksums)}
        bad = {k: v for k, v in worker_checksums.items() if v != expected}
        if bad:
            raise RuntimeError(
                f"G9h-ii FAILED at round {round_k}: source replicas did not run the pinned "
                f"policy. expected={expected} got={bad}. The dual update at this round would "
                "have consumed estimates of a policy that is not theta_k."
            )
        after = policy_checksum(agents, self.agent_ids)
        if after != expected:
            raise RuntimeError(
                f"G9h-iii FAILED at round {round_k}: theta_k changed during source collection "
                f"({expected} -> {after}). Collection must be read-only."
            )

        wall = time.time() - t0
        n_traj = self.M * self.sc.R_m
        result = BatchSourceResult(
            round_k=round_k,
            policy_checksum=expected,
            worker_checksums=worker_checksums,
            source_means=source_means,
            per_trajectory=per_traj,
            seeds=seeds,
            n_trajectories=n_traj,
            env_steps=n_traj * self.cfg.rollout_length,
            wall_clock_s=wall,
        )

        if collect_reference:
            t1 = time.time()
            ref_pairs = self._draw_seeds(self._reference_rng, self.sc.R_ref)
            ref_items = [("__reference__", r, e, t) for r, (e, t) in enumerate(ref_pairs)]
            ref_checksums, ref_out = self._run_jobs(payload, ref_items)
            bad_ref = [c for c in ref_checksums if c != expected]
            if bad_ref:
                raise RuntimeError(
                    f"G9h-ii FAILED for the validation reference at round {round_k}: {bad_ref}"
                )
            ref_traj = {aid: [float("nan")] * self.sc.R_ref for aid in self.agent_ids}
            for _rid, r, g in ref_out:
                for aid in self.agent_ids:
                    ref_traj[aid][r] = g[aid]
            result.reference_per_trajectory = ref_traj
            result.reference_mean = {aid: float(np.mean(ref_traj[aid])) for aid in self.agent_ids}
            result.reference_seeds = ref_pairs
            result.reference_n = self.sc.R_ref
            result.reference_wall_clock_s = time.time() - t1

        return result

    # -- checkpointing --------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "replica_rngs": [r.bit_generator.state for r in self._replica_rngs],
            "reference_rng": self._reference_rng.bit_generator.state,
            "issued_env_seeds": sorted(self._issued_env_seeds),
            "issued_torch_seeds": sorted(self._issued_torch_seeds),
            "seed_collisions": list(self.seed_collisions),
        }

    def load_state_dict(self, state: dict) -> None:
        for rng, s in zip(self._replica_rngs, state["replica_rngs"], strict=True):
            rng.bit_generator.state = s
        self._reference_rng.bit_generator.state = state["reference_rng"]
        self._issued_env_seeds = set(state["issued_env_seeds"])
        self._issued_torch_seeds = set(state["issued_torch_seeds"])
        self.seed_collisions = list(state["seed_collisions"])

    def seed_audit(self) -> dict:
        """G9g-ii/iii, as a checkable object rather than a claim."""
        return {
            "n_env_seeds_issued": len(self._issued_env_seeds),
            "n_torch_seeds_issued": len(self._issued_torch_seeds),
            "duplicate_seed_events": len(self.seed_collisions),
            "collisions": self.seed_collisions[:50],
            "spawn_keys": json.loads(json.dumps(self.spawn_keys)),
        }
