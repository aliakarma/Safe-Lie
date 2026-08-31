"""The learner's training loop: env -> rollout -> sources -> attack ->
aggregator -> dual update -> policy update -> log.

Report reference: PROJECT_REPORT.md §7.3 (training loop sketch), Phase 7
exit criterion ("One complete cell reproduces end-to-end from a config
hash, with the artifact directory containing everything needed to rerun
it"), smoke test S15.

One "round" here is one on-policy PPO iteration over a full
`rollout_length`-step episode (the environment is reset at the start of
every round — see `safelie.envs.synthetic`'s docstring for why this keeps
the discounted-return bookkeeping simple and exact for this toy problem).
This is a `[DECISION]`: real Safe MAMuJoCo training does not reset every
PPO iteration, but nothing in the reference pipeline below depends on
that choice — swapping in a real environment (`safelie.envs.mamujoco`)
with genuine multi-round episodes only changes how `AgentRollout` spans
episode boundaries, not any of the attack/defense/dual-update logic.

**Isolation boundary (S10).** This module is the learner: it constructs
the environment, rolls out trajectories, aggregates sources, applies the
dual update, and updates policies. It never imports
`safelie.eval.oracle` and never calls `env.oracle_handle()` or
`env._oracle_handle_privileged()` — true cost plays no role anywhere in
this file, by construction, not merely by discipline. The withheld-oracle
evaluation (`safelie.eval.harness.evaluate_true_cost`) is a structurally
separate rollout, run by the top-level orchestrator
(`safelie.experiment`), never by `ExperimentRun` itself.
`tests/isolation/test_oracle_isolation.py` greps this package to enforce
exactly that.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import numpy as np
import torch

from safelie.algos.networks import AgentBundle
from safelie.attacks import apply_attack
from safelie.attacks.ledger import AttackLedger
from safelie.consensus.topologies import build_topology
from safelie.defenses import aggregate
from safelie.envs.dual_cost import AgentID
from safelie.envs.factory import build_env
from safelie.eval.margin import calibrate_epsilon_offline, compute_guarantee_in_force
from safelie.sources.estimators import DiversifiedReplica
from safelie.sources.registry import SourceRegistry
from safelie.training.buffer import AgentRollout
from safelie.training.dual import dual_update
from safelie.training.ppo import ppo_lagrangian_update
from safelie.utils.config import ExperimentConfig, SourceSpec
from safelie.utils.logging import JsonlLogger
from safelie.utils.seeding import seed_everything


def select_corrupted_sources(specs: list[SourceSpec], f: int) -> set[str]:
    """Deterministic choice of which `f` sources are attacker-controlled:
    the first `f` non-`own_critic` sources, in config order. Kept out of
    the attack module itself, per Phase 5's "one clean hook" principle --
    the attack module transforms residuals, it does not decide who is
    compromised."""
    candidates = [s.source_id for s in specs if s.source_type != "own_critic"] or [
        s.source_id for s in specs
    ]
    return set(candidates[:f])


class ExperimentRun:
    """Owns all mutable state for one (config, seed) experiment."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.seed_bundle = seed_everything(cfg.seed)

        self.env = build_env(cfg.env, rollout_length=cfg.rollout_length)
        self.agents: dict[AgentID, AgentBundle] = {
            aid: AgentBundle(cfg.env.obs_dim, cfg.env.action_dim, cfg.ppo.hidden_dim, cfg.ppo.lr)
            for aid in self.env.agent_ids
        }
        self.lam = np.zeros(cfg.env.n_agents)
        self.W = build_topology(
            cfg.topology.name, cfg.topology.n_agents, p=cfg.topology.p, graph_seed=cfg.topology.graph_seed
        )
        self.source_registry = SourceRegistry(cfg.sources)
        self.corrupted_ids = select_corrupted_sources(cfg.sources.sources, cfg.attack.f)
        self.ledger = AttackLedger()

        self.env_rng = self.seed_bundle.rng("env")
        self.attack_rng = self.seed_bundle.rng("attack")

        self.replicas: dict[str, DiversifiedReplica] = {}
        for i, spec in enumerate(cfg.sources.sources):
            if spec.source_type in ("ensemble_replica", "monitor"):
                self.replicas[spec.source_id] = DiversifiedReplica(
                    obs_dim=cfg.env.obs_dim, seed=cfg.seed * 1000 + i
                )

        self.output_dir = Path(cfg.output_dir) / cfg.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.round_logger = JsonlLogger(self.output_dir / "rounds.jsonl")
        self.clean_run_disagreements: list[float] = []
        self.round_index = 0

    def _collect_source_value(self, spec: SourceSpec, owner_id: AgentID, owner_finalized: dict) -> float:
        if spec.source_type == "own_critic":
            return owner_finalized["cost_return_estimate"]
        if spec.source_type == "peer_critic":
            peer_id = spec.source_id.replace("peer_critic_", "agent_")
            peer_agent = self.agents.get(peer_id, next(iter(self.agents.values())))
            obs0 = torch.as_tensor(owner_finalized["obs"][0], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                return float(peer_agent.cost_value(obs0).item())
        # ensemble_replica / monitor
        replica = self.replicas[spec.source_id]
        return replica.refit_and_predict(
            owner_finalized["obs"], owner_finalized["ret_c"], owner_finalized["obs"][0]
        )

    def run_round(self) -> dict:
        cfg = self.cfg
        round_seed = int(self.env_rng.integers(0, 2**31 - 1))
        step = self.env.reset(seed=round_seed)
        rollouts = {aid: AgentRollout() for aid in self.env.agent_ids}

        for _t in range(cfg.rollout_length):
            actions_taken: dict[AgentID, np.ndarray] = {}
            raw_actions: dict[AgentID, np.ndarray] = {}
            logprobs: dict[AgentID, float] = {}
            values: dict[AgentID, float] = {}
            cost_values: dict[AgentID, float] = {}
            for aid in self.env.agent_ids:
                obs_t = torch.as_tensor(step.obs[aid], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    dist = self.agents[aid].policy.distribution(obs_t)
                    raw_action = dist.sample()
                    logprob = dist.log_prob(raw_action).sum(-1)
                    action = torch.tanh(raw_action)
                    value = self.agents[aid].value(obs_t)
                    cost_value = self.agents[aid].cost_value(obs_t)
                raw_actions[aid] = raw_action.squeeze(0).numpy()
                actions_taken[aid] = action.squeeze(0).numpy()
                logprobs[aid] = float(logprob.item())
                values[aid] = float(value.item())
                cost_values[aid] = float(cost_value.item())

            prev_obs = step.obs
            step = self.env.step(actions_taken)

            for aid in self.env.agent_ids:
                done = bool(step.terminated[aid] or step.truncated[aid])
                rollouts[aid].add(
                    prev_obs[aid], raw_actions[aid], logprobs[aid], step.reward,
                    step.reported_cost[aid], values[aid], cost_values[aid], done,
                )

        finalized = {}
        for aid in self.env.agent_ids:
            obs_last = torch.as_tensor(step.obs[aid], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                last_value = float(self.agents[aid].value(obs_last).item())
                last_cost_value = float(self.agents[aid].cost_value(obs_last).item())
            finalized[aid] = rollouts[aid].finalize(cfg.ppo.gamma, cfg.ppo.gae_lambda, last_value, last_cost_value)

        round_record: dict[str, Any] = {"round_k": self.round_index, "constraints": {}}
        new_lam = np.zeros_like(self.lam)
        for i, aid in enumerate(self.env.agent_ids):
            reports = self.source_registry.collect(
                aid, self.round_index,
                functools.partial(self._collect_source_value, owner_id=aid, owner_finalized=finalized[aid]),
            )
            residuals = {r.source_id: r.value - cfg.env.budget for r in reports}

            corrupted_here = self.corrupted_ids & set(residuals)
            attacked = apply_attack(
                cfg.attack, residuals, corrupted_here, self.round_index, cfg.env.budget,
                rng=self.attack_rng, ledger=self.ledger,
            )
            values_arr = np.array([attacked[r.source_id] + cfg.env.budget for r in reports])
            agg = aggregate(cfg.defense.name, values_arr, cfg.defense.f, beta=cfg.defense.beta,
                             sigma_min=cfg.defense.sigma_min, min_retained=cfg.defense.min_retained) \
                if cfg.defense.name == "rce" else aggregate(cfg.defense.name, values_arr, cfg.defense.f)

            point_estimate = agg.pessimistic_estimate if hasattr(agg, "pessimistic_estimate") else agg.point_estimate
            residual_i = point_estimate - cfg.env.budget

            if cfg.attack.name == "none":
                self.clean_run_disagreements.append(agg.spread)
            epsilon_offline = calibrate_epsilon_offline(self.clean_run_disagreements) if self.clean_run_disagreements else 0.0
            guarantee_in_force = compute_guarantee_in_force(
                getattr(agg, "applied_margin", 0.0), epsilon_offline
            ) if cfg.defense.name == "rce" else None

            new_lam[i] = residual_i  # temporarily store per-agent residual; mixed below

            round_record["constraints"][aid] = {
                "reports": [{"source_id": r.source_id, "value": r.value} for r in reports],
                "corrupted_source_ids": sorted(corrupted_here),
                "aggregate": {
                    "point_estimate": agg.point_estimate,
                    "spread": agg.spread,
                    "retained_n": agg.retained_n,
                    "degenerate": agg.degenerate,
                },
                "guarantee_in_force": guarantee_in_force,
                "reported_cost_return": finalized[aid]["cost_return_estimate"],
                "task_return": float(finalized[aid]["ret_r"][0]) if len(finalized[aid]["ret_r"]) else 0.0,
            }

        residual_vec = new_lam
        self.lam = dual_update(self.lam, self.W, cfg.dual.eta_lambda, residual_vec, cfg.dual.lambda_max)

        for i, aid in enumerate(self.env.agent_ids):
            ppo_lagrangian_update(self.agents[aid], finalized[aid], float(self.lam[i]), cfg.ppo)
            round_record["constraints"][aid]["lambda_after"] = float(self.lam[i])

        self.round_logger.write(round_record)
        self.round_index += 1
        return round_record

    def run(self) -> Path:
        num_rounds = max(1, self.cfg.total_steps // self.cfg.rollout_length)
        for _ in range(num_rounds):
            self.run_round()
        self.round_logger.close()
        return self.output_dir

    def checkpoint(self, path: Path, extra: dict | None = None) -> None:
        """Smoke test S14: a restored run must continue bitwise-identically
        to an uninterrupted one. That requires every source of randomness
        touched between rounds to be captured, not just the model weights:
        the two child-seed generators this class owns directly (env,
        attack), the diversified replicas' own bootstrap-resampling RNGs,
        and — easy to miss — the *global* torch and numpy RNGs, which
        `safelie.training.ppo` (np.random.permutation) and
        `GaussianPolicy.act` (torch's default generator) both read from
        implicitly rather than through an explicit Generator argument.
        """
        state = {
            "agents": {aid: b.state_dict() for aid, b in self.agents.items()},
            "replicas": {sid: r.state_dict() for sid, r in self.replicas.items()},
            "lam": self.lam,
            "round_index": self.round_index,
            "clean_run_disagreements": list(self.clean_run_disagreements),
            "rng_state": {
                "env": self.env_rng.bit_generator.state,
                "attack": self.attack_rng.bit_generator.state,
                "replicas": {sid: r.rng.bit_generator.state for sid, r in self.replicas.items()},
                "torch_global": torch.get_rng_state(),
                "numpy_global": np.random.get_state(),
            },
            "extra": extra or {},
        }
        torch.save(state, path)

    def restore(self, path: Path) -> dict[str, Any]:
        state = torch.load(path, weights_only=False)
        for aid, sd in state["agents"].items():
            self.agents[aid].load_state_dict(sd)
        for sid, sd in state["replicas"].items():
            self.replicas[sid].load_state_dict(sd)
        self.lam = state["lam"]
        self.round_index = state["round_index"]
        self.clean_run_disagreements = list(state["clean_run_disagreements"])
        self.env_rng.bit_generator.state = state["rng_state"]["env"]
        self.attack_rng.bit_generator.state = state["rng_state"]["attack"]
        for sid, rng_state in state["rng_state"]["replicas"].items():
            self.replicas[sid].rng.bit_generator.state = rng_state
        torch.set_rng_state(state["rng_state"]["torch_global"])
        np.random.set_state(state["rng_state"]["numpy_global"])
        return state.get("extra", {})


def run_experiment(cfg: ExperimentConfig) -> Path:
    run = ExperimentRun(cfg)
    return run.run()
