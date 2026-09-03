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
from safelie.eval.margin import compute_guarantee_in_force
from safelie.sources.estimators import DiversifiedReplica
from safelie.sources.registry import SourceRegistry
from safelie.training.buffer import AgentRollout
from safelie.training.dual import dual_update
from safelie.training.ppo import ppo_lagrangian_update
from safelie.training.source_batch import (
    BatchSourceResult,
    ParallelBatchSourceCollector,
    policy_checksum,
)
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

    def __init__(self, cfg: ExperimentConfig, _skip_calibration: bool = False):
        self.cfg = cfg

        # P0 #8 fix: `guarantee_in_force` must be calibrated from a
        # dedicated clean (attack-disabled) reference distribution that
        # exists independently of this run, never from this run's own
        # online history (which is empty for a run's entire length
        # whenever attack.name != "none" -- see safelie.eval.calibration's
        # docstring for the vacuous-guarantee bug this replaces). Run
        # BEFORE `seed_everything(cfg.seed)` below so the calibration
        # phase's own RNG usage cannot perturb this run's determinism:
        # `seed_everything` unconditionally resets global numpy/torch
        # state afterward.
        self.calibration = None
        if cfg.defense.name == "rce" and not _skip_calibration:
            from safelie.eval.calibration import run_clean_calibration

            self.calibration = run_clean_calibration(cfg)

        self.seed_bundle = seed_everything(cfg.seed)

        self.env = build_env(cfg.env, rollout_length=cfg.rollout_length)
        # Network shapes come from the *constructed* environment, never
        # from the config. A real MuJoCo factorization determines its own
        # dimensions (ManySegmentAnt 6x1: 63-dim observations, 4-dim
        # actions), while `EnvConfig.obs_dim`/`action_dim` default to 8/2
        # and the pilot configs never set them -- reading them here built
        # 8-dim policies and fed them 63-dim observations.
        self.obs_dim = int(self.env.obs_dim)
        self.action_dim = int(self.env.action_dim)
        self.agents: dict[AgentID, AgentBundle] = {
            aid: AgentBundle(
                self.obs_dim, self.action_dim, cfg.ppo.hidden_dim, cfg.ppo.lr, critic_lr=cfg.ppo.critic_lr
            )
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
                    obs_dim=self.obs_dim, seed=cfg.seed * 1000 + i
                )

        # G2-peer (docs/g2_gates.md): one constraint-report head PER
        # PHYSICAL AGENT, refit once per round on that agent's own masked
        # MC cost-to-go targets (see `_cost_to_go_targets`), then queried
        # -- without refitting -- by every owner that has this agent as a
        # `peer_critic` source this round (`_collect_source_value`). This
        # is a separate object from both `self.agents[aid].cost_value_net`
        # (PPO's own cost critic, trained against `ret_c` for GAE/advantage
        # estimation, untouched by this repair) and from `self.replicas`
        # (which stays fit-per-owner-per-call, unchanged since G1).
        self.constraint_report_heads: dict[AgentID, DiversifiedReplica] = {
            aid: DiversifiedReplica(obs_dim=self.obs_dim, seed=cfg.seed * 4000 + i)
            for i, aid in enumerate(self.env.agent_ids)
        }

        # G9 (docs/g9_gates.md). When the parallel trajectory-batch source
        # architecture is selected, the M sources are no longer critics or
        # fitted heads read off this round's PPO rollout -- they are M
        # independent rollout BATCHES collected under a pinned theta_k by
        # `safelie.training.source_batch`. `self.replicas` is empty in that
        # mode (no spec has type ensemble_replica/monitor) and
        # `self.constraint_report_heads` is constructed but never refit and
        # never queried, which `_source_pipeline_audit` records every round.
        self.batch_sources: ParallelBatchSourceCollector | None = None
        if cfg.source_collection.mode == "parallel_trajectory_batch":
            self.batch_sources = ParallelBatchSourceCollector(cfg, list(self.env.agent_ids))

        self.output_dir = Path(cfg.output_dir) / cfg.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.round_logger = JsonlLogger(self.output_dir / "rounds.jsonl")
        self.source_logger: JsonlLogger | None = (
            JsonlLogger(self.output_dir / "source_seeds.jsonl") if self.batch_sources else None
        )
        self.reference_logger: JsonlLogger | None = (
            JsonlLogger(self.output_dir / "validation_reference.jsonl") if self.batch_sources else None
        )
        self.clean_run_disagreements: list[float] = []
        self.round_index = 0

        if self.calibration is not None:
            from safelie.eval.calibration import write_calibration_report

            write_calibration_report(self.calibration, self.output_dir)

    def _peer_agent_id(self, source_id: str, owner_id: AgentID) -> AgentID:
        """Map a `peer_critic_<k>` source to a peer **relative to the
        owner**: agent (owner_index + k) mod N.

        Bug fixed here (P0 #7): the previous mapping was
        `peer_critic_k -> literal agent_k`, the same physical agent for
        every owner. Under the pilot's own M=7 config (`peer_critic_1..4`
        on N=6 agents), that made `peer_critic_i`'s target literally equal
        the owner for every owner in {agent_1, agent_2, agent_3, agent_4}
        -- 4 of 6 agents (67%) received their own critic, on their own
        observation, relabeled as an independent "peer" source. Since `k`
        ranges over 1..N-1 and is never a multiple of N for any owner
        offset, `(owner_index + k) mod N == owner_index` is impossible,
        which is asserted below rather than merely hoped for.
        """
        offset_str = source_id.rsplit("_", 1)[-1]
        if not offset_str.isdigit():
            raise ValueError(
                f"peer_critic source_id {source_id!r} must end in an integer offset "
                f"(e.g. 'peer_critic_1'); got a non-numeric suffix {offset_str!r}."
            )
        offset = int(offset_str)
        agent_ids = self.env.agent_ids
        n = len(agent_ids)
        owner_idx = agent_ids.index(owner_id)
        peer_idx = (owner_idx + offset) % n
        peer_id = agent_ids[peer_idx]
        if peer_id == owner_id:
            raise ValueError(
                f"peer_critic source_id {source_id!r} (offset={offset}) resolved to "
                f"owner {owner_id!r} itself for N={n} agents -- a source config whose "
                f"offset is a multiple of N self-collides for every owner and must not "
                f"be used as a peer source (P0 #7). Fix the source spec's offset."
            )
        return peer_id

    def _cost_to_go_targets(self, owner_finalized: dict) -> tuple[np.ndarray, np.ndarray]:
        """The (obs, cost-to-go) regression pairs a replica/monitor source
        fits each round, under the configured constraint estimator.

        Under `mc_window` these are proper discounted Monte-Carlo
        cost-to-go targets, masked to the leading rows whose target is
        complete to within 1% of its own discounted mass -- rows near the
        window edge carry systematically shrunk targets (at gamma=0.99 the
        target at t = T-10 is missing 90% of its mass) and fitting on them
        would reintroduce the downward bias through the regression head.
        Under `gae_lambda` they are the pre-G1 `ret_c` targets, unmasked.
        """
        if self.cfg.constraint_estimator == "mc_window":
            n = int(owner_finalized["n_mc_targets"])
            return owner_finalized["obs"][:n], owner_finalized["mc_cost_to_go"][:n]
        return owner_finalized["obs"], owner_finalized["ret_c"]

    def _collect_source_value(
        self,
        spec: SourceSpec,
        owner_id: AgentID,
        owner_finalized: dict,
        batch: BatchSourceResult | None = None,
    ) -> float:
        """One source's return-scale estimate of the owner's J_C^i.

        The constraint estimator selected by
        `ExperimentConfig.constraint_estimator` changes WHAT the
        critic-free sources estimate, not the mechanism: every source
        still produces one return-scale scalar, those scalars are still
        the only thing that leaves this method, and
        `safelie.attacks.apply_attack` still corrupts them downstream in
        `run_round` before aggregation. Nothing here reads true cost.

          own_critic      -- under `mc_window`, the owner's own discounted
                             Monte-Carlo constraint return over this
                             round's rollout (the direct empirical
                             estimate of J_C^i); under `gae_lambda`, the
                             pre-G1 `ret_c[0]` bootstrap target.
          peer_critic     -- G2-peer (docs/g2_gates.md): a peer's
                             **constraint-report head**
                             (`self.constraint_report_heads[peer_id]`),
                             evaluated at the owner's initial observation.
                             The head is refit once per round, before this
                             method is ever called for that round, on the
                             PEER's own masked MC cost-to-go targets (see
                             `run_round`) -- never on `ret_c`, never on the
                             owner's data, and never on another source's
                             estimate. This is a change from G1, where this
                             branch queried the peer's PPO cost critic
                             (`AgentBundle.cost_value`, trained against
                             `ret_c` for GAE/advantage estimation) and
                             measured corr(prediction, true) ~= 0 as a
                             result (docs/g2_gates.md). PPO's own cost
                             critic is untouched: GAE, `ret_c`, and the
                             cost-value network's optimizer are not read by
                             this branch at all any more.
          replica/monitor -- a small independently-initialized head refit
                             each round on (obs, cost-to-go); the targets
                             follow the selected estimator, see
                             `_cost_to_go_targets`.
          trajectory_batch -- G9 (docs/g9_gates.md): replica `spec.source_id`'s
                             sample mean of `sum_t gamma^t C_t^i` over its
                             own `R_m` trajectories, drawn under the pinned
                             `theta_k` by `safelie.training.source_batch`
                             BEFORE this method is called. Nothing is
                             computed here -- the value was produced by a
                             separate rollout batch, not by any network --
                             so this branch is a lookup, and deliberately
                             so: any arithmetic here would be arithmetic
                             the estimator's definition does not contain.
                             `owner_finalized` is not read at all on this
                             path, which is what keeps the PPO rollout out
                             of the constraint estimate.
        """
        if spec.source_type == "trajectory_batch":
            if batch is None:
                raise RuntimeError(
                    "trajectory_batch source requested without a collected batch; "
                    "ExperimentRun.run_round must collect sources under the pinned "
                    "theta_k before assembling reports (docs/g9_gates.md step 3)."
                )
            return batch.source_means[spec.source_id][owner_id]
        if spec.source_type == "own_critic":
            if self.cfg.constraint_estimator == "mc_window":
                return float(owner_finalized["mc_cost_return"])
            return owner_finalized["cost_return_estimate"]
        if spec.source_type == "peer_critic":
            peer_id = self._peer_agent_id(spec.source_id, owner_id)
            return self.constraint_report_heads[peer_id].predict(owner_finalized["obs"][0])
        # ensemble_replica / monitor
        replica = self.replicas[spec.source_id]
        fit_obs, fit_targets = self._cost_to_go_targets(owner_finalized)
        return replica.refit_and_predict(fit_obs, fit_targets, owner_finalized["obs"][0])

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
                    # P0 #2: the policy is trained on normalized
                    # observations (safelie.training.ppo), so it must
                    # also act on them; `.value`/`.cost_value` already
                    # normalize internally (safelie.algos.networks) and
                    # denormalize their return-scale output.
                    obs_n = self.agents[aid].normalize_obs_tensor(obs_t)
                    dist = self.agents[aid].policy.distribution(obs_n)
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

            # A truncation (not a genuine termination) must bootstrap
            # from the value AT THE TRUE FINAL OBSERVATION of the episode
            # that just ended, not from 0 -- see
            # safelie.training.gae.compute_gae's docstring. Environments
            # that auto-reset internally on truncation (safelie.envs.
            # mamujoco) expose that observation via
            # `info["final_observation"]`; environments that don't
            # auto-reset (safelie.envs.synthetic) never need this, since
            # their own truncation always falls on the buffer's last
            # index, where compute_gae's `last_value`/`last_cost_value`
            # argument already covers it.
            final_obs = step.info.get("final_observation")
            for aid in self.env.agent_ids:
                trunc_v = trunc_cv = 0.0
                if final_obs is not None and bool(step.truncated[aid]) and not bool(step.terminated[aid]):
                    fobs_t = torch.as_tensor(final_obs[aid], dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        trunc_v = float(self.agents[aid].value(fobs_t).item())
                        trunc_cv = float(self.agents[aid].cost_value(fobs_t).item())
                rollouts[aid].add(
                    prev_obs[aid], raw_actions[aid], logprobs[aid], step.reward,
                    step.reported_cost[aid], values[aid], cost_values[aid],
                    terminated=bool(step.terminated[aid]), truncated=bool(step.truncated[aid]),
                    truncation_value_bootstrap=trunc_v, truncation_cost_value_bootstrap=trunc_cv,
                )

        finalized = {}
        for aid in self.env.agent_ids:
            obs_last = torch.as_tensor(step.obs[aid], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                last_value = float(self.agents[aid].value(obs_last).item())
                last_cost_value = float(self.agents[aid].cost_value(obs_last).item())
            finalized[aid] = rollouts[aid].finalize(cfg.ppo.gamma, cfg.ppo.gae_lambda, last_value, last_cost_value)

        # ------------------------------------------------------------------
        # G9 step 3 (docs/g9_gates.md). theta_k is pinned HERE: the PPO
        # rollout above is complete, and the PPO update below has not run.
        # Everything from this point to the dual update therefore refers to
        # exactly one policy, which is what makes `Jhat_agg^i` an estimate
        # of `J_C^i(theta_k)` -- the same quantity G0-G2's dual update
        # consumed, estimated differently. The 90 trajectories collected
        # here enter no buffer, touch no optimizer, and update no
        # normalization statistic; `finalized` above stays PPO's only data.
        # ------------------------------------------------------------------
        batch: BatchSourceResult | None = None
        theta_k_checksum: str | None = None
        if self.batch_sources is not None:
            theta_k_checksum = policy_checksum(self.agents, list(self.env.agent_ids))
            batch = self.batch_sources.collect(
                self.agents,
                self.round_index,
                collect_reference=self.round_index in set(cfg.source_collection.validation_rounds),
            )
        else:
            # G2-peer (docs/g2_gates.md): refit every agent's constraint-report
            # head EXACTLY ONCE this round, on that agent's own masked MC
            # cost-to-go targets, before any owner's `peer_critic` sources are
            # collected below. Refitting once per round (not once per query)
            # is required: a physical agent is queried as a peer by several
            # different owners in the loop that follows, and re-fitting on
            # each query would silently re-bias the head toward whichever
            # owner queried it most recently, rather than reporting one
            # consistent belief for the whole round.
            #
            # Skipped entirely under the G9 trajectory-batch architecture:
            # no source queries a head there, so refitting one would be
            # both wasted compute and a neural component executing inside a
            # round whose gate G9a-v says none may.
            for aid in self.env.agent_ids:
                fit_obs, fit_targets = self._cost_to_go_targets(finalized[aid])
                self.constraint_report_heads[aid].refit(fit_obs, fit_targets)

        round_record: dict[str, Any] = {"round_k": self.round_index, "constraints": {}}
        if batch is not None:
            round_record["source_batch"] = {
                "policy_checksum": batch.policy_checksum,
                "worker_checksums_all_match": True,  # collect() raises otherwise
                "n_chunks": len(batch.worker_checksums),
                "M": self.batch_sources.M,
                "R_m": cfg.source_collection.R_m,
                "n_trajectories": batch.n_trajectories,
                "env_steps": batch.env_steps,
                "wall_clock_s": batch.wall_clock_s,
                "reference_collected": batch.reference_mean is not None,
                "reference_n": batch.reference_n,
                "reference_wall_clock_s": batch.reference_wall_clock_s,
                # Per-owner within-round sampling statistics: the pooled
                # per-trajectory sd over this round's own 3 x R_m draws
                # (G9f's `s2_within`) and the between-source variance
                # (`s2_between`). Logged per round because sigma is a
                # property of theta_k and pooling it across rounds would
                # fold policy drift into a sampling-variance estimate.
                "per_owner": {
                    aid: {
                        "source_means": {
                            rid: batch.source_means[rid][aid] for rid in self.batch_sources.replica_ids
                        },
                        "sigma_hat": float(
                            np.sqrt(
                                np.mean([
                                    np.var(batch.per_trajectory[rid][aid], ddof=1)
                                    for rid in self.batch_sources.replica_ids
                                ])
                            )
                        ),
                        "s2_between": float(
                            np.var(
                                [batch.source_means[rid][aid] for rid in self.batch_sources.replica_ids],
                                ddof=1,
                            )
                        ),
                    }
                    for aid in self.env.agent_ids
                },
            }
            if self.source_logger is not None:
                self.source_logger.write({
                    "round_k": self.round_index,
                    "policy_checksum": batch.policy_checksum,
                    "seeds": {rid: [list(p) for p in pairs] for rid, pairs in batch.seeds.items()},
                    "reference_seeds": [list(p) for p in batch.reference_seeds],
                })
            if batch.reference_mean is not None and self.reference_logger is not None:
                # Written to its own file, never into `rounds.jsonl`, so
                # there is no path by which the high-precision reference
                # could be mistaken for something the dual update saw.
                self.reference_logger.write({
                    "round_k": self.round_index,
                    "policy_checksum": batch.policy_checksum,
                    "R_ref": batch.reference_n,
                    "reference_mean": batch.reference_mean,
                    "reference_per_trajectory": batch.reference_per_trajectory,
                    "source_means": batch.source_means,
                    "source_per_trajectory": batch.per_trajectory,
                })
        new_lam = np.zeros_like(self.lam)
        for i, aid in enumerate(self.env.agent_ids):
            reports = self.source_registry.collect(
                aid, self.round_index,
                functools.partial(
                    self._collect_source_value,
                    owner_id=aid,
                    owner_finalized=finalized[aid],
                    batch=batch,
                ),
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
                # Retained as a diagnostic (this run's own observed
                # disagreement over time -- useful to compare against the
                # calibration phase's reference distribution and to
                # detect drift) but P0 #8: no longer the source of
                # `epsilon_offline` below, which must come from a
                # dedicated clean calibration phase computed once, before
                # round 0, independent of this run's own history (see
                # safelie.eval.calibration).
                self.clean_run_disagreements.append(agg.spread)

            if cfg.defense.name == "rce" and self.calibration is not None:
                epsilon_offline = self.calibration.epsilon_offline
                guarantee_in_force = compute_guarantee_in_force(
                    getattr(agg, "applied_margin", 0.0), epsilon_offline
                )
            else:
                epsilon_offline = None
                guarantee_in_force = None

            new_lam[i] = residual_i  # temporarily store per-agent residual; mixed below

            # P0 #6 (return reporting). Everything in this record is a
            # LEARNER TRAINING quantity computed from this round's own
            # on-policy rollout under the pre-update policy theta_k:
            # `reported_cost_return`/`task_return` are GAE(lambda) value
            # TARGETS (ret_c[0]/ret_r[0]), not oracle-style episodic
            # Monte-Carlo returns, and `mechanism_reported_cost_return` is
            # the post-attack, post-aggregation return-scale estimate that
            # actually drove this round's dual update (`point_estimate`
            # above). None of these are the paper's reported task/cost
            # return for evaluation purposes -- those are the oracle's
            # fresh-rollout `episodic_*` quantities in oracle.jsonl,
            # computed by safelie.eval.harness/safelie.experiment. Do not
            # read this block as an evaluation metric.
            cost_values_arr = finalized[aid]["cost_values"]
            values_arr_r = finalized[aid]["values"]
            round_record["constraints"][aid] = {
                "reports": [{"source_id": r.source_id, "value": r.value} for r in reports],
                "corrupted_source_ids": sorted(corrupted_here),
                "aggregate": {
                    "point_estimate": agg.point_estimate,
                    "spread": agg.spread,
                    "retained_n": agg.retained_n,
                    "degenerate": agg.degenerate,
                    "applied_margin": float(getattr(agg, "applied_margin", 0.0)),
                    "pessimistic_estimate": float(getattr(agg, "pessimistic_estimate", agg.point_estimate)),
                },
                "guarantee_in_force": guarantee_in_force,
                "epsilon_offline": epsilon_offline,
                # The return-scale estimate that actually fed the dual
                # update this round (== point_estimate above): mean or
                # pessimistic_estimate, whichever cfg.defense.name uses.
                "mechanism_reported_cost_return": float(point_estimate),
                # The constraint residual the dual update consumes for
                # this agent BEFORE consensus mixing by W: point_estimate
                # - d. Logged because it is the middle link of the causal
                # chain G0 checks (critic -> residual -> lambda -> policy
                # -> true cost) and was previously only inferable by
                # re-deriving it from two other fields.
                "constraint_residual": float(residual_i),
                "reported_cost_return": finalized[aid]["cost_return_estimate"],
                "task_return": float(finalized[aid]["ret_r"][0]) if len(finalized[aid]["ret_r"]) else 0.0,
                # G1 (docs/g1_gates.md). BOTH constraint-objective
                # estimators, every round, computed from the SAME rollout,
                # whichever one is active -- so "did replacing the
                # estimator remove the systematic under-read?" is
                # answerable from one run's own logs rather than by
                # comparing two campaigns that also differ in their
                # trajectories. All values are raw return scale. None of
                # them is an evaluation metric: the oracle's
                # `true_cost_return` is truth, and it lives in
                # oracle.jsonl, written by a separate process.
                "constraint_estimators": {
                    # ret_c[0], the pre-G1 dual input: a GAE(lambda)
                    # bootstrap target, NOT an estimator of J_C.
                    "gae_lambda": finalized[aid]["cost_return_estimate"],
                    # sum_t gamma^t C_t over the round window on one global
                    # clock -- the same functional the oracle measures.
                    "mc_window": float(finalized[aid]["mc_cost_return"]),
                    # per-episode own-clock mean over COMPLETE episodes;
                    # diagnostic only (short-episode censoring bias, see
                    # safelie.training.constraint_return).
                    "mc_episodic": float(finalized[aid]["mc_cost_return_episodic"]),
                    "mc_n_complete_episodes": int(finalized[aid]["mc_n_complete_episodes"]),
                    "mc_censored_length": int(finalized[aid]["mc_censored_length"]),
                    "mc_episode_lengths": list(finalized[aid]["mc_episode_lengths"]),
                    "n_mc_targets": int(finalized[aid]["n_mc_targets"]),
                    # Discounted task return on the same window clock, so
                    # the learner's rollout carries a quantity directly
                    # comparable with the oracle's `episodic_task_return`.
                    "mc_window_task_return": float(finalized[aid]["mc_task_return"]),
                    # Which of the two actually fed `own_critic` and the
                    # replica/monitor regression targets this round.
                    "active": cfg.constraint_estimator,
                },
                "training_diagnostics": {
                    "train_reward_mean": finalized[aid]["reward_mean"],
                    "train_cost_rate_mean": finalized[aid]["cost_rate_mean"],
                    "n_terminated": finalized[aid]["n_terminated"],
                    "n_truncated": finalized[aid]["n_truncated"],
                    "cost_critic_prediction_t0": float(cost_values_arr[0]) if len(cost_values_arr) else 0.0,
                    "cost_critic_prediction_mean": float(np.mean(cost_values_arr)) if len(cost_values_arr) else 0.0,
                    "cost_value_target_t0": finalized[aid]["cost_return_estimate"],
                    "cost_value_target_mean": float(np.mean(finalized[aid]["ret_c"])) if len(finalized[aid]["ret_c"]) else 0.0,
                    "cost_advantage_mean": float(np.mean(finalized[aid]["adv_c"])) if len(finalized[aid]["adv_c"]) else 0.0,
                    "task_critic_prediction_t0": float(values_arr_r[0]) if len(values_arr_r) else 0.0,
                    "task_value_target_t0": float(finalized[aid]["ret_r"][0]) if len(finalized[aid]["ret_r"]) else 0.0,
                    "task_advantage_mean": float(np.mean(finalized[aid]["adv_r"])) if len(finalized[aid]["adv_r"]) else 0.0,
                },
            }

        residual_vec = new_lam
        # G9d-i is checked from the artifact rather than asserted here, so
        # record the two inputs the identity needs: the pre-update mixed
        # multiplier `W @ lambda_k` and the residual the dual consumed.
        mixed_lam = self.W @ self.lam
        self.lam = dual_update(self.lam, self.W, cfg.dual.eta_lambda, residual_vec, cfg.dual.lambda_max)

        # G9h-iii, extended to cover steps 4-6: the attack hook, the
        # aggregation and the dual update must not have moved theta
        # either. If any of them ever did, the dual update at round k
        # would be consuming an estimate of a policy that no longer
        # exists, and the timing of the algorithm would have changed
        # silently -- the section-19 stop condition this check exists for.
        if theta_k_checksum is not None:
            now = policy_checksum(self.agents, list(self.env.agent_ids))
            if now != theta_k_checksum:
                raise RuntimeError(
                    f"G9h FAILED at round {self.round_index}: theta changed between source "
                    f"collection and the PPO update ({theta_k_checksum} -> {now})."
                )
            round_record["source_batch"]["theta_k_checksum_stable_through_dual"] = True

        # G9c-v: `||theta_{k+1} - theta_k|| / ||theta_k||` over policy
        # parameters only (critics excluded -- they do not define the
        # trajectory distribution). `approx_kl` already says the update was
        # non-trivial in distribution space, but the gate was declared in
        # parameter space and the two can disagree: a policy whose log_std
        # has collapsed can post a healthy KL from a tiny parameter step.
        # One concatenation of ~52k floats per round; the cost is nil next
        # to a 2000-step rollout.
        theta_before = torch.cat([
            p.detach().reshape(-1) for aid in self.env.agent_ids
            for p in self.agents[aid].policy.parameters()
        ])

        for i, aid in enumerate(self.env.agent_ids):
            # `ppo_lagrangian_update` has always returned PPOUpdateStats;
            # the return value was simply dropped here, so policy loss /
            # value loss / entropy / approx-KL -- the four quantities that
            # say whether the optimizer itself is healthy -- were computed
            # every round and never recorded. Capturing them changes no
            # numerical result (the same call, the same order); it only
            # writes down what was already being produced.
            ppo_stats = ppo_lagrangian_update(self.agents[aid], finalized[aid], float(self.lam[i]), cfg.ppo)
            round_record["constraints"][aid]["lambda_after"] = float(self.lam[i])
            # G9d-i: `lambda_after - lambda_mixed_before == eta * residual`
            # on every unclipped cell, checkable directly from the log
            # without re-deriving `W @ lambda_k` from six other fields.
            round_record["constraints"][aid]["lambda_mixed_before"] = float(mixed_lam[i])
            round_record["constraints"][aid]["ppo"] = {
                "policy_loss": ppo_stats.policy_loss,
                "value_loss": ppo_stats.value_loss,
                "cost_value_loss": ppo_stats.cost_value_loss,
                "entropy": ppo_stats.entropy,
                "approx_kl": ppo_stats.approx_kl,
            }

        theta_after = torch.cat([
            p.detach().reshape(-1) for aid in self.env.agent_ids
            for p in self.agents[aid].policy.parameters()
        ])
        denom = float(torch.norm(theta_before))
        round_record["policy_param_rel_change"] = (
            float(torch.norm(theta_after - theta_before) / denom) if denom > 0 else 0.0
        )

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
            "constraint_report_heads": {
                aid: h.state_dict() for aid, h in self.constraint_report_heads.items()
            },
            "lam": self.lam,
            "round_index": self.round_index,
            "clean_run_disagreements": list(self.clean_run_disagreements),
            "rng_state": {
                "env": self.env_rng.bit_generator.state,
                "attack": self.attack_rng.bit_generator.state,
                "replicas": {sid: r.rng.bit_generator.state for sid, r in self.replicas.items()},
                "constraint_report_heads": {
                    aid: h.rng.bit_generator.state for aid, h in self.constraint_report_heads.items()
                },
                "torch_global": torch.get_rng_state(),
                "numpy_global": np.random.get_state(),
                # G9: the M+1 spawned source streams and the full set of
                # seeds already issued. Without these a resumed run would
                # restart its source streams from the spawn point and
                # re-issue seeds it had already used, which is exactly the
                # duplication gate G9g-ii forbids.
                "batch_sources": self.batch_sources.state_dict() if self.batch_sources else None,
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
        for aid, sd in state.get("constraint_report_heads", {}).items():
            self.constraint_report_heads[aid].load_state_dict(sd)
        self.lam = state["lam"]
        self.round_index = state["round_index"]
        self.clean_run_disagreements = list(state["clean_run_disagreements"])
        self.env_rng.bit_generator.state = state["rng_state"]["env"]
        self.attack_rng.bit_generator.state = state["rng_state"]["attack"]
        for sid, rng_state in state["rng_state"]["replicas"].items():
            self.replicas[sid].rng.bit_generator.state = rng_state
        for aid, rng_state in state["rng_state"].get("constraint_report_heads", {}).items():
            self.constraint_report_heads[aid].rng.bit_generator.state = rng_state
        torch.set_rng_state(state["rng_state"]["torch_global"])
        np.random.set_state(state["rng_state"]["numpy_global"])
        bs = state["rng_state"].get("batch_sources")
        if self.batch_sources is not None and bs is not None:
            self.batch_sources.load_state_dict(bs)
        return state.get("extra", {})

    def close(self) -> None:
        """Release the source worker pool. Idempotent, and safe to call on
        a run that never had one."""
        if self.batch_sources is not None:
            self.batch_sources.close()
        for logger in (self.source_logger, self.reference_logger):
            if logger is not None:
                logger.close()


def run_experiment(cfg: ExperimentConfig) -> Path:
    run = ExperimentRun(cfg)
    try:
        return run.run()
    finally:
        run.close()
