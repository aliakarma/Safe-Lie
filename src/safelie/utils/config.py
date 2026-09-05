"""Structured, validated experiment configuration.

Report reference: Phase 0; smoke tests S2, S5, S12; decisions D3, D10, D12, D14.

The report recommends Hydra + structured configs. This repository uses
Pydantic v2 models over YAML instead: the project does not run Hydra-style
multi-run sweeps (compute is far too limited for the paper's 300+ run
grid, see PROJECT_REPORT.md §10.2), so Hydra's sweep machinery buys
nothing, while Pydantic gives the same fail-fast, schema-validated
construction with a much smaller dependency footprint. This is a
`[DECISION]` documented in docs/assumptions.md.

Every cross-field rule the report calls CRITICAL is enforced here, at
construction time, not at aggregation time:

- `M <= 2f` for a trimmed-mean-family defense raises (S5, D3).
- `effective_M` (independence classes, not raw report count) is what is
  checked against `2f + 1`, so a config that only "looks" robust by
  counting correlated replicas is rejected (S12, D10, W4).
- Topology agent count must match the environment agent count.
"""

from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class SourceSpec(BaseModel):
    source_id: str
    source_type: Literal[
        "own_critic", "peer_critic", "ensemble_replica", "monitor", "trajectory_batch"
    ]
    independence_class: str


class SourceCollectionConfig(BaseModel):
    """How the M constraint sources are produced (G9, docs/g9_gates.md).

    `"neural"` is the G0-G2 path: every source is a critic or a fitted
    regression head evaluated on the round's own PPO rollout
    (`safelie.sources.estimators`). It is the default so that no existing
    config changes meaning by the mere existence of this field.

    `"parallel_trajectory_batch"` is the G8-recommended architecture: per
    dual update the policy is pinned at `theta_k` and `M` replicas, each
    with its own environment instance and its own RNG stream, each collect
    `R_m` independent trajectories and report one Monte-Carlo scalar
    `(1/R_m) sum_r sum_t gamma^t C_{r,t}^i`. No critic, no GAE, no fitted
    head anywhere in the path (`safelie.training.source_batch`).

    `workers` is a **compute knob with no scientific content**: every
    trajectory's seeds are drawn in the main process from per-replica
    spawned streams, so the collected values are bitwise independent of
    how many processes ran them (asserted by
    `tests/unit/test_source_batch.py`). It is deliberately NOT tied to
    `M`: G8 requires `M` independent *batches*, not `M` processes, and
    conflating the two would either cap parallelism at 3 or make the
    replica count a performance decision.
    """

    mode: Literal["neural", "parallel_trajectory_batch"] = "neural"
    M: int = Field(default=3, ge=1)  # G8's operating point; not tuned by G9
    R_m: int = Field(default=30, ge=1)  # G8's operating point; not tuned by G9
    workers: int = Field(default=1, ge=1)
    # Chunks dispatched per worker per round. Scheduling only, same
    # rationale as `workers`: it changes how the trajectory list is
    # partitioned across `Pool.map`, never which trajectories are drawn.
    chunks_per_worker: int = Field(default=3, ge=1)
    seed_entropy: int = Field(default=286_314_957_402_113_664_887_331_205_920_951_063_913)
    # Independent high-precision reference collected under the SAME pinned
    # theta_k at these rounds, logged separately and never fed to the dual
    # update (docs/g9_gates.md step 3b, gate G9b).
    validation_rounds: list[int] = Field(default_factory=list)
    R_ref: int = Field(default=120, ge=1)


class SourcesConfig(BaseModel):
    sources: list[SourceSpec] = Field(min_length=1)

    @property
    def M(self) -> int:
        return len(self.sources)

    @property
    def effective_M(self) -> int:
        """Distinct independence classes — the M Theorem 2 is entitled to use (§R10.3)."""
        return len({s.independence_class for s in self.sources})


class TopologyConfig(BaseModel):
    name: Literal["complete", "ring", "star", "erdos_renyi", "identity", "shared_constraint"]
    n_agents: int = Field(gt=0)
    p: float | None = Field(default=None, ge=0.0, le=1.0)
    graph_seed: int = 0

    @model_validator(mode="after")
    def _erdos_renyi_needs_p(self) -> TopologyConfig:
        if self.name == "erdos_renyi" and self.p is None:
            raise ValueError("topology.p is required when topology.name == 'erdos_renyi'")
        return self


class EnvConfig(BaseModel):
    name: Literal[
        "synthetic_constrained_marl",
        "manyagent_ant",
        "halfcheetah_2x3",
        "halfcheetah_6x1",
        "ant_4x2",
        "safety_gym_nav",
    ]
    n_agents: int = Field(gt=0)
    budget: float = Field(gt=0)
    horizon: int = Field(default=200, gt=0)

    # Synthetic environment only. A real MuJoCo factorization determines
    # its own dimensions, and `safelie.training.loop` sizes networks from
    # the constructed environment (`env.obs_dim` / `env.action_dim`), not
    # from these fields -- setting them for a `manyagent_ant` config used
    # to silently build 8-dim policies for a 63-dim observation.
    obs_dim: int = Field(default=8, gt=0)
    action_dim: int = Field(default=2, gt=0)

    # Safe MAMuJoCo only -- see safelie.envs.mamujoco's docstring.
    agent_obsk: int = Field(default=1, ge=0)
    cost_mode: Literal["auto", "safe_mamujoco_shared", "per_agent_velocity"] = "auto"
    velocity_threshold: float | None = Field(default=None, gt=0)


class AttackConfig(BaseModel):
    name: Literal["none", "primary", "stealth", "benign_control"] = "none"
    f: int = Field(default=0, ge=0)
    budget_ratio: float = Field(default=0.0, ge=0.0)  # B/d, [SPEC] {0, 0.25, 0.5, 1.0}
    kappa: float = Field(default=1.0, gt=0.0)  # [GAP] G7, no paper value
    nu_ratio: float = Field(default=0.05, ge=0.0)  # [GAP] G7, no paper value
    direction: Literal["negative", "positive", "mixed"] = "negative"
    support: Literal["persistent", "selective"] = "persistent"
    adaptivity: Literal["static", "adaptive"] = "static"
    consistency: Literal["consistent", "byzantine"] = "consistent"

    # A1 (docs/a1_attack_gates.md §3). Which sources the adversary holds,
    # by `source_id`. `None` keeps the historical rule -- the first `f`
    # non-`own_critic` sources in config order
    # (`safelie.training.loop.select_corrupted_sources`), which always
    # picks `batch_1` under the G10 source list and would make "the
    # attacked source" a constant across every seed.
    #
    # Why this field exists rather than reordering the source list: under
    # `parallel_trajectory_batch`, replica RNG streams are assigned BY
    # POSITION -- `ParallelBatchSourceCollector.replica_ids` zips
    # `cfg.sources.sources` against `SeedSequence(seed_entropy).spawn()`
    # children in order (`safelie.training.source_batch`). Moving
    # `batch_2` to the front to attack it would hand it `batch_1`'s
    # stream, changing which 7,500 trajectories every replica draws and
    # destroying the common-random-number pairing between a seed's clean
    # and attacked runs. Naming the source instead leaves all M streams
    # exactly where they were.
    #
    # Validated at construction: every id must exist in `sources`, and
    # the count must equal `f` -- so a typo fails the config rather than
    # silently running an unattacked "attack".
    corrupted_source_ids: list[str] | None = None


class DefenseConfig(BaseModel):
    name: Literal["mean", "coordinate_median", "krum", "trimmean", "rce"] = "mean"
    f: int = Field(default=0, ge=0)
    beta: float = Field(default=1.5, ge=0.0)  # [SPEC] margin coefficient
    sigma_min: float = Field(default=1e-3, ge=0.0)  # floor on degenerate spread, §R10.2
    min_retained: int = Field(default=3, ge=1)  # |T| < this floors + warns, §R10.2
    use_reliability_weights: bool = False  # [GAP] G1, shipped OFF by decision D12

    # P0 #8 fix. `guarantee_in_force` must be calibrated from a dedicated
    # clean (attack-disabled) rollout, never from the run's own history:
    # an attacked run's `attack.name != "none"` for its entire length, so
    # self-referential calibration collected zero samples and
    # `epsilon_offline` silently defaulted to 0.0, making
    # `applied_margin >= epsilon_offline` trivially true every round. See
    # `safelie.eval.calibration.run_clean_calibration`.
    calibration_rounds: int = Field(default=20, ge=1)
    calibration_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)


class PPOConfig(BaseModel):
    clip: float = 0.2  # [SPEC]
    gae_lambda: float = 0.95  # [SPEC]
    gamma: float = 0.99  # [SPEC]
    lr: float = 3e-4  # [SPEC]
    epochs: int = 4  # [GAP] G10, pinned per D14
    minibatches: int = 4  # [GAP] G10, pinned per D14
    # P0 #4 methodological correction (not [SPEC], still [GAP] G11 --
    # revising a D14-pinned default the paper never specifies, not
    # overriding a specified value): was 0.0. A diagonal-Gaussian policy
    # with no entropy bonus has no pressure to maintain log_std once the
    # reward/cost gradient starts pulling it down, and D14's own
    # single-optimizer, unnormalized-target design (P0 #1/#2/#3/#5, fixed
    # alongside this) made that pull unusually strong early in training.
    # 0.001 is a small, standard continuous-control PPO value (e.g. the
    # CleanRL/SB3 default range for MuJoCo tasks is 0.0-0.01); it is a
    # generic exploration safeguard, not tuned against this repository's
    # own results.
    entropy_coef: float = 0.001  # [GAP] G11, revised per P0 #4
    value_coef: float = 0.5  # [GAP] G11, pinned per D14
    # Per-network now (P0 #5: safelie.training.ppo clips each of the
    # policy/value/cost-value optimizers separately), not a joint clip
    # over all three networks' concatenated parameters.
    grad_clip: float = 0.5  # [GAP] G11, pinned per D14
    hidden_dim: int = 64  # [GAP] G12, pinned per D14
    # P0 #5: None means "use `lr` for the critics too" (the pre-fix
    # behaviour, now via three separate optimizer objects instead of one
    # shared one -- see safelie.algos.networks.AgentBundle). Exposed so a
    # different critic learning rate can be configured without code
    # changes; not set to a nonzero-different value by default, since the
    # paper's [SPEC] `lr` says nothing about actor/critic separation and
    # this repository does not invent a second [SPEC] number.
    critic_lr: float | None = None


class DualConfig(BaseModel):
    eta_lambda: float = 0.035  # [SPEC]
    lambda_max: float = 25.0  # [SPEC]


class ExperimentConfig(BaseModel):
    run_id: str
    seed: int = 0
    env: EnvConfig
    topology: TopologyConfig
    sources: SourcesConfig
    attack: AttackConfig = AttackConfig()
    defense: DefenseConfig = DefenseConfig()
    ppo: PPOConfig = PPOConfig()
    dual: DualConfig = DualConfig()
    total_steps: int = Field(gt=0)
    rollout_length: int = Field(default=200, gt=0)
    output_dir: str = "results/runs"
    # Which quantity the constraint sources report -- i.e. what the dual
    # update actually estimates J_C^i(theta) with. This is a scientific
    # setting, not a tuning knob, and it is recorded in every run's
    # `run_metadata.json` config snapshot so no artifact is ambiguous
    # about which estimator produced it.
    #
    #   "mc_window"  (default) -- an explicit discounted Monte-Carlo sum
    #       over the round's own sampled reported cost, on one global
    #       clock, computed by `safelie.training.constraint_return`. No
    #       critic and no GAE in the path. Scale-matched by construction
    #       to `safelie.eval.oracle.OracleEvaluator`.
    #   "gae_lambda" -- the pre-G1 behaviour: `ret_c[0]`, the GAE(lambda)
    #       bootstrap target at the round's first step. Retained ONLY so
    #       the G0 artifacts remain reproducible and so the two estimators
    #       can be compared head to head. It is a documented defect as a
    #       constraint-objective estimator (see
    #       `safelie.training.constraint_return`'s docstring) and must not
    #       be selected for new science.
    #
    # `mc_window` is the default because the alternative is the defect;
    # a default that preserved the defect would silently propagate it into
    # every config that does not mention this field.
    constraint_estimator: Literal["mc_window", "gae_lambda"] = "mc_window"

    # G9 (docs/g9_gates.md). Absent => the G0-G2 neural source path, so
    # every pre-existing config keeps its exact meaning.
    source_collection: SourceCollectionConfig = SourceCollectionConfig()

    @model_validator(mode="after")
    def _cross_field_checks(self) -> ExperimentConfig:
        if self.topology.n_agents != self.env.n_agents:
            raise ValueError(
                f"topology.n_agents ({self.topology.n_agents}) must equal "
                f"env.n_agents ({self.env.n_agents})"
            )

        f = self.defense.f
        m_eff = self.sources.effective_M
        if self.defense.name in ("trimmean", "rce"):
            if m_eff <= 2 * f:
                raise ValueError(
                    f"Defense '{self.defense.name}' requires effective_M > 2f "
                    f"(distinct independence classes, not raw report count), but "
                    f"effective_M={m_eff} and f={f} give M-2f={m_eff - 2 * f} <= 0. "
                    "Trimmed mean is mathematically undefined at this operating "
                    "point (PROJECT_REPORT.md §13.2, W2). Raising rather than "
                    "clamping or falling back, per decision D3."
                )
        if self.attack.f > self.sources.M:
            raise ValueError(
                f"attack.f ({self.attack.f}) cannot exceed the number of "
                f"sources M ({self.sources.M})"
            )

        # A1. An explicit adversary set must name real sources and must
        # have exactly `f` of them. Both failures are silent otherwise: a
        # typo'd id corrupts nothing and the run looks clean, while a set
        # of the wrong size makes `f` disagree with what the defense's
        # M >= 2f+1 accounting assumed.
        if self.attack.corrupted_source_ids is not None:
            known = {s.source_id for s in self.sources.sources}
            named = self.attack.corrupted_source_ids
            unknown = [sid for sid in named if sid not in known]
            if unknown:
                raise ValueError(
                    f"attack.corrupted_source_ids names sources that do not exist: "
                    f"{unknown}. Configured sources are {sorted(known)}."
                )
            if len(set(named)) != len(named):
                raise ValueError(
                    f"attack.corrupted_source_ids contains duplicates: {named}"
                )
            if len(named) != self.attack.f:
                raise ValueError(
                    f"attack.corrupted_source_ids has {len(named)} entries but "
                    f"attack.f is {self.attack.f}; the adversary's size is one "
                    "quantity and the two must not disagree."
                )

        # P0 #7: `peer_critic_<k>` is resolved owner-relatively as agent
        # (owner_index + k) mod N (safelie.training.loop._peer_agent_id).
        # An offset that is a multiple of N would resolve to the owner
        # itself for every owner -- a self-peer for the whole config, not
        # an edge case for one agent -- so reject it here rather than
        # letting it surface as a per-round ValueError once training has
        # already started.
        n = self.env.n_agents
        for spec in self.sources.sources:
            if spec.source_type != "peer_critic":
                continue
            suffix = spec.source_id.rsplit("_", 1)[-1]
            if not suffix.isdigit():
                raise ValueError(
                    f"peer_critic source_id {spec.source_id!r} must end in an integer "
                    f"offset (e.g. 'peer_critic_1'); got {suffix!r}."
                )
            offset = int(suffix)
            if offset % n == 0:
                raise ValueError(
                    f"peer_critic source_id {spec.source_id!r} has offset {offset}, a "
                    f"multiple of env.n_agents ({n}); safelie.training.loop resolves "
                    f"peer_critic_<k> as agent (owner_index + k) mod N, so this offset "
                    f"would make every owner its own peer (P0 #7). Use an offset in "
                    f"1..{n - 1} not divisible by {n}."
                )

        # G9. The two source-production paths are mutually exclusive by
        # construction, and a config that mixes them would silently feed
        # the dual update a mean over two incomparable kinds of estimate
        # (a state-conditional head prediction and a policy-level Monte-
        # Carlo mean). Reject rather than aggregate.
        batch_specs = [s for s in self.sources.sources if s.source_type == "trajectory_batch"]
        if self.source_collection.mode == "parallel_trajectory_batch":
            if len(batch_specs) != len(self.sources.sources):
                offenders = sorted(
                    s.source_id for s in self.sources.sources if s.source_type != "trajectory_batch"
                )
                raise ValueError(
                    "source_collection.mode='parallel_trajectory_batch' requires every source "
                    f"to have source_type='trajectory_batch'; these do not: {offenders}. "
                    "Mixing a neural source with a trajectory-batch source would average a "
                    "state-conditional prediction into a policy-level Monte-Carlo mean."
                )
            if self.source_collection.M != len(self.sources.sources):
                raise ValueError(
                    f"source_collection.M ({self.source_collection.M}) must equal the number of "
                    f"configured sources ({len(self.sources.sources)}); they are the same "
                    "quantity (one replica per source) and must not disagree."
                )
            bad_rounds = [
                k for k in self.source_collection.validation_rounds
                if not 0 <= k < max(1, self.total_steps // self.rollout_length)
            ]
            if bad_rounds:
                raise ValueError(
                    f"source_collection.validation_rounds contains rounds outside this run's "
                    f"0..{max(1, self.total_steps // self.rollout_length) - 1} range: {bad_rounds}"
                )
        elif batch_specs:
            raise ValueError(
                "sources of source_type='trajectory_batch' require "
                "source_collection.mode='parallel_trajectory_batch'; got mode="
                f"{self.source_collection.mode!r}. A trajectory-batch source has no neural "
                "estimator to fall back on and would have no value to report."
            )
        return self


def load_experiment_config(path: str) -> ExperimentConfig:
    """Load and validate a YAML experiment config.

    Raises a Pydantic ``ValidationError`` (a ``ValueError`` subclass) on any
    malformed or scientifically-invalid config, before any environment,
    policy, or source is constructed. This is smoke test S2.
    """
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ExperimentConfig.model_validate(raw)
