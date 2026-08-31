# Assumptions and Decisions

Every non-trivial engineering choice made while building this repository
that the paper (`main_iclr.tex`) or report (`PROJECT_REPORT.md`) does not
fully specify. Each entry states what was missing, what was decided, why,
and how to change it later. Cross-referenced against the report's own
`[DECISION]` labels (D1-D14) and gap register (G1-G20) where applicable.

## Configuration system: Pydantic + YAML, not Hydra

**What was missing.** The report recommends Hydra + structured configs.

**Decision.** Pydantic v2 models loaded from plain YAML
(`safelie.utils.config`).

**Why.** Hydra's core value is multi-run sweep orchestration. This
repository's compute budget does not support the paper's 300+ run grid,
and the compact pilot's 12-15 runs are run sequentially by a Colab
notebook, not swept by a job launcher (per the report's own decision, §R7.1).
Hydra would add a real dependency and complexity for no exercised benefit,
while Pydantic gives the same fail-fast, schema-validated construction
the report actually needs (S2, S5, S12).

**To change:** swap `safelie.utils.config.load_experiment_config` for a
Hydra `@hydra.main` entry point; the `ExperimentConfig` dataclass schema
can be reused as a Hydra structured config with minimal changes.

## Environment: synthetic CPU stand-in, not Safe MAMuJoCo

**Classification: (C) Approximation**, clearly labeled everywhere it
appears.

**What was missing.** Safe MAMuJoCo / Safety-Gymnasium require MuJoCo and
multi-agent wrapper packages not installed in this build, and the report
itself assigns real MARL training to a Colab GPU stage, not local
construction (§R7.1).

**Decision.** `safelie.envs.synthetic.SyntheticConstrainedMarlEnv`: each
agent drives a scalar state toward zero under a shared reward; per-agent
cost equals squared action norm. Chosen only so a random policy's
discounted cost return clears a reasonable budget and a trained policy
can bring it below one — i.e., so the constraint can genuinely bind.

**Why this scale, not literally d=25.** `d=25` is calibrated to Safe
MAMuJoCo's cost dynamics. On the synthetic environment, an untrained
policy's GAE-corrected cost-return estimate is on the order of 5-25
depending on horizon/gamma, and peer-critic/monitor sources need several
rounds of training before their estimates approach that scale at all. At
`d=25`, the aggregate estimate never approaches the budget within a
laptop-feasible run, `lambda` stays clipped at 0 throughout, and — this
was measured, not assumed — every attack/defense/control condition
produced bitwise-identical logs, because the projection ate the entire
attack's effect. This is exactly the "constraint not binding" confound
`PROJECT_REPORT.md` §R6.1 names. `local_demo_*.yaml` uses `budget=5.0`,
found by direct measurement (see `docs/reproducibility.md`) to make the
constraint begin to bind by round ~30 of a ~75-round run.

**To change:** implement `safelie.envs.mamujoco.build_mamujoco_env` per
its docstring; nothing else in the pipeline needs to change.

## Oracle isolation: no `true_cost` field, not a per-field guard

**What the report suggests.** Put `true_cost` on `DualCostStep` (the
per-step object returned to the learner) and rely on a runtime guard to
stop the learner reading it.

**Decision.** `DualCostStep` has no `true_cost` field, under any name, at
all. True cost is obtainable only via a capability-based handle
(`env._oracle_handle_privileged()`) that is not part of the public
`DualCostEnvWrapper` protocol.

**Why.** The report itself warns: *"If the learner can reach true_cost
through any path (shared dict, info field, logging callback), all
results are silently invalid."* Removing the field removes the path
structurally rather than relying on discipline at every call site. See
`safelie.envs.guards` and `docs/architecture.md`'s isolation-boundary
section.

## The oracle evaluation is a separate rollout, not inline in training

**What was initially tried, and reverted.** An earlier version of
`safelie.training.loop.ExperimentRun.run_round()` constructed an
`OracleEvaluator` directly against the training rollout's own environment
instance, for convenience.

**Why reverted.** This violated the report's literal S10 requirement
("no learner module imports the oracle module") even though the value
was only used for logging, never for any decision. `safelie.eval.oracle`
was moved out of `safelie.training` entirely; the oracle evaluation is now
a structurally separate rollout (`safelie.eval.harness.evaluate_true_cost`),
run by the orchestrator (`safelie.experiment`) after the learner's round
completes, using the just-updated policy weights. This is enforced by an
AST-level grep test (`tests/isolation/test_oracle_isolation.py`), not a
convention.

## Peer critic observability (`[GAP]` G4)

**Decision.** Report's recommended option (1): a peer's cost-value
network is evaluated on the constraint owner's own initial observation
for the round — i.e., restricted to state any agent's network can read,
never to another agent's private internal state.

**Why.** The paper never states an observability assumption for how a
peer critic can estimate agent i's constraint at all; the report
recommends this restriction as "safest, and probably intended."

## Ensemble/monitor diversification (`[GAP]` G5)

**Decision.** `safelie.sources.estimators.DiversifiedReplica`: an
independently-initialized small regression head, refit every round via a
few gradient steps on a **bootstrap resample** of the constraint owner's
own (observation, cost-to-go) pairs.

**Why.** The report calls this the mechanism that determines "whether
replica sources are independent at all," and recommends exactly
independent init + bootstrap-resampled minibatches.

## Reliability weights: implemented as a flag, never a mechanism (`[GAP]` G1, decision D12)

**Decision.** `DefenseConfig.use_reliability_weights` exists and defaults
to `False`. No update rule is implemented behind it.

**Why.** Algorithm 1 declares and initializes these weights (lines 2,
10) but no line of the paper's own pseudocode reads them. Inventing an
update rule the paper does not specify and presenting it as "the"
reliability-weight mechanism would misattribute a design choice to the
paper. Per decision D12, shipped off so the component with an actual
guarantee (trimmed mean + margin) is what runs by default.

## RCE degenerate cases: raise, don't clamp (decision D3)

**Decision.** `M <= 2f` raises `ValueError` at both config-validation
time (`ExperimentConfig`) and aggregation time
(`trimmed_mean_aggregator`). `|T| < min_retained` (default 3) floors the
spread at `sigma_min` and sets `degenerate=True` with a logged warning —
never a silent `spread=0.0`.

**Why.** The report identifies a silently-zero margin as "the single most
dangerous bug this repository could contain," because it produces
plausible-looking numbers while RCE has silently degraded to plain
trimmed mean.

## MAD: unscaled, not the normal-consistent 1.4826× convention

**Decision.** `safelie.defenses.base.mad` computes the raw median absolute
deviation, no scaling constant.

**Why.** The paper specifies "MAD" without a scaling convention. This
only rescales the effective `beta`; it does not change any qualitative
conclusion (which aggregator wins, whether the margin degrades at large
f, etc.). Documented here rather than silently choosing one convention.

## Attack strength scale: `budget_ratio * d`, matching the paper exactly

No decision needed here — the paper is explicit (`B/d in {0, 0.25, 0.5,
1.0}`) and this repository implements it literally
(`AttackConfig.budget_ratio`).

## Independence-class construction for the two named source configurations

**Decision.** `safelie.sources.registry.default_m7_sources()` gives every
one of the 7 sources its own independence class (own critic + 4 peers +
2 monitors, each a distinct process in the paper's intended reading).
`default_m5_sources()` (the paper's own N=2, M=5 configuration)
deliberately gives the 3 ensemble replicas a **shared** independence
class, since they live in one agent's process and are corrupted
simultaneously by a single host compromise.

**Why.** This is not an arbitrary choice — it is the concrete
demonstration of weakness W4 the report identifies: `effective_M` for
the N=2 config is 3, not the nominal 5, and `safelie.governance.auditor`
correctly rejects it at `f=2` even though `5 >= 2*2+1` holds on paper.

## Network architecture, PPO hyperparameters (`[GAP]` G10-G12, decision D14)

**Decision.** One pinned, documented set (PPO clip 0.2, GAE λ 0.95, γ
0.99, lr 3e-4 — all `[SPEC]`; epochs=4, minibatches=4, hidden_dim=64,
entropy_coef=0, value_coef=0.5, grad_clip=0.5 — all `[GAP]`, chosen as
reasonable small-network defaults), identical across every condition in
a given experiment matrix.

**Why.** The report lists 16+ hyperparameters the paper never specifies
and states the requirement is that they be "identical and recorded," not
individually optimal. `PPOConfig` in `safelie.utils.config` is the single
place these are pinned; every config file inherits the same defaults
unless explicitly overridden.

## Notebooks: orchestration only (decision D13)

**Decision.** No notebook cell defines a `def` or `class`; enforced by
`scripts/lint_notebooks.py` (AST-based, so a docstring merely discussing
a function name doesn't trip it) and run in CI.

**Why.** The report: "notebook-only code is invisible to code review,
untested by CI, and unreachable from the local smoke suite."

## No `.env.example`

**Decision.** Not created.

**Why.** This repository has no secrets, API keys, or external service
credentials of any kind — every configuration is a versioned YAML file.
Creating an unused `.env.example` would misrepresent the repository as
needing environment-based secrets it does not need. Documented here
rather than silently omitted.

## No `docker-compose.yml`

**Decision.** Not created; a single `Dockerfile` suffices.

**Why.** Nothing in this repository requires multi-service orchestration
— there is no database, message queue, or separate API server. Forcing a
compose file would add complexity with no exercised benefit.

## No `scripts/preprocess.py`

**Decision.** Not created.

**Why.** This is on-policy RL — data is generated by interaction, not
loaded from a static dataset. There is no preprocessing step to script.
