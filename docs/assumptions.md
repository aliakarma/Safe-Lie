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

## Environment: synthetic CPU stand-in, alongside Safe MAMuJoCo

**Classification: (C) Approximation**, clearly labeled everywhere it
appears.

**What it is for.** The real environments are now implemented
(`safelie.envs.mamujoco`, see the next section). The synthetic
environment remains the default for the test suite and the local demos,
so CI and a laptop need no MuJoCo, and so every component can be
exercised end to end without an optional dependency.

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

**To change:** point `env.name` at a MuJoCo configuration; nothing else
in the pipeline needs to change.

## Safe MAMuJoCo: velocity threshold calibrated, not inherited

**Classification: (B) Under-determined by the source materials** — the
paper specifies no threshold for its own primary environment.

**What was missing.** `main_iclr.tex` §5.1 names ManyAgent Ant (N=6) as
the primary environment and gives the budget `d=25`, but no velocity
threshold. Nor can one be inherited: the reference Safe MAMuJoCo's
`TASK_VELCITY_THRESHOLD` table has **no ManyAgent Ant entry at all**, and
its constructor asserts on the name. The paper's primary environment does
not exist in the implementation the paper cites.

**Why the threshold, not the budget, is the free parameter.** `d=25` is
`[SPEC]`. The threshold is not specified by anything, so it is the honest
place to absorb the calibration — the reverse choice would override a
number the paper actually states.

**Decision.** `velocity_threshold: 0.75` for ManySegmentAnt 6x1, set by
direct measurement (`scripts/calibrate_cost.py`) and then **corrected by a
completed run**, which is the part worth recording.

The first calibration picked `1.0`. At Safe MAMuJoCo's nearest table entry
(Ant 4x2 = 2.418) a freshly-initialized policy incurs cost on 0.75% of
steps, for a discounted cost return near 0.75 against `d=25` — roughly 30x
from binding, so `lambda` never leaves zero and all five pilot conditions
coincide for reasons unrelated to the hypothesis. This is the same §R6.1
confound, and the same measurement-driven fix, that set
`local_demo_*.yaml`'s budget to `d=5` on the synthetic environment. At
`1.0` the measured *true* per-agent returns are 11.6-36.9 against `d=25`,
which passes the initial-calibration bar.

**Passing that bar turned out not to be sufficient.** A full 250-round
condition-A run at `1.0` behaved as follows:

| | thr = 1.00 | thr = 0.75 |
|---|---|---|
| first round with `lambda` > 0 | 159 / 250 | 62 / 250 |
| `lambda` > 0, share of (round, agent) cells | 14.0% | 41.7% |
| `lambda` peak (cap `lambda_max` = 25) | 1.67 | 4.17 |

The reason is the caveat below: the dual update compares the budget
against the *learner's* estimate, and at `1.0` that estimate did not cross
`d=25` until round ~159 of 250. Roughly 86% of the run was therefore
structurally incapable of distinguishing condition A from B/C/D — the
§R6.1 failure re-entering through the estimate rather than through the
physics. At `0.75` the true cost is 1.66x budget and the initial estimate
0.42x rather than 0.23x, and a run shows a genuine closed-loop cycle:
`lambda` peaks near 3.6 around rounds 100-124, drives true cost down to
25.1 (at budget) while task return bottoms out, then relaxes as cost falls
below budget and re-engages as it drifts back.

**The general lesson**, worth applying to any future environment: an
initial-policy calibration bounds the problem from one side only. The
binding question is when the *learner's estimate* crosses the budget
relative to the run length, and that depends on cost-critic convergence
time (~150 rounds here), which is 60% of a 5x10^5-step pilot but 3% of the
paper's 10^7-step scale. Calibrate, then verify on a completed run before
spending a matrix.

**The caveat behind all of the above.** The dual update compares the
budget against the *learner's* cost-return estimate, not the true cost.
At initialization those differ by 3-4x — an untrained cost critic under
GAE (`gamma=0.99, lambda=0.95`, effective horizon ~17 steps) reads far
low. A run can therefore be genuinely constraint-relevant while `lambda`
sits at zero for many rounds. `scripts/calibrate_cost.py` reports both
numbers for this reason; neither alone settles whether a config is
well-posed.

## Safe MAMuJoCo: per-agent cost, resolving `[GAP]` G4

**Classification: (B) Under-determined**, and resolved differently from
the reference implementation for a stated reason.

**What was missing.** `[GAP]` G4 asks how one agent's cost could be
observable to a peer — the premise the peer-critic sources rest on.

**What the reference implementation does.** Safe MAMuJoCo computes a
single global speed indicator from the torso velocity and assigns the
identical scalar to every agent. Under that cost function G4 is vacuous:
every agent's cost is the same number, and the paper's per-agent
constraint `C^i` collapses to one shared constraint.

**Decision.** `cost_mode: per_agent_velocity` on the `gymnasium_robotics`
backend: each agent's cost is the speed of its own torso segment, read
from shared simulator state — measurable by a peer in principle, which is
exactly what G4 asks for, while keeping the per-agent structure the paper
assumes. Measured per-agent cost rates differ substantially across agents
(0.12-0.37 at initialization), so the per-agent constraint is doing real
work rather than six copies of one number.

`cost_mode: safe_mamujoco_shared` reproduces the reference behaviour
exactly on either backend, so the two are directly comparable.

## P0 implementation repair: the learner/source layer, before G0

**Classification: (A) Bug fixes and (B) methodological corrections**,
distinguished explicitly below per the repair task's own taxonomy. None
of this changes the threat model, the attack injection point, the
mathematical objective, the RCE mechanism, the source model, or any
theoretical definition. Full file-by-file detail is in `CHANGELOG.md`'s
"P0 implementation repair (pre-G0)" entry; this section records the
*evidence*, not just the diff.

**Why this section exists.** A full 5-seed, 5-condition pilot matrix
(`results/runs/pilot_{A..E}_*_seed{0..4}`) was executed before this
repair, under an implementation that had every bug below. That data
predates every fix in this section and must not be used for the paper --
it is retained on disk (not deleted) only as forensic evidence of what
the bugs did, via `scripts/audit_source_independence.py` run against it
(numbers below). Any future pilot run must write to a path that cannot
be confused with those directories (see this document's G0 section).

**Cost-critic bias, measured before and after.** Running
`scripts/audit_source_independence.py` against the pre-repair
`pilot_A_clean_seed0` (a *clean*, undefended, 250-round run -- no attack
to explain the numbers away) shows `own_critic` biased −10.8 to −15.1
across agents and `peer_critic` sources biased as far as −34.8, all
sustained for the full 250 rounds. A post-repair smoke run of 75 rounds
on the synthetic environment (`local_demo_clean.yaml`, fast enough to run
as a same-session check, not the paper's environment) shows the same
quantities -- own-critic bias and the aggregate ("mechanism") bias that
actually drives the dual update -- converging from −6.7 / −9.9 at the
first 10 rounds to −0.03 / +0.06 at the last 10, i.e. to within noise of
zero. Task return improved over the same window (−14.9 → −10.9), and the
dual variable responded and saturated at `lambda_max` under a budget the
policy could not satisfy within 75 rounds (true cost stayed ~4x the
`d=5` local-demo budget throughout -- expected behaviour for an
unconstrained-within-budget problem on this short a schedule, not a bug;
see this document's G0 section for the actual pilot's own calibration).
This is the direct evidence that P0 #1/#2/#3/#5's fixes address the
cost-critic bias mechanism, not merely its symptoms.

**Source independence remains NOT supported by measurement, even after
the peer-critic wiring fix -- report this honestly, do not paper over
it.** The peer-critic self-collision bug (owner-relative mapping, P0 #7)
was real: under the M=7 config, 4 of 6 agents received their own critic
relabeled as one of their four "peer" sources. Fixing it did not fix the
deeper problem. `scripts/audit_source_independence.py`'s participation-
ratio "statistical effective M" measures ~1.10 (against a nominal
`effective_M=7`) on the pre-repair `pilot_A_clean_seed0` run, and
measures the *same* ~1.10 on the post-repair local-demo smoke run. The
wiring bug was not the dominant cause of the M=7 -> ~1 collapse: all
seven sources are ultimately critic or small-regression networks trained
by similar procedures on a highly symmetric environment (six agents
facing near-identical dynamics), so their *errors* move together over
training time even when their weights are genuinely distinct and no
source literally duplicates another's computation. Assumption 1(ii)
(source independence) is not supported by this measurement, before or
after this repair. This is not something a further code fix resolves
without changing the environment's or the source ensemble's actual
diversity (e.g. genuinely heterogeneous agents, or sources with
structurally different failure modes rather than different random seeds
on the same architecture) -- which would be a scientific redesign, not a
P0 implementation fix, and is out of this repair's scope. Report
`nominal M` and `statistical effective M` side by side in the paper;
do not present the former as if it were a measured property.

**A previously undocumented compounding bias mechanism: MaMuJoCo's own
time limit.** `safelie.envs.mamujoco`'s underlying Gym environment
truncates at ~1000 steps (measured directly: `ManySegmentAnt` 6x1 under
zero action truncates at exactly step 999) -- well below every pilot
config's `rollout_length=2000`. Before this repair, `compute_gae`
treated that truncation exactly like a genuine termination (the agent
fell over): zero bootstrap, recursion cut. That silently capped
`ret_c[0]`'s effective horizon at whatever fraction of `rollout_length`
elapsed before the first time-limit reset, on top of the
already-documented GAE-effective-horizon-~17-steps bias from
gamma=0.99/lambda=0.95. Fixed (`safelie.training.gae`, `safelie.envs.
mamujoco`'s `info["final_observation"]`) by bootstrapping a truncation
from the critic's own value at the true final observation, matching
`terminated`'s and `truncated`'s distinct semantics rather than
collapsing them. This does not change any synthetic-environment number
(its own truncation always coincides with the buffer's last index,
where there is nothing left to bootstrap into regardless), but changes
every MaMuJoCo-backed round's cost-critic targets from what any run
executed before this fix produced -- another reason the pre-repair pilot
matrix cannot be reused.

**To change:** none of the above requires further action beyond what
`CHANGELOG.md` records, except the source-independence finding, which
requires either a documented, honest limitation in the paper or a
future, separately-scoped redesign of the source ensemble/environment
heterogeneity -- not a further patch to `safelie.sources`.

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
