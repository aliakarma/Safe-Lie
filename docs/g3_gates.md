# G3 -- source-architecture isolation, pre-declared decision rules

**Status: DECLARED BEFORE ANY G3 NUMBER EXISTS.** Written and committed
while `results/g3_source_diagnostic/` does not yet exist. Nothing below may
be edited after a G3 number has been read. If a rule turns out to have been
badly chosen, the correct response is to record that and say so, not to
move it.

## What G3 is for

`docs/g2_gates.md` recorded a FAIL: G2b, G2c, G2d and G2h(effective-M
reporting) all show `peer_critic` is still not a usable estimator of
`J_C^i` after the target-repair (peer's own MC cost-to-go instead of PPO's
`ret_c`). G3 does **not** attempt another repair. It is a fixed-dataset,
frozen-policy diagnostic whose only job is to decide **why** `peer_critic`
fails: an estimator/head defect (H1 -- normalization, capacity,
optimization, training budget) or a semantic/query defect (H2 -- a peer's
head is asked to answer a question, `J_C^i(o^i)`, that is different in
kind from the one it was trained to answer, `J_C^j(o^j)`).

No PPO update, no GAE, no dual update, no attack, no RCE, and no change to
the environment or the safety budget occurs anywhere in this diagnostic.
The env is built exactly as `configs/experiment/pilot_A_clean.yaml`
specifies and stepped under a **frozen** policy (the completed G2
`pilot_A_clean_seed0` checkpoint, round 249) with zero gradient updates to
`policy`/`value_net`/`cost_value_net`. The only things trained here are
the small regression heads under test, on a fixed, disk-persisted dataset
shared by every variant.

## Data construction (pre-declared)

* One frozen policy: `results/runs_constraint_mc_g2/pilot_A_clean_seed0/checkpoint.pt`
  (round 249, `git.sha == 7551e6f`), restored via `ExperimentRun.restore`.
  Single seed, not three -- this is an architecture diagnostic, not a
  replacement for G2's own multi-seed campaign; `docs/g3_gates.md` does
  not claim seed-generality and section M of the report must say so.
* 60 independent rollout rounds, `rollout_length=2000` (identical to the
  training config), reset seeds drawn from a generator seeded `777`
  (disjoint from every training/eval seed used anywhere else in this
  repo). Actions are sampled from the frozen policy's own distribution
  (matching how the production rollout that trains `peer_critic` heads
  is itself generated -- on-policy stochastic actions, not the
  deterministic mean action).
* Rounds 0-39 ("fit rounds"): masked `(obs, mc_cost_to_go)` pairs, using
  the SAME `discounted_cost_to_go`/`complete_target_count` functions
  `safelie.training.constraint_return` already uses for
  `constraint_report_heads`/`monitor`/`ensemble_replica` (`gamma=0.99`,
  `tol=0.01`, masking the trailing 459 of 2000 steps) -- pooled per agent
  across all 40 rounds (~61,600 rows/agent).
* Rounds 40-59 ("eval rounds"): per agent, the round's `obs[0]` (the exact
  query point `_collect_source_value` uses for `peer_critic` in
  production) and `mc_cost_return` (`discounted_window_return` at `t=0`,
  gamma=0.99) as ground truth `J_C^agent` for that round. This target is
  the same MC estimator G0/G1 already validated as unbiased by
  construction (no critic in the path); it is not the withheld oracle, by
  design (section 13 of the task: isolate source estimation from oracle
  sampling variance and dual/policy dynamics).
* One dataset, written once, read by every variant below. No variant
  collects its own data.

## Model variants (pre-declared)

**Architecture** (both normalization conditions): `Linear(63,16) -> Tanh
-> Linear(16,1)`, Xavier-uniform init, Adam. This is
`safelie.sources.estimators.DiversifiedReplica`'s existing architecture --
kept fixed so a pass/fail cannot be attributed to a capacity change that
was never tested against the production architecture.

**Training budget** (deliberately adequate, section 5 of the task):
2500 full-batch-bootstrap gradient steps per head (vs. production's 20),
lr=1e-3 (vs. production's 1e-2, lowered because 2500 steps at 1e-2
diverges on unnormalized targets of this scale), bootstrap resample size
= min(2048, n_rows) per step. Documented in full in the diagnostic
script; this budget is chosen to make "did not converge" an implausible
explanation for any remaining miscalibration, not to match the deployed
system.

**Normalization axis:** *raw* (no normalization, matching
`DiversifiedReplica` today) vs. *normalized* (`safelie.algos.
normalization.RunningMeanStd` on both input observation and regression
target, fit on that head's own fit-rows, output denormalized before any
metric is computed) -- the same convention `AgentBundle` already applies
to the policy and both PPO critics, per the task's instruction to reuse
the existing convention rather than invent a new one.

**One head per physical agent** (6 heads, each trained on that agent's
own fit-rows), under each normalization condition -- 12 heads total. V1
through V4 and the full 6x6 owner x source matrix are all read off this
one set of heads (see "Matrix construction" below); no separate model is
trained per (owner, source) cell.

## Matrix construction and its relation to V1-V4

Let `h_s` be the head trained on physical agent `s`'s own fit-rows, and
let `o_q[r]`/`J[a][r]` be eval round `r`'s query observation for agent
`q` / realized MC ground truth for agent `a`. Two 6x6 tables are computed
from the same 20 eval rounds x 6x6 predictions `h_s.predict(o_q[r])`:

* **Table "vs owner truth"**: cell `(q, s)` scored against `J[q]` -- this
  is exactly what a `peer_critic` source is asked to be useful for in
  production (does source `s`'s report help owner `q`?). Off-diagonal
  cells of this table ARE **V2**.
* **Table "vs source truth"**: cell `(q, s)` scored against `J[s]` --
  does source `s` at least track its own realized cost, regardless of
  where it was queried? Off-diagonal cells of this table ARE **V4**
  (head trained on `J_C^s`, queried at a foreign `o_q`, scored against
  its own training target).
* **The diagonal** (`q == s`) is identical in both tables and IS **V1**
  (`= V3`, since "a head trained on the owner's own target, queried on
  the owner's own observation" and "agent `j`'s head trained on agent
  `j`'s own target, queried on agent `j`'s own observation" are the same
  construction once `j` is set equal to the owner -- V3 is reported
  explicitly, not silently dropped, but it reuses V1's numbers rather
  than training a redundant model).

This produces the full 36-cell matrix section 7 of the task asks for
without training 36 separate models, and it is read from the same fixed
dataset and the same 12 heads used for every other number in the report.

## Metrics (pre-declared)

For every (variant, agent-or-cell): `bias = mean(pred - truth)`,
`MAE = mean(|pred - truth|)`, `RMSE = sqrt(mean((pred-truth)^2))`,
`corr = Pearson(pred, truth)` over the 20 eval-round samples for that
cell, and `FSR = P(pred <= d | truth > d)` with `d = 25` (the config's own
budget, unmodified). Pooled (mean over cells) AND per-cell numbers are
both reported; the task explicitly warns against pooling-only reporting
after G2's owner heterogeneity.

## Calibration bar ("good" vs "poor"), pre-declared

Reused directly from `docs/g2_gates.md`'s own conventions so this
diagnostic's bar is not invented after seeing the data:

* **Good**: `|bias| <= 0.20 * d = 5.0` AND `corr >= 0.5` (the same
  correlation floor `g2_gates.md` already used to describe `own_critic`
  and `monitor` as "actually tracking" `J_C`, distinct from
  `peer_critic`'s measured -0.10 to +0.01).
* **Poor**: `corr < 0.2` (the range G1 actually measured for the
  pre-repair peer critic) OR `|bias| > 5.0`.
* Anything between "good" and "poor" is reported as-is and called neither.
* **FSR acceptable**: `<= 0.20`, G2d's own absolute bar, now computed
  against the MC target rather than the oracle.
* **Normalization materially fixes X** iff normalization reduces
  `|bias|` or `MAE` by `>= 50%` relative to the raw variant (G2b's own
  relative-improvement convention) **and** the normalized variant reaches
  `corr >= 0.5`. Anything short of both clauses is "does not materially
  fix", per the task's explicit instruction not to credit a modest
  improvement as a fix.

## Exchangeability bar, pre-declared

Per agent, over the 20 eval rounds: mean and variance of `J[agent]`, and
the 6x6 Pearson correlation matrix of `J[agent]` across rounds. Agents
are called **exchangeable** iff every off-diagonal pairwise correlation
is `>= 0.5` and no agent's mean `J` differs from the pooled mean by more
than 2x. `agent_0` (ManySegmentAnt's head segment, structurally distinct
per `safelie.envs.mamujoco`'s Deviation-1/observation-padding docstring)
is reported separately from the five interior segments for exactly this
reason -- it is flagged as a plausible exchangeability outlier before any
number is read, not after.

## H1 vs H2 decision rule, pre-declared (task section 12)

* **Outcome A (H1 supported)**: V1 good AND V2 good in the *normalized*
  condition but not the raw condition (i.e., normalization alone closes
  the gap per the bar above), and V3 does not do meaningfully better than
  V2 once both are normalized.
* **Outcome B (H2 supported)**: V1 (=V3) good, in BOTH normalization
  conditions, while V2 is poor in BOTH conditions -- normalization does
  not materially fix V2 per the bar above.
* **Outcome C (Both)**: normalization materially improves V2 (per the bar
  above) but V2's normalized corr/bias still falls short of V1/V3's
  normalized numbers by a wide margin (corr gap `>= 0.3` absolute, or V2
  fails "good" while V1/V3 pass it) -- both an estimator defect and a
  semantic defect are present.
* **Outcome D (Neither / deeper problem)**: V1 (=V3) itself fails "good"
  in the normalized condition -- even the in-distribution, correctly-wired
  case cannot hit calibration, meaning the defect is upstream of both H1
  and H2 (target construction, environment cost definition, or insufficient
  fit-data), and the whole `constraint_report_heads` design needs
  reconsideration, not just its cross-agent query.

Only one of A/B/C/D may be declared true; if the raw numbers are
ambiguous between two, that ambiguity itself is reported in section I
rather than forced into a box.

## G3 verdict rule, pre-declared

* **PASS** -- V2 (current peer design) reaches "good" calibration in at
  least the normalized condition. The architecture is valid enough to
  continue; G2's failure was purely an estimator defect already fixable
  by a bounded normalization change.
* **CONDITIONAL PASS** -- Outcome A: H1 alone explains the gap, and a
  bounded, already-scoped implementation change (add normalization to
  `constraint_report_heads`, matching `AgentBundle`'s existing convention)
  is sufficient in principle. Requires a note that this has NOT been
  re-verified inside the full 250-round training loop -- that is future
  work, explicitly not run here per the task's ceiling.
* **FAIL** -- Outcome B, C, or D. The current `peer_critic` design (a
  physical peer's head, trained on that peer's own target, queried at a
  different agent's observation) is scientifically invalid as a `J_C^i`
  estimator, or the underlying head architecture cannot hit calibration
  even in-distribution. Per the task's absolute rule, this blocks the
  attack study and blocks re-running G2's training loop again until a
  redesigned source architecture is chosen (task section 9, options A-D)
  and itself diagnosed the same way before being trained end-to-end.

On FAIL, `M_effective` for any future run must be recomputed by counting
only sources whose calibration clears "good" above -- `own_critic` and
`monitor` (already G1/G2-validated) plus whichever `constraint_report_heads`
variant, if any, clears the bar. `M = 7` (or any specific M) must NOT be
assumed; the task's section 10 applies verbatim.

## What this diagnostic explicitly does not establish

Regardless of verdict: no claim about Assumption 1(ii) (source
independence / honest-majority), no claim about Theorem 2, no claim about
attack stealth or RCE, no claim generalizing beyond `pilot_A_clean_seed0`'s
one frozen policy snapshot, and no claim that a "good" V1/V3 number here
would survive being re-embedded in the full nonstationary 250-round
training loop (where the head is refit every round on a moving policy,
not evaluated once on a frozen one). These are the same standing
limitations `docs/g2_gates.md` already declared for correlation-based
diagnostics, restated here because the task's section 11 requires them
restated, not assumed carried over.
