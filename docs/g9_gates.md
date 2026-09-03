# G9 -- does the parallel trajectory-batch source architecture survive integration into the LIVE clean training loop? Pre-declared decision rules

**Status: DECLARED BEFORE ANY G9 NUMBER EXISTS.** Written and committed
while `results/runs_constraint_batch_g9/` does not exist, before a single
line of the G9 source-collection implementation has been written. Nothing
below may be edited after a G9 number has been read. If a rule turns out
to have been badly chosen, the correct response is to record that and say
so, not to move it.

The G0-G2 baseline numbers quoted in this file (G9e's bar in particular)
are read from artifacts that already existed on disk before G9 began --
`results/runs_g0/`, `results/runs_constraint_mc_g1/`,
`results/runs_constraint_mc_g2/`, all from committed campaigns. They are
inputs to the pre-declaration, not G9 results.

## What G9 is for

G8 (`docs/g8_gates.md`, `results/g8_nonstationary_source_diagnostic/`)
reached a conditional pass. It established, at the construction level:

1. disjoint trajectory batches give calibrated estimates of the same
   policy-level `J_C^i(theta_k)` when the policy is genuinely frozen
   (G8a: 3 of 5 anchors at the 5-of-6 owner bar; the two failing anchors
   failed by finite-sample noise at |z| slightly over 2, not by bias);
2. **sequential** source collection is not viable at the required
   precision -- at `R_m = 30` the batch spans 30 training rounds and the
   induced policy drift reaches `R_drift ~ 5-8` standard errors
   (G8c-Delta failed at every anchor);
3. **parallel** collection under a pinned `theta_k` removes target drift
   by construction (`R_drift = 0` identically), at the price of `M`
   simulators;
4. `M = 3`, `R_m = 30` is the viable statistical operating point.

G8's own closing sentence names the remaining gap verbatim: "It does not
authorise wiring the construction into
`safelie.training.loop.ExperimentRun`; that remains a separate decision
for the reviewer after this report."

G9 closes exactly that gap and nothing else:

> Can the parallel trajectory-batch source architecture replace the
> neural source machinery inside the real clean training loop without
> destroying learning or producing a false-safe constraint signal?

## What G9 does not touch

No attack runs (`attack.name = "none"` throughout). No RCE. No topology
sweep. No baseline comparison beyond reading already-committed G0/G1/G2
artifacts. No beta/f sweep. No environment change. `budget_d = 25.0`
unchanged. `eta_lambda = 0.035`, `lambda_max = 25.0`, `gamma = 0.99`,
`rollout_length = 2000`, `total_steps = 500000` (250 rounds) unchanged.
`M = 3` and `R_m = 30` are inherited from G8 and are **not** tuned by G9
under any circumstances. No PPO hyperparameter is changed. No paper edit.

The only new code is (i) a source-collection module that produces the
three scalars, (ii) the config plumbing that selects it, and (iii) G9's
own analysis script.

## Notation, fixed before any number exists

* `G_r^i = sum_t gamma^t C_{r,t}^i` -- one trajectory's realized
  discounted cost return over the fixed 2000-step window, computed by
  `safelie.training.constraint_return.discounted_window_return` on the
  learner-visible `reported_cost` stream. The same functional G3-G8 used,
  the same one `safelie.eval.oracle.OracleEvaluator` computes. Under
  `attack.name = "none"` reported cost is numerically identical per step
  to the environment's true-cost stream (verified in G8 over all 1500
  (round, agent) pairs of the G2 seed-0 run, max abs diff 0.0), so a
  clean-condition `G_r^i` is a true-cost quantity. That coincidence is a
  correctness check, not a channel: the collection path never touches
  `_oracle_handle_privileged`.
* `theta_k` -- the policy parameters (policy network **and** observation
  normalization statistics, since both determine the trajectory
  distribution) in force when round `k`'s PPO rollout is drawn, before
  round `k`'s PPO update.
* `Jhat_{C,m}^i(theta_k) = (1/R_m) sum_{r in B_m} G_r^i` -- replica `m`'s
  sample mean over its own batch `B_m` of `R_m = 30` trajectories drawn
  under pinned `theta_k` from its own RNG stream.
* `Jhat_agg^i = (1/M) sum_m Jhat_{C,m}^i` -- the aggregate, `M = 3`. Under
  `defense.name = "mean"` and `attack.name = "none"` this is exactly the
  `point_estimate` that reaches the dual update.
* `Jhat_ref^i(theta_k)` -- an `R_ref = 120` trajectory sample mean under
  the same pinned `theta_k`, drawn from a fourth RNG stream disjoint from
  all three source streams, collected only at the five validation rounds
  below. **Never enters the dual update, the aggregate, or any log the
  learner reads.** The best available stand-in for the unknown
  `J_C^i(theta_k)`; always reported with its own `SE_ref`, never called
  exact.
* `sigmahat_i(theta_k)` -- the within-round, per-trajectory sample sd of
  `G_r^i`, pooled across the round's own 3 x 30 source trajectories (87
  degrees of freedom). Per round, per owner. Never pooled across rounds,
  because `theta` changes between rounds and pooling would fold policy
  drift into a sampling-variance estimate.
* `SE_m = sigmahat_i/sqrt(R_m)`, `SE_agg = SE_m/sqrt(M) = sigmahat_i/sqrt(90)`,
  `SE_ref = sigmahat_i/sqrt(R_ref)`.
* `d = 25.0`, the per-agent budget. Unchanged.

## The algorithm G9 implements, fixed before implementation

Per round `k`, in this exact order. `theta_k` denotes every agent's
policy state at the top of the round.

```
1. PPO ROLLOUT       one 2000-step on-policy rollout under theta_k,
                     env seeded from ExperimentRun.env_rng.
                     ---> PPO's data. Unchanged from G0-G2.
2. FINALIZE          GAE(gamma=0.99, lambda=0.95), ret_r, ret_c, adv_r,
                     adv_c from that rollout. Unchanged.
3. SOURCE COLLECTION theta_k pinned. M=3 replicas, each with its own
                     RNG stream and its own environment instance,
                     each collecting R_m=30 independent 2000-step
                     trajectories. Read-only: no gradient, no optimizer
                     step, no normalization-statistics update, no head
                     refit, no critic forward pass.
                     ---> 3 scalars per owner, Jhat_{C,m}^i.
3b. (VALIDATION ROUNDS ONLY, k in {25,75,125,175,225})
                     a fourth stream collects R_ref=120 further
                     trajectories under the SAME pinned theta_k.
                     Logged to a separate file. Not fed anywhere.
4. ATTACK HOOK       apply_attack on the 3 residuals. G9 is clean, so
                     this is the identity map; the hook is exercised
                     to prove the plumbing is intact.
5. AGGREGATION       defense.name="mean" over the 3 values -> point
                     estimate -> residual_i = point_estimate - d.
6. DUAL UPDATE       lambda_{k+1} = clip(W lambda_k + eta * residual, 0, 25).
                     Unchanged.
7. PPO UPDATE        theta_k -> theta_{k+1} using STEP 1's data and
                     lambda_{k+1}. Unchanged.
8. ORACLE EVAL       withheld evaluator, separate rollout under
                     theta_{k+1}. Unchanged, evaluator-side only.
```

Two properties of this ordering are the whole point and are gated below:

* **PPO's data is step 1's rollout and only step 1's rollout.** The 90
  source trajectories are never added to any PPO buffer, never update a
  normalization statistic, never touch an optimizer. The source
  architecture is being tested; PPO is not being redesigned. (G9a-iv.)
* **Steps 3-6 all occur while theta is still `theta_k`.** The dual update
  at round `k` therefore consumes an estimate of `J_C^i(theta_k)` -- the
  same quantity it consumed in G0-G2, estimated differently. The
  mathematical algorithm is unchanged; only the estimator is replaced.
  (G9h.)

The neural source path (`own_critic`, `peer_critic`,
`ensemble_replica`, `monitor`, `constraint_report_heads`,
`DiversifiedReplica`) is not merely unused in G9 -- it must be
structurally absent from the source pipeline, asserted in code (G9a-v).

## Determinism requirement, declared before implementation

Trajectory seeds are drawn **in the main process** from `M` PCG64 streams
spawned from one `numpy.random.SeedSequence`, one stream per replica, and
handed to workers as data. A worker is a pure function of
`(theta_k, env_seed, torch_seed)`. Therefore the number of worker
processes is a compute knob with **no** effect on any number G9 reports.
This is asserted by a test (`workers=1` and `workers=3` must produce
bitwise-identical source values), not assumed. A worker count that
changed the science would be a defect, not a configuration.

## Computational cost, computed and declared BEFORE the run

Measured on the run machine (Windows 11, 12 logical CPUs, `.venv`
CPython 3.11.9, `torch.set_num_threads(1)` per worker) prior to writing
any G9 code:

| quantity | measured |
|---|---|
| frozen-policy 2000-step source rollout, dedicated process | 5.65-5.96 s |
| ... of which `env.step` | 2.60 s |
| ... of which policy forward + sample (6 agents) | 2.86 s |
| G2-style training round (rollout + 6 head refits + 7 neural sources + PPO) | 10.23 s |
| aggregate throughput, 1 / 3 / 6 / 9 / 11 workers (rollouts/s) | 0.177 / 0.490 / 0.811 / 0.909 / 0.963 |

Per dual update, at `M = 3`, `R_m = 30`, `rollout_length = 2000`:

| quantity | G2 (neural sources) | G9 (trajectory batches) | ratio |
|---|---:|---:|---:|
| PPO environment steps | 2,000 | 2,000 | 1x |
| source environment steps | 0 | 180,000 | -- |
| oracle (evaluator) environment steps | 2,000 | 2,000 | 1x |
| total environment steps | 4,000 | 184,000 | **46x** |
| source : PPO step ratio | 0 : 1 | **90 : 1** | -- |

Wall clock, projected at 9 workers (0.909 rollouts/s aggregate):

* source collection: 90 / 0.909 = **99 s per round**
* PPO round (rollout + update, neural sources removed): **~8 s** (estimate;
  replaced by the smoke test's measurement before the primary run)
* oracle evaluation: **~6 s**
* **total ~113 s per round -> 250 rounds = 28,250 s = 7.85 h**
* plus 5 validation rounds x 120 reference trajectories = 660 s = 0.18 h
* **projected total ~8.0 h**, against G2's measured 5,250.7 s (1.46 h)
  for the identical 250 rounds -- **5.5x wall clock at 46x the
  environment steps**, the gap being what the 9-way parallelism buys.

Serial (single-process) source collection would be 90 x 5.8 = 522 s per
round, 36.3 h for 250 rounds. Parallelism is not an optimization here; it
is what makes the architecture runnable at all, and it is also what G8
requires scientifically (a pinned `theta_k` across replicas).

This cost is stated here, in advance, rather than behind the phrase
"three sources", because it is a first-class result: **the architecture
that makes the constraint estimate valid costs 46x the environment
interaction of the one that does not.** G9's verdict must state what that
implies for the paper's full factorial, not only for one clean run.

## Gates (pre-declared)

### G9a -- source correctness (structural, asserted in code)

For every round `k` and every owner `i`:

* **i.** exactly `M = 3` source values exist;
* **ii.** all three were generated under the same pinned `theta_k`,
  verified by a SHA-256 checksum over each replica's received policy
  parameters plus observation-normalization statistics, compared against
  the main process's checksum of `theta_k` (see G9h);
* **iii.** the three source batches used pairwise-disjoint RNG seed sets,
  and no source seed collides with the learner's own `env_rng` stream or
  the evaluator's `eval` stream (see G9g);
* **iv.** the 90 source trajectories contributed nothing to PPO: no
  optimizer step, no `update_normalization_stats` call, no buffer append
  occurs inside source collection;
* **v.** no neural source path executes: no `DiversifiedReplica.refit`,
  no `constraint_report_heads` query, no `AgentBundle.cost_value` call,
  no `compute_gae` call, no `ret_c` read anywhere in the source pipeline.

**Passes iff all five hold at all 250 rounds.** Any failure invalidates
every downstream G9 number and is a section-19 stop condition.

### G9b -- source calibration

At each validation round `k in {25, 75, 125, 175, 225}`, per owner:

`z_m = (Jhat_{C,m}^i - Jhat_ref^i(theta_k)) / sqrt(SE_m^2 + SE_ref^2)`

* **good**: `|z_m| <= 2`
* **poor**: `|z_m| > 3`
* between: reported as-is, called neither.

Thresholds are G7's and G8's, verbatim, unchanged. Reported per owner per
validation round: bias, MAE, RMSE, source spread, `SE_m`, `SE_agg`,
`SE_ref`, 95% CIs, pairwise source differences, and the aggregate's own
`z_agg = (Jhat_agg^i - Jhat_ref^i)/sqrt(SE_agg^2 + SE_ref^2)`.

**A validation round passes iff `>= 5` of 6 owners have all three sources
good. G9b passes iff `>= 4` of the 5 validation rounds pass.** The 5-of-6
owner bar is G7d's and G8a's verbatim. The 4-of-5 round bar is set here,
before the data, and is *looser* than G8a's all-5 for a stated reason:
G8's own frozen-policy control failed 2 of 5 anchors at the all-5 bar
purely on finite-sample noise (90 z-scores at a 2-sigma threshold produce
several |z| > 2 by construction), so demanding all-5 would be demanding
better-than-chance behaviour from a correctly calibrated estimator. The
bar is stated now so that it cannot be read as having been relaxed later.

Pointwise equality is **not** required and its absence is not a failure:
`Jhat_{C,m}^i != J_C^i` on every round by construction at `R_m = 30`. The
question is whether the estimator is centred, which is what `z` measures.

### G9c -- training health

Over the 250 rounds of the seed-0 run:

* **i.** no NaN or Inf in any logged field -- source values, aggregate,
  residual, lambda, PPO losses, KL, entropy. Hard requirement.
* **ii.** task return improves: mean oracle `episodic_task_return` over
  the last 50 rounds exceeds the mean over the first 50 rounds.
* **iii.** PPO remains stable: the `approx_kl` median and 95th percentile
  are each within 3x of the G2 seed-0 run's values on the identical
  config (G2 measured: median 0.00230, p95 0.00516). A 3x band is used
  rather than an absolute threshold because the reference is the existing
  repaired learner, not a number chosen by G9.
* **iv.** no lambda saturation: the fraction of (round, agent) cells with
  `lambda >= lambda_max - 1e-6` is `< 0.05`.
* **v.** meaningful policy updates: policy entropy stays finite and
  strictly above `-20` (a diagonal-Gaussian collapse guard), and the mean
  per-round relative policy-parameter change is `> 1e-8`.

**Passes iff all five hold.** (i) and (iii) failing are section-19 stop
conditions.

### G9d -- constraint feedback and dual wiring

* **i.** *Exact wiring.* On every (round, agent) cell where the dual
  update did not clip, `lambda_{k+1,i} - (W lambda_k)_i == eta * residual_i`
  to within 1e-9. This is an identity if the plumbing is correct and a
  detectable defect if it is not.
* **ii.** *Sign.* `residual_i > 0` implies lambda rises relative to its
  mixed value; `residual_i < 0` implies it falls. Implied by (i);
  reported separately so a reader does not have to derive it.
* **iii.** *Response, not saturation.* `lambda` is neither pinned at 0 for
  the whole run nor pinned at `lambda_max`: at least 10% of (round, agent)
  cells have `lambda > 0`, and G9c-iv bounds the top.
* **iv.** *No source-noise-driven oscillation.* The round-to-round sign of
  `Delta lambda` must not alternate at a rate consistent with pure noise
  for the whole run: the fraction of sign flips in `Delta lambda`, pooled
  over agents, is reported, and compared against 0.5 (the pure-noise
  value) and against G2's own figure. Reported and compared;
  **descriptive, not gated**, because no threshold for it can be
  justified in advance.
* **v.** *True cost responds.* The oracle's `true_cost_return` and
  `lambda` are cross-plotted and their lagged relationship reported.
  **Descriptive, not gated**: task section 9 forbids requiring a
  particular Pearson correlation without justification, and G9 has none.

**Passes iff (i), (ii) and (iii) hold.** (i) failing is a section-19 stop
condition (the dual timing would have silently changed the algorithm).

### G9e -- clean false-safe rate (the central question)

Two definitions, both reported, one gated.

**FS-oracle**, over all 250 rounds x 6 owners (1500 cells):

`I = 1[ Jhat_mech^i <= d AND J_true^i > d ]`

where `J_true^i` is the withheld oracle's single-trajectory
`true_cost_return`. This is byte-for-byte the definition
`scripts/analyze_g2.py` uses, so the G9-vs-G2 comparison is of one
quantity, not two. Its known imperfections are stated here rather than
discovered later: the oracle rollout is drawn under `theta_{k+1}` while
the mechanism estimates `theta_k`, and a single trajectory has
per-trajectory sd ~6 against a mechanism SE of ~0.6, so the *event*
`J_true > d` is itself noisy. Both defects are present identically in the
G0/G1/G2 numbers this is compared against, which is exactly why the
comparison is legitimate even though neither number is pure.

Primary reported statistic, matching `analyze_g2.py::Seed.fsr`:
the conditional miss rate `P(Jhat_mech <= d | J_true > d)`. Also reported:
the joint rate `P(Jhat_mech <= d AND J_true > d)`, per owner, and split
early / middle / late (rounds 1-83 / 84-166 / 167-250), with worst and
best owner named.

**Already-committed baselines (read before G9 began, from artifacts that
predate it):**

| run | conditional FSR | 95% Wilson CI | mech bias |
|---|---:|---|---:|
| G0 seed 0 | 0.4777 | [0.4439, 0.5117] | -6.10 |
| G1 seed 0 | 0.3810 | [0.3508, 0.4122] | -5.85 |
| G2 seed 0 | 0.3819 | [0.3530, 0.4117] | -7.10 |

**G9e passes iff the G9 seed-0 conditional FSR's 95% Wilson upper bound
lies strictly below 0.3530**, the G2 seed-0 Wilson lower bound. This is a
pre-declared, non-overlapping-interval test against an already-committed
number; it cannot be tuned by anything G9 does.

**FS-reference**, at the 5 validation rounds only, replacing the noisy
single-trajectory oracle with the independent `R_ref = 120` reference
under the same pinned `theta_k`:

`I = 1[ Jhat_agg^i <= d AND Jhat_ref^i > d ]`

Reported per owner per validation round with `Jhat_ref`'s own CI.
**Descriptive, not gated** -- 30 cells is too few to gate on -- but it is
the cleanest measurement G9 produces, because both sides refer to the
same policy and the reference is 4x more precise than one source.

**Pre-declared interpretation guard.** A calibrated estimator does *not*
drive the false-safe rate to zero. Where the policy's true `J_C^i` sits
within a couple of standard errors of `d`, a centred estimator reports
"within budget" close to half the time, and that is correct behaviour, not
a pathology. The false-safe rate is therefore additionally stratified by
the margin `J_true^i - d` into bands (`<1`, `1-3`, `3-6`, `>6`), and a
high rate confined to the `<1` band is to be read as calibration working,
not failing. This is written down now so it cannot be deployed later as
an excuse.

### G9f -- aggregate precision matches the sampling law

G8 predicts `SE_agg = SE_m/sqrt(3)`. G9 measures whether the three
sources actually behave as three independent draws inside the live loop.

Per round `k`, per owner `i`:

* `s2_between = (1/(M-1)) sum_m (Jhat_{C,m}^i - Jhat_agg^i)^2` (2 df)
* `s2_within` = pooled per-trajectory sample variance over the round's
  3 x 30 trajectories (87 df); `expected = s2_within / R_m`.

**Statistic**: `ratio = mean_over_(k,i)(s2_between) / mean_over_(k,i)(expected)`
-- a ratio of means, not a mean of ratios, because the latter has a heavy
right tail at 2 df. Expectation is 1 if the three sources are independent
draws under a common pinned policy.

**G9f passes iff `ratio` lies in [0.8, 1.25]**, with a bootstrap 95% CI
reported alongside. A ratio materially above 1.25 means the sources are
*more* spread than the sampling law allows and triggers the section-10
investigation list (accidental source dependence, environmental
correlation, shared RNG, policy changing during collection, implementation
error). A ratio materially below 0.8 means they are *less* spread than
independent draws, which would indicate shared randomness -- equally a
defect. The band is two-sided for that reason.

### G9g -- RNG independence in the live loop (asserted in code)

* **i.** the `M = 3` replica streams are spawned from one
  `SeedSequence` via `.spawn(M)` and are therefore independent by
  construction; the spawn keys are logged in the run metadata.
* **ii.** across the whole run, the multiset of all `250 x 3 x 30 = 22,500`
  source `env_seed` values contains **no duplicate**, and likewise for the
  `torch_seed` values -- asserted at the end of every round and at run
  end, not inspected afterwards.
* **iii.** no source seed collides with any value drawn from the learner's
  `env_rng` (the PPO rollout stream) or the orchestrator's `eval` stream.
* **iv.** each worker process holds its own environment instance and its
  own torch RNG, re-seeded per trajectory from the supplied `torch_seed`;
  no environment object and no RNG object is shared between replicas.
* **v.** all source seeds are recorded to disk (`source_seeds.jsonl`) so
  any source value in the run can be regenerated independently.

**Passes iff all five hold.** Failure is a section-19 stop condition.

### G9h -- the pinned policy really is pinned (asserted in code)

* **i.** before source collection, the main process computes
  `H_k` = SHA-256 over every agent's policy `state_dict` tensors **and**
  observation-normalization statistics (mean, var, count), in a fixed key
  order, at full float precision.
* **ii.** every worker recomputes the same checksum on the parameters it
  actually loaded and returns it; the main process asserts
  `H_worker == H_k` for all three replicas, every round.
* **iii.** after source collection and before the PPO update, the main
  process recomputes `H_k` and asserts it is unchanged -- i.e. collection
  itself mutated nothing.
* **iv.** `H_k` is logged every round, so `theta_source_1 = theta_source_2
  = theta_source_3 = theta_k` is a checkable claim in the artifact, not an
  assertion in prose.
* **v.** no `torch.optim` step, no `.backward()`, and no
  `update_normalization_stats` call occurs inside the collection path.

**Passes iff all five hold.** Failure is a section-19 stop condition.

### G9i -- oracle isolation survives integration

* **i.** `tests/isolation/test_oracle_isolation.py` passes after
  integration. Its AST check already covers every module under
  `src/safelie/training/`, so a source-collection module placed there is
  checked automatically; G9 additionally extends `LEARNER_MODULES` to
  cover `sources` explicitly.
* **ii.** the worker entry point reads only `step.reported_cost`,
  `step.obs`, `step.terminated`, `step.truncated` -- the learner-visible
  `DualCostStep` fields -- and never constructs an oracle handle.
* **iii.** the full test suite passes at the G9 implementation commit.

**Passes iff all three hold.** Failure is a section-19 stop condition.

### G9j -- computational cost is reported, not hand-waved

Environment steps (PPO / source / oracle), wall-clock seconds (PPO /
source / oracle / total), effective environment-step ratio, per-round
time, and the 250-round total are measured and reported, together with
what they imply for the paper's full factorial. **Descriptive: passes iff
the numbers are reported, fails iff the report substitutes a phrase for a
number.**

## Run protocol, fixed before the data

1. **Smoke test first** (task section 14): a short run (at most 5 rounds)
   into a scratch directory, verifying source collection, PPO
   continuation, dual update, lambda movement, checkpoint/resume, RNG
   independence, policy pinning, and worker-count invariance. **Not a
   scientific result**; its artifacts do not live under
   `results/runs_constraint_batch_g9/`. If it fails, the implementation is
   fixed and the smoke test rerun before any primary data exists.
2. **Seed 0 only** (task section 15): clean condition, `d = 25`, the
   repaired-clean-learner PPO configuration, 250 rounds, into
   `results/runs_constraint_batch_g9/`. Never merged with any previous
   run directory.
3. **Review before expanding** (task section 16). Seeds 1 and 2 are not
   launched by G9 under any circumstances -- G9's own recommendation
   section decides whether a multi-seed clean validation is warranted, and
   the reviewer launches it, not G9.
4. **No attack** (task section 20), whatever the verdict.

## Verdict rule, fixed before the data

* **PASS** -- G9a, G9b, G9c, G9d, G9e, G9f, G9g, G9h, G9i all pass.
* **CONDITIONAL PASS** -- every structural gate (G9a, G9d-i, G9g, G9h,
  G9i) passes and no section-19 stop condition fired, but at least one of
  the statistical or health gates (G9b, G9c-ii/iv/v, G9e, G9f) does not.
  The specific nonfatal issue is named.
* **FAIL** -- any structural gate fails, or any section-19 stop condition
  fires, or the integration invalidates the clean experiment.

A gate is never relaxed to obtain a verdict. If a bar turns out to have
been badly chosen, the report says so and keeps the original bar's
result.

## What G9 cannot establish, whatever the verdict

G9 does not establish Theorem 2, does not test any attack, does not test
RCE, and does not establish that the clean constrained policy is *safe* --
a calibrated estimator and a satisfied constraint are different claims,
and G9 measures the first while only observing the second. It is one
seed, one environment, one budget, one topology, one `(M, R_m)` operating
point, one 250-round horizon. It says nothing about generalisation to
other seeds; that is precisely what the recommended follow-up is for.
