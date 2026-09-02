# G1 dual-constraint-estimator calibration: pre-declared acceptance gates

**Status: DECLARED BEFORE ANY G1 RUN EXISTS.** Written and committed
while `results/runs_constraint_mc_g1/` did not yet exist; the commit that
introduces this file precedes every G1 run artifact. Nothing below may be
edited after a G1 number has been read. If a gate turns out to have been
badly chosen, the correct response is to record that it failed and say
so, not to move it.

Unlike `docs/g0_gates.md`, whose thresholds were re-declared after the
fact and said so, every numeric bar here is justified against a
measurement that **already exists on disk at declaration time**: the
three completed G0 clean runs in `results/runs_g0/`. Each justification
below names the G0 number it is derived from. This is genuine
pre-registration and is treated as such.

## What G1 is for

G0 was a CONDITIONAL PASS. The learner learns, the dual responds, true
cost moves toward the budget, three seeds reproduce. One finding blocked
a full pass: the value the dual update compared against the budget `d`
was `ret_c[0]`, a GAE(lambda) bootstrap target. That is a valid
cost-critic regression target and a valid input to advantage estimation.
It is not a valid direct estimator of

    J_C^i(theta) = E_{pi_theta}[ sum_t gamma^t C_t^i ]

because a lambda-return keeps only about `1/(1 - gamma*lambda) = 16.8`
steps of genuine sampled-cost evidence at gamma=0.99, lambda=0.95 and
hands the rest of the roughly `1/(1-gamma) = 100` steps of discounted
mass to the cost critic's bootstrap, inheriting the critic's bias.

G1 asks exactly one question: **does replacing that estimator with a
properly computed discounted Monte-Carlo constraint return produce a dual
constraint signal that actually tracks `J_C`, without destabilising
anything G0 established?**

A G1 pass licenses only the claim that, when the attack is finally
introduced, a measured divergence can be attributed to the corrupted
safety-feedback channel rather than to a defective constraint estimator.
It licenses nothing about attack stealth, adversarial specificity, RCE,
Theorem 2, topology invariance, or a second environment.

## The single scientific change

`ExperimentConfig.constraint_estimator` switches from the pre-G1
behaviour (`gae_lambda`) to `mc_window`. Under `mc_window`:

* the `own_critic` source reports
  `safelie.training.constraint_return.discounted_window_return` over the
  round's own sampled reported cost, instead of `ret_c[0]`;
* the `ensemble_replica` / `monitor` sources fit their regression heads on
  discounted Monte-Carlo cost-to-go instead of on `ret_c`, masked to the
  leading rows whose target is complete to within 1% of its own
  discounted mass;
* the `peer_critic` sources are **unchanged** -- they remain a peer's
  cost-value network evaluated at the owner's initial observation,
  because the cost critic's own regression target is GAE(lambda) and this
  experiment does not touch GAE.

Nothing else changes. GAE keeps gamma=0.99 and lambda=0.95 for policy
optimisation. `safelie.training.ppo` is untouched. The threat model, RCE,
attack magnitude, the safety budget `d=25`, and the statistical
methodology are untouched. The attacker still corrupts the communicated
source residual after estimation and before consensus; the oracle remains
withheld.

## Run definition

Three seeds (0, 1, 2) of `configs/experiment/pilot_A_clean.yaml`,
unmodified except for `--seed` and
`--output-dir results/runs_constraint_mc_g1`. 250 rounds x 2000 steps;
ManyAgent Ant; N=6; d=25; M=7; ring topology; `attack: none`;
`defense: mean`, f=0; oracle evaluation every round. Identical in every
respect to the G0 run definition except the estimator.

`results/runs/` and `results/runs_g0/` are never written to and never
merged with the G1 directory.

## The five quantities, kept apart

Section 7 of the task is a naming discipline, not a formality. These are
five different things and none of them is "the reported cost":

| # | Quantity | Where |
| --- | --- | --- |
| 1 | True oracle MC return, privileged | `oracle.jsonl: true_cost_return` |
| 2 | Clean reported MC return, same oracle rollout, learner-visible stream | `oracle.jsonl: episodic_reported_cost_return` |
| 3 | Dual estimator before corruption (the honest source reports) | `rounds.jsonl: constraints.*.reports[].value` |
| 4 | Corrupted source report | applied in `apply_attack`; f=0 here, so identical to 3 |
| 5 | Aggregate the dual actually consumed | `rounds.jsonl: mechanism_reported_cost_return` |

Both estimators are logged from the same rollout every round, whichever
one is active, in `rounds.jsonl: constraints.*.constraint_estimators`
(`gae_lambda`, `mc_window`, `mc_episodic`) and paired against the same
oracle episode in `oracle.jsonl: agents.*.constraint_estimators`
(`bias_gae_lambda`, `bias_mc_window`, `bias_mc_episodic`).

**One caveat stated up front, because it limits every bias number below.**
The learner's estimate is computed from its own rollout under
theta_k; the oracle measures a fresh independent rollout under
theta_{k+1}, with a stochastic policy. Neither bias is pure estimator
error: both carry the same two-rollout sampling noise and the same
one-update policy offset. That is exactly why the gates below are stated
on the *difference* between the two biases and on *correlation*, both of
which cancel or are insensitive to the shared component, rather than on
either bias in isolation. G0's measured true-cost variance across rounds
is 205-242, so the shared component is large and must not be ignored.

---

## Gate G1 -- Estimator correctness

Bias per (round, agent) is `estimate - true_cost_return`. Both estimators
are read from the same rounds and paired with the same oracle episodes.

* **G1a (relative improvement, the primary clause).**
  `|mean bias_mc_window| <= 0.5 * |mean bias_gae_lambda|` over the whole
  run, in **all three** seeds.

  Rationale for 0.5: G0 measured whole-run own-critic bias of -6.91,
  -8.75, -8.53 (0.28x to 0.35x of d=25). Halving lands at roughly -3.5 to
  -4.4, which is the smallest reduction that moves the whole-run bias from
  outside G0's own already-declared calibration bar (Gate K, 0.20 x d =
  5.0) to inside it for all three seeds. A bar below 50% would let the
  estimator pass G1 while still failing the calibration standard the
  project declared at G0. It is a relative bar, so it cannot be met by the
  run merely being easier.

* **G1b (absolute).** `|mean bias_mc_window| <= 0.20 * d = 5.0` over the
  whole run, in all three seeds. G0's Gate K bar, reused verbatim so the
  two stages are commensurable.

* **G1c (phase consistency).** Split the run into five 50-round phases.
  `|mean bias_mc_window| < |mean bias_gae_lambda|` in at least **4 of 5**
  phases, in all three seeds. Guards against an improvement that is really
  one phase's luck; 4-of-5 rather than 5-of-5 because the earliest phase
  contains the critics' cold start, where both estimators are expected to
  be poor for different reasons.

* **G1d (mechanism, reported and gated, with a pre-declared exemption).**
  `|mean bias| of mechanism_reported_cost_return <= 0.20 * d = 5.0` in all
  three seeds. G0 measured -6.10, -6.00, -5.87.

  **Declared in advance:** only 3 of the 7 sources (`own_critic`,
  `monitor_1`, `monitor_2`) change under this repair. The 4 `peer_critic`
  sources are unchanged learned-value-function estimators by design, and
  they are 4/7 of a mean aggregation. If G1a, G1b and G1c pass and G1d
  fails, the cause is identified and located (the unchanged peer-critic
  sources), it does not implicate the new estimator, and the verdict is
  CONDITIONAL PASS -- not FAIL, and not a licence to change the
  aggregator, the source ensemble, or `M` in order to rescue it. That
  would be the next experiment's subject, not this one's.

**Gate G1 passes iff G1a, G1b, G1c and G1d all hold. G1a+G1b+G1c holding
with G1d failing is the pre-declared CONDITIONAL PASS branch.**

## Gate G2 -- Return-scale correctness

* **G2a (exact, within one rollout).** In the clean condition the
  environment's reported and privileged true per-step cost are identical,
  so their discounted returns over the *same* oracle rollout must agree to
  floating-point error:
  `max |episodic_reported_cost_return - true_cost_return| <= 1e-9` over
  every round, agent and seed. G0 measured exactly 0.0. This is a
  necessary correctness check on the discounting convention and is **not**
  by itself evidence that the training system is correct.

* **G2b (the estimator computes the oracle's functional).** The unit test
  `tests/unit/test_constraint_return.py::TestScaleMatchesTheOracleExactly`
  must pass: `discounted_window_return` reproduces
  `OracleEvaluator.record`'s accumulation to `rel_tol=1e-12`, including
  the requirement that the discount clock does **not** reset at an
  internal auto-reset boundary. This is the clause that makes "bias
  against the oracle" a measurement of estimator error rather than of a
  definitional mismatch.

* **G2c (cross-rollout tracking).** `corr(mc_window, true_cost_return)`
  across the 250 rounds, averaged over agents, `>= 0.70` in all three
  seeds.

  Rationale for 0.70: G0 measured `corr(own_critic GAE estimate, true)` of
  0.502 / 0.476 / 0.434 and `corr(mechanism aggregate, true)` of 0.338 /
  0.322 / 0.281. In the same G0 logs, a crude *undiscounted per-step cost
  rate* computed from the learner's own rollout already achieves 0.844 /
  0.863 / 0.818 against the identical oracle series. 0.70 therefore sits
  well above every value the broken estimator attains and well below what
  a correctly-scaled quantity drawn from the learner's own rollout is
  already demonstrated to reach. It separates "tracks the policy's actual
  constraint level" from "lags it".

**Gate G2 passes iff G2a, G2b and G2c all hold.**

## Gate G3 -- Dual correctness

* **G3a (identity).** `constraint_residual == mechanism_reported_cost_return - d`
  to within 1e-9, every round, agent and seed. Arithmetic; a failure means
  a rescaling was slipped into the estimator path.

* **G3b (sign).** Restricted to (round, agent) cells where the projection
  in `safelie.training.dual.dual_update` is not active -- i.e. the
  pre-projection value `(W lambda_k)_i + eta * residual_i` lies strictly
  inside `[0, lambda_max]` -- the sign of
  `lambda_{k+1} - (W lambda_k)_i` must equal the sign of `residual_i` in
  `>= 99%` of such cells, in all three seeds. Essentially arithmetic; 99%
  rather than 100% only tolerates floating-point ties at exactly zero.

* **G3c (responsiveness).** `corr(residual_k, lambda_{k+1} - lambda_k) > 0`
  across rounds, averaged over agents, in all three seeds. A strictly
  positive bar rather than a magnitude, because the magnitude is set by
  eta_lambda = 0.035, which is `[SPEC]` and is not under test here.

**Gate G3 passes iff G3a, G3b and G3c all hold.**

## Gate G4 -- Learning

G0's Gate L, reused verbatim, on `oracle.jsonl: episodic_task_return`.
Window: first 20 rounds against last 20 rounds.

* **G4a.** `mean(last 20) - mean(first 20) > 0` for all three seeds.
* **G4b.** Averaged across seeds, that improvement is at least **1.0x the
  pooled standard deviation of the first-20 window**.

Rationale: identical to G0's, and reusing the bar verbatim is the point --
"the estimator change did not break learning" is then measured on exactly
the bar the learner has already been shown to clear (G0 improved from
about -167 to about -46).

**Gate G4 passes iff G4a and G4b both hold.**

## Gate G5 -- Constraint response

G0's Gate C, reused verbatim, on `true_cost_return` averaged over agents.

* **G5a (the constraint is exercised).** True cost return exceeds d=25 in
  at least one round for at least one agent, in all three seeds. A run in
  which the constraint never binds tests nothing.
* **G5b (trend).** `mean(last 20% of rounds) < mean(first 20% of rounds)`,
  with the **same sign in all three seeds**.

Deliberately NOT required, for the same reason declared at G0: that true
cost end below d=25. At d=25 with `velocity_threshold=0.75` the
initial-policy measurement is about 1.66x budget and 5e5 steps is 1/20 of
the paper's scale. Whether d is met is reported as a headline number and
feeds the next-experiment decision; it is not a gate. This exemption is
declared in advance precisely so it cannot be granted later.

**Gate G5 passes iff G5a and G5b both hold.**

## Gate G6 -- Stability

* **G6a (numerics).** No NaN or Inf in any logged field, any round, any
  agent, any seed.

* **G6b (no lambda saturation).** `max lambda` over the whole run
  `< 0.9 * lambda_max = 22.5`, and **zero** (round, agent) cells at
  `lambda >= 22.5`, in all three seeds.

  Rationale: G0 measured max lambda of 11.01 / 10.67 / 12.54, i.e. 0.44x
  to 0.50x of lambda_max. The specific pathology this experiment could
  introduce is a larger-magnitude, higher-variance constraint estimate
  driving the controller into its clip, at which point the dual stops
  being a controller and the run stops testing anything. 0.9x is the
  boundary of that failure and is roughly twice the worst G0 excursion.

* **G6c (no runaway PPO).** Mean `approx_kl` over the last 20% of rounds
  `<= 0.05`, and `max approx_kl` over the whole run `<= 0.5`, in all three
  seeds.

  Rationale: G0 measured mean KL 0.0025-0.0026, last-50 mean
  0.0026-0.0029, p99 0.0076-0.0083 and a single-round max of 0.058-0.074.
  0.05 on the sustained mean is roughly 17x G0's and above G0's own
  worst single round, so it cannot fire on ordinary variation; 0.5 on the
  max is 7x G0's worst round and catches an actual trust-region blowout.

* **G6d (no entropy collapse).** Mean policy entropy over the last 20% of
  rounds `> 0.5`, in all three seeds. G0 measured last-20 entropy of
  1.88-1.97 against a first-20 value of about 3.61, with a whole-run
  minimum of 1.67. 0.5 is well below the observed floor and fires only on
  a genuinely degenerate, near-deterministic policy.

* **G6e (dual volatility tripwire, section 13).** Mean `|lambda_{k+1} -
  lambda_k|` over the whole run `<= 3x` G0's, i.e. `<= 0.45`, in all three
  seeds.

  Rationale, and the reason this is a tripwire rather than a variance
  gate: G0's mechanism estimate has cross-round variance of 48.7-57.5
  against a true-cost variance of 205-242. It is **over-smoothed by about
  a factor of four**; its low variance is lag, not precision. A correctly
  tracking estimator is therefore *expected* to have variance rising
  toward 205-242, roughly a 2x increase in standard deviation, and hence
  an expected roughly 2x increase in mean `|delta lambda|` from G0's
  0.145-0.153. That increase is the intended effect and must not be
  scored as instability. 3x leaves headroom above the expected doubling
  while still catching a genuine runaway, and it is paired with G6b, which
  is the clause that actually detects a controller destroyed by variance.

**Gate G6 passes iff G6a-G6e all hold.**

## Gate G7 -- Reproducibility

* **G7a.** All three runs reach round 249 with
  `run_metadata.json: status == "complete"`, `rounds.jsonl` and
  `oracle.jsonl` each holding exactly 250 sequential records, and a
  checkpoint present.
* **G7b.** `run_metadata.json` records `git.dirty == false` and the same
  `git.sha` for all three seeds, and
  `config_snapshot.constraint_estimator == "mc_window"` for all three.
* **G7c.** `scripts/analyze_matrix.py --runs-dir results/runs_constraint_mc_g1`
  reports no integrity problems.
* **G7d.** The Gate-G5b true-cost trend direction is the same across all
  three seeds, and the Gate-G1a improvement direction is the same across
  all three seeds.

**Gate G7 passes iff G7a-G7d all hold.**

---

## Reported in full, gating nothing

* **MC variance analysis (section 13).** `Var(mc_window)`,
  `Var(gae_lambda)` and `Var(true_cost_return)` across rounds, per seed,
  with the correlation figures from G2c. Declared interpretation, fixed in
  advance: if `Var(mc_window)` rises toward `Var(true_cost_return)` **and**
  `corr(mc_window, true)` rises, the additional variance is signal the old
  estimator was suppressing, not noise the new one introduced. Only a
  variance rise *without* a correlation rise, or a G6b/G6e failure, would
  support the opposite reading. No variance-reduced estimator is
  introduced in this experiment.
* **`mc_episodic` versus `mc_window`.** The per-episode own-clock
  estimator is logged every round as a diagnostic so the size of the
  definitional difference is measured rather than assumed. It is not the
  dual signal and gates nothing; see
  `safelie.training.constraint_return`'s docstring for the censoring bias
  that makes it the diagnostic and the window sum the signal.
* **Source independence / effective M.** Reported per seed, gates nothing,
  for the reason already declared at G0.
* **Feasibility.** Whether true cost ends below d=25.
* **Per-source behaviour.** In particular the 4 unchanged `peer_critic`
  sources, so the ensemble's remaining critic dependence stays visible.

## Verdict rule

* **PASS** -- G1 through G7 all pass.
* **CONDITIONAL PASS** -- exactly one gate fails, the failure has an
  identified cause, and that cause does not invalidate the others. The
  G1d branch described above is the one such failure anticipated in
  advance.
* **FAIL** -- two or more gates fail, or any single failure whose cause
  implicates the estimator itself (G1a, G2b or G3a failing is
  automatically FAIL, regardless of count).

On FAIL: stop. Do not run the attack study, do not relax a threshold, do
not discard a seed, do not raise the budget, do not redefine the return,
do not change the metric or the statistical criterion. Identify the
failed gate, find the root cause, make the minimum justified repair,
rerun this validation, and report what changed.

Passing G1 does **not** license the attack study on its own; it licenses
the claim that the clean dual constraint estimator tracks `J_C`, which is
the precondition the attack study needs in order for a measured
divergence to be attributable to the corrupted safety-feedback channel.
