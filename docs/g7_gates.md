# G7 -- estimand audit and multi-source test for the SAME policy-level J_C^i, pre-declared decision rules

**Status: DECLARED BEFORE ANY G7 NUMBER EXISTS.** Written and committed
while `results/g7_estimand_source_diagnostic/` does not yet exist. Nothing
below may be edited after a G7 number has been read. If a rule turns out to
have been badly chosen, the correct response is to record that and say so,
not to move it.

## What G7 is for

`docs/g3_gates.md` through `docs/g6_gates.md` all diagnosed ONE query:
`o_t -> G_t^i(o_t)`, a neural regression of a single realized trajectory's
discounted cost-to-go onto a per-step observation (`peer_critic`,
`monitor`, `ensemble_replica`, all built on
`safelie.sources.estimators.DiversifiedReplica`). G6 recorded a MIXED
verdict: capacity, temporal history, and richer causal features all failed
to close the gap for `agent_4`/`agent_5`, and G6's own ceiling diagnostic
left open whether that is a fixable estimator defect or a fundamental
limit of predicting a stochastic trajectory functional from one state.

G7 asks a prior question none of G3-G6 asked: **is `o_t -> G_t^i(o_t)` even
the right estimand for what Assumption 1(ii) and Eq. 1 require?** The
paper's object is

```
J_C^i(theta) = E_{pi_theta}[ sum_t gamma^t C_t^i ]          (main_iclr.tex Eq. 1)
```

a single scalar, the expectation over the POLICY's trajectory
distribution at a fixed theta -- not a function of state. Assumption 1
(main_iclr.tex, Sec. "Guarantees") requires "honest sources report
unbiased estimates of `J_C^i(theta_k)` with sub-Gaussian parameter
`varsigma`" -- again a statement about estimators of the scalar
`J_C^i(theta_k)`, not about a regression function of state. G3-G6 never
tested this object. They tested whether a neural network can predict
`G_t^i(o_t)`, a state-conditioned single-trajectory return, which is
closer to `V_C^i(s)` (or a noisy one-sample proxy for it) than to
`J_C^i(theta)`. G7 tests the estimand the theorem actually assumes:
whether **independent empirical means of `G_C^i(tau)` over freshly drawn
trajectories** can serve as `M` honest sources.

No PPO update, no GAE, no dual update, no attack, no RCE, and no change to
the environment, the safety budget, or the policy occurs anywhere in this
diagnostic. G7 collects NEW frozen-policy rollouts (it does not reuse
`results/g3_source_diagnostic/dataset/`, because that dataset's 60 rounds
were collected under generator seed 777, already consumed by G3-G6 as
training/eval data for neural heads, and section 3 of the task requires a
seed independent of every prior collection). No neural network of any kind
is fit anywhere in the primary G7 construction (task section 11): every
source here is a sample mean of `mc_cost_return` values, nothing more.

## Estimand notation, fixed before any number exists

* `G_r^i = sum_t gamma^t C_{r,t}^i` -- one trajectory's realized discounted
  cost return, round `r`, owner `i`. Computed by
  `safelie.training.constraint_return.discounted_window_return` on the
  round's `reported_cost` stream, identical convention to
  `AgentRollout.finalize()["mc_cost_return"]` and to what
  `_collect_source_value` already returns for `own_critic` under
  `constraint_estimator == "mc_window"` (`src/safelie/training/loop.py:245-248`).
  This is the SAME functional G3-G6's `eval_truth`/`mc_cost_return` used as
  ground truth, not a new definition.
* `J_C^i(theta)` -- the theoretical policy-level expectation, Eq. 1.
  Unknown exactly; G7 never claims to know it exactly, only to approximate
  it with a reference whose own standard error is reported (section
  "Reference" below).
* `Ĵ_{C,m}^i = (1/R_m) sum_{r in B_m} G_r^i` -- source `m`'s empirical
  estimate, a sample mean over `R_m` trajectories in that source's own
  batch `B_m`.
* `Ĵ_reference^i` -- a high-`R` sample mean from a batch DISJOINT from
  every source's batch, used as the best available stand-in for the
  unknown `J_C^i(theta)`.

## Data generation (pre-declared, task section 3)

* One frozen policy, identical checkpoint to G3-G6:
  `results/runs_constraint_mc_g2/pilot_A_clean_seed0/checkpoint.pt`
  (round 249). One config: `configs/experiment/pilot_A_clean.yaml`
  (`rollout_length=2000`, `gamma=0.99`, `budget_d=25.0`, 6 agents). Frozen
  throughout -- `ExperimentRun.restore`, zero gradient updates to
  `policy`/`value_net`/`cost_value_net`, mirroring
  `scripts/g3_collect_calibration_dataset.py`'s `collect_one_round`
  exactly (on-policy stochastic actions, tanh-squashed, no critic call in
  the collection path).
* Three independent round-seed generators, all NEW and disjoint from every
  seed used anywhere in G0-G6 (G3-G6's collection seed was 777; training
  seeds were 0 and derived offsets 1000/4000/6000/21000/22000/23000/24000
  per docs/g4_gates.md-g6_gates.md):
  - `SEED_EVAL_A = 8801` -- 90 rounds, the primary evaluation pool
    (`eval_pool_a`), used to build S1/S2/S3/S4.
  - `SEED_REFERENCE = 8802` -- 270 rounds, the reference pool
    (`reference_pool`), used ONLY to compute `Ĵ_reference^i`. Disjoint
    physical rollouts from `eval_pool_a` (asserted in code: no round-seed
    integer drawn by the reference generator equals one drawn by the
    eval-A or eval-B generator -- a collision is astronomically unlikely
    at 2^31 range over 450 draws, and the assertion converts "unlikely"
    into "checked").
  - `SEED_EVAL_B = 8803` -- 90 rounds, a second evaluation pool
    (`eval_pool_b`), used ONLY for the reproducibility check (G7f). Scored
    against the SAME `Ĵ_reference^i` as `eval_pool_a` (valid: the
    reference approximates `J_C^i(theta)`, a fixed property of the frozen
    policy, not of which seed queried it).
* 450 rounds total (90 + 270 + 90), each 2000 steps, all 6 agents
  simultaneously (`G_r^i` for all six owners comes out of the SAME
  physical joint rollout `r` -- this repository's environment is one
  shared multi-segment body, not six independent sub-environments; see
  "Practical feasibility" below for why this matters for section 14).
  Collected by `scripts/g7_collect_estimand_dataset.py`, which stores
  ONLY `mc_cost_return` per round per agent (no per-step observations, no
  regression targets) -- no neural predictor is trained anywhere in this
  diagnostic, so no per-step data needs to be retained.
* **No training-time data.** None of G0-G6's rollouts, checkpoints'
  intermediate rounds, or the 250-round training loop's own `own_critic`
  history are read by this diagnostic.

## Batch counts (pre-declared, task sections 4/9)

From `eval_pool_a`'s 90 rounds, per owner:

* **S1 (M=1)**: one batch, all 90 rounds. `Ĵ_full^i`.
* **S2 (M=3, disjoint)**: `numpy.array_split(arange(90), 3)` -> three
  30-round contiguous blocks (matches G5's `R2` block-split convention).
* **S2 (M=5, disjoint)**: `numpy.array_split(arange(90), 5)` -> five
  18-round blocks.
* **S3 (M=3, bootstrap)**: three resamples of size 30, drawn WITH
  replacement from the pooled 90-round `eval_pool_a`, via
  `np.random.default_rng(BOOTSTRAP_SEED + i)`, `BOOTSTRAP_SEED = 90003`,
  `i` = replica index 0..2.
* **S3 (M=5, bootstrap)**: five resamples of size 18, same pool, same
  generator family, `BOOTSTRAP_SEED = 90005`.
* **S4 (repeated "initialization")**: the primary G7 estimator (a sample
  mean) has no stochastic-fit component, so an "init-seed" control means
  re-deriving the SAME batch's mean under a different summation order
  (`np.random.default_rng(seed).permutation` of the batch's row order
  before summing), seeds `{1, 2, 3}`, on S2-M3's block-0 batch for every
  owner. Reported as a structural/analytic point (see G7d below), not a
  swept experiment: floating-point order changes cannot alter a sample
  mean's value by more than machine epsilon, and this is checked (not
  assumed) by asserting the three re-orderings agree with each other to
  `<1e-9` relative difference.

`R_eval = 90` is exactly divisible by both 3 and 5 (30 and 18 rows/block),
so the M=3 and M=5 conditions share the same fixed total trajectory
budget, per the task's explicit instruction (section 9) to hold the
budget fixed while varying `M`.

## Reference construction (pre-declared, task section 5)

`Ĵ_reference^i = mean(reference_pool.mc_cost_return[i])`, `R_ref = 270`,
disjoint rounds from `eval_pool_a`/`eval_pool_b`. `R_ref = 3 * R_eval`,
chosen so that `SE_ref = sigma_hat_i / sqrt(270)` is smaller than
`SE_m(R_m=30)` by a factor of `sqrt(30/270) ~ 3.0x` and smaller than
`SE_m(R_m=18)` by `sqrt(18/270) ~ 3.9x` -- both comfortably inside
"substantially smaller" without requiring an implausible reference budget.
`sigma_hat_i` (per owner, used for every closed-form variance calculation
below) is the pooled sample standard deviation of `G_r^i` over ALL 450
rounds collected (`eval_pool_a` + `reference_pool` + `eval_pool_b`) --
using the full collected sample for the variance estimate is a stability
choice (it does not leak information into any one source's OWN mean,
since variance and mean are estimated from possibly-overlapping data only
in the sense that both come from the same iid process, never from the
same realized value being read twice into one comparison). `Ĵ_reference^i`
is explicitly labeled non-exact throughout the report (task section 5): a
270-round mean, not `J_C^i(theta)` itself, with its own reported
`SE_ref`.

**Design decision, stated explicitly**: the reference pool is DISJOINT
from `eval_pool_a`, not merely the S1 pooled mean relabeled. This is a
deliberate choice over the cheaper alternative (calling `Ĵ_full^i` from S1
"the reference"), because `Ĵ_full^i` is the pool `eval_pool_a` that S2/S3
also draw from -- comparing S2/S3 sources against a reference built from
the SAME rows they were drawn from would silently shrink measured bias/
correlation-with-reference through data reuse. Section H's closed-form
covariance derivations below depend on this disjointness (they use
`Cov(batch_m, reference) = 0` as a construction fact, which would be false
under the cheaper alternative).

## Calibration bar -- pre-declared, and DIFFERENT from G3-G6's bar, with the reason stated up front

G3-G6's `|bias| <= 5.0 and corr >= 0.5` bar was designed for a neural
regression head, where "bias" can reflect a genuine, fixable estimator
defect (miscalibration) at any sample size. That bar does not fit here: a
sample mean of an unbiased quantity (`E[G_r^i] = J_C^i(theta)` by
linearity of expectation over the same trajectory distribution the
theorem itself conditions on) is unbiased **by mathematical construction
at every `R_m`** -- any observed `Ĵ_{C,m}^i - Ĵ_reference^i` is pure
finite-sample sampling noise, not a model defect, and a fixed absolute
threshold (5.0, chosen for a different estimator class in G2) would
conflate "this source has a small budget" with "this source is wrong
about something." G7's calibration bar is therefore expressed in units of
the comparison's OWN predicted sampling noise:

* `z_m = (Ĵ_{C,m}^i - Ĵ_reference^i) / sqrt(SE_m^2 + SE_ref^2)`, where
  `SE_m = sigma_hat_i / sqrt(R_m)` is source `m`'s own predicted standard
  error and `SE_ref = sigma_hat_i / sqrt(270)`.
* **Good**: `|z_m| <= 2` (consistent with a genuinely unbiased estimator
  of the same quantity at ~95% two-sided confidence, treating `sigma_hat_i`
  as known -- a standard, pre-declared approximation, not tuned after
  seeing the data).
* **Poor**: `|z_m| > 3`.
* Anything between is reported as-is, called neither, exactly as
  G3-G6's convention for their own middle band.
* Absolute bias, reported for context only (not gating): `|bias_m|` as a
  fraction of `budget_d = 25.0`.

This bar is stated here, before any G7 row is computed, precisely so it
cannot be tuned to whatever the data turns out to show.

## Error-dependence / independence protocol (pre-declared, task sections 7/10/13)

Two distinct correlation objects are computed, kept separate throughout
the report per the task's explicit warning (section 13) not to lean on an
unstable 3-5-point Pearson `r` as the primary evidence:

1. **Theoretical / closed-form**, derived from the sampling design and
   `sigma_hat_i^2`, NOT from the 3 or 5 realized numbers:
   - S2 (disjoint blocks): `Cov(Ĵ_{C,m}, Ĵ_{C,n}) = 0` for `m != n`
     EXACTLY, because disjoint round-seed batches share no random draw
     whatsoever (proof, not estimate: the two batches' `G_r^i` values are
     functions of disjoint, independently-seeded environment rollouts).
   - S3 (bootstrap): `Corr(Ĵ_{C,m}, Ĵ_{C,n}) = R_m / (R_m + N_pool)` for
     `m != n`, `N_pool = 90`, derived from the law of total covariance
     over the shared finite pool's own mean (both replicas' conditional
     mean, given the fixed 90-row pool, is the pool's own sample mean;
     only the within-pool resampling noise is independent across
     replicas). At `R_m = 30`: predicted `0.25`. At `R_m = 18`:
     predicted `~0.167`.
   - Error-vs-reference correlation `Corr(e_m, e_n)`, `e_m = Ĵ_{C,m} -
     Ĵ_reference`: for S2, `= SE_ref^2 / (SE_ref^2 + SE_m^2)` (small,
     `<=1/10` given the `R_ref=3*R_eval` design choice); for S3 bootstrap,
     `= (sigma_hat_i^2/90 + SE_ref^2) / (sigma_hat_i^2/90 + SE_m^2 +
     SE_ref^2)`, dominated by the `sigma_hat_i^2/90` pool-sharing term.
2. **Empirical**, the realized Pearson `r` and covariance matrix computed
   from the actual `M=3` or `M=5` numbers per owner/condition -- reported
   alongside (1), explicitly labeled unstable at `n=3`/`n=5` per the
   task's section 13, and NEVER used alone as evidence of independence or
   its absence. Where (1) and (2) disagree, (1) (the closed-form
   sampling-design result) is authoritative, because it is derived from
   the collection mechanism itself rather than estimated from a handful
   of points.

`effective sample/source count`: `M_eff = M` for S2 by construction (the
theoretical pairwise covariance is exactly zero, so the participation-
ratio measure G4/G5 used would return `M_eff = M` trivially here);
`M_eff` for S3 is computed via the same participation-ratio formula
(`PR = (sum lambda)^2 / sum lambda^2` of the theoretical `M x M`
covariance matrix) as a direct, closed-form counterpart to G4/G5's
empirical PR, reported for comparison.

## Sample-size / source-count tradeoff (pre-declared, task section 9)

For `M in {1, 3, 5}` at the FIXED `R_eval=90` budget: report, per owner,
`R_m`, `SE_m`, the fraction of that condition's sources individually
rated "good" per the calibration bar above, and the theoretical
cross-source correlation (0 for all `M` under S2, by construction).
**Pre-declared reading**: because disjoint-batch correlation is exactly
zero regardless of `M` (unlike G5's neural replicas, where correlation
was empirically observed to depend on the replication mechanism), this
tradeoff is expected to be a PURE precision cost (`SE_m` grows as
`1/sqrt(R_m)`, more sources means noisier individual sources) with NO
accompanying independence cost -- the opposite structure from what G5
found for neural replicas. This is stated as the pre-registered
expectation, not asserted as the outcome; section I of the report
verifies it against the actual `sigma_hat_i` and calibration numbers
rather than assuming it.

## Gates (pre-declared)

* **G7a (correct estimand, no leakage)**: assert in code (not by
  inspection) that (i) `eval_pool_a`, `reference_pool`, `eval_pool_b`
  round-seed sets are pairwise disjoint, (ii) every `Ĵ_{C,m}^i`/
  `Ĵ_reference^i` is computed via `discounted_window_return`/
  `mc_cost_return` on `reported_cost`, never via GAE/`ret_c`/a critic
  forward pass, and (iii) no PPO/GAE/dual/attack code path executes
  during collection. Pass/fail on the assertions; failure invalidates
  every downstream G7 number.
* **G7b (calibration)**: per owner, per (S2 or S3, M) condition: pass iff
  ALL `M` sources in that condition are individually "good" (`|z_m|<=2`)
  per the bar above. Reported per owner AND per condition, never
  pooled-only (mirrors G5a/G6b's per-owner discipline).
* **G7c (source diversity)**: pass iff, for every owner and every `M in
  {3,5}`, S2's theoretical pairwise correlation (`=0`) is materially
  smaller than S3's theoretical pairwise correlation at the matched `R_m`
  (`R_m/(R_m+90)`) by at least `0.10` absolute -- trivially true given
  S2's correlation is exactly zero and S3's is `0.25`/`0.167` at
  `R_m=30`/`18`, but checked explicitly per owner rather than assumed
  global. A secondary check reports whether the EMPIRICAL 3-5-point
  correlation is directionally consistent (S2 empirical `<` S3 empirical)
  without gating on it, per the section 13 instruction.
* **G7d (three-source viability)**: pass iff, for `M=3` disjoint (S2),
  G7b passes for at least 5 of 6 owners (mirrors G5a's 5-of-6 bar, chosen
  because G5's own full-data B1 control already left 2/6 owners short of
  "good", so G7d must not silently raise the bar for a construction that
  has strictly LESS data per source than G5's full-data control did for
  its neural heads) AND G7c passes.
* **G7e (owner coverage)**: every table in sections G-J of the report has
  a per-owner row; no aggregate-only table is permitted (mirrors
  G5b/G6h).
* **G7f (reproducibility)**: `eval_pool_b` (`SEED_EVAL_B=8803`) reproduces
  S2-M3's qualitative finding -- pass iff every owner's `eval_pool_b`-built
  S2-M3 sources are ALSO all "good" under the SAME `Ĵ_reference^i`, AND
  the S2-vs-S3 correlation ordering (S2 exactly 0, S3 `>0`) holds
  identically (it is a closed-form property of `R_m` and `N_pool`, not of
  which seed drew the rows, so it is expected to reproduce exactly --
  reported as a confirmation that the theoretical derivation, not merely
  one lucky draw, explains the S2 finding).

## Practical feasibility question, pre-declared framing (task section 14)

Answered descriptively in section K of the report, not gated pass/fail,
per the task's own three-way framing (YES / CONDITIONAL / NO). The
question to be answered without forcing a positive result: this
environment's 6 agents are segments of ONE shared MuJoCo body
(`configs/experiment/pilot_A_clean.yaml`: `env.name: manyagent_ant`), so
one "round" is one joint rollout that produces `G_r^i` for all six owners
AT ONCE from the SAME physical trajectory -- getting `M` independent
trajectory-based sources for ONE owner's `J_C^i` requires `M` independent
ROLLOUTS of the whole joint system (not `M` different agents), which is
straightforward in a simulator (this diagnostic's own 450 rollouts prove
it is computationally cheap here) but is a genuinely different
requirement from a live single-instance physical deployment, where only
one physical trajectory unfolds at any given decision point. Section K
must state which of the two regimes (simulated/twin re-evaluation vs.
single live trajectory vs. physical fleet) each of S1-S4 would require in
production, without amending Theorem 2 or Assumption 1(ii) to make the
answer come out YES.

## What this diagnostic explicitly does not establish

Regardless of verdict: G7 does not establish that `J_C^i(theta_k)` stays
fixed long enough, inside the full 250-round nonstationary training loop,
for `M` sequentially-collected trajectory batches to all estimate the
SAME target -- `theta_k` changes every round in the real loop, whereas G7
deliberately freezes `theta` throughout (identical standing limitation to
G3-G6). No claim about Theorem 2's numerical bound
`epsilon(M,f,alpha)` being achieved by any specific deployed `M`/`f`. No
claim about attack stealth or RCE (no attack, no RCE code path runs in
this diagnostic). No claim generalizing beyond `pilot_A_clean_seed0`'s one
frozen checkpoint. A PASS on G7b-d establishes only that `M` disjoint-
trajectory sample-mean sources of the SAME owner's `J_C^i`, collected
under one frozen policy, are simultaneously well-calibrated (by the
z-score bar above) and provably (not just empirically) uncorrelated with
each other -- it does not by itself establish that this construction is
what should be wired into `safelie.training.loop.ExperimentRun` in place
of `constraint_report_heads`, which remains a separate, scoped
implementation decision for after this report is reviewed (task section
16: do not implement it into the training loop, do not run RCE, do not
run attacks).
