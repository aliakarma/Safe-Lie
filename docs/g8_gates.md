# G8 -- does the G7 disjoint-trajectory source construction survive the NONSTATIONARY training regime? Pre-declared decision rules

**Status: DECLARED BEFORE ANY G8 DIAGNOSTIC NUMBER EXISTS.** Written and
committed while `results/g8_nonstationary_source_diagnostic/dataset/` does
not yet exist. Nothing below may be edited after a G8 diagnostic number
has been read. If a rule turns out to have been badly chosen, the correct
response is to record that and say so, not to move it.

One narrow exception, stated explicitly rather than glossed:
`scripts/g8_collect_training_checkpoints.py` (step 1 below) was launched
before this file was committed. That script produces **no G8 diagnostic
number**. It re-runs `pilot_A_clean_seed0` bitwise and writes out
intermediate policy states that the G2 campaign already produced once and
then overwrote; it asserts, round by round, that every `mc_window`,
`lambda_after`, `approx_kl` and oracle `true_cost_return` equals the value
already committed in `results/runs_constraint_mc_g2/pilot_A_clean_seed0/`,
and aborts on the first mismatch. It recovers existing state; it measures
nothing. Every rule below was fixed before a single fresh G8 rollout was
collected.

## What G8 is for

G7 (`docs/g7_gates.md`, `results/g7_estimand_source_diagnostic/`) reached
a conditional pass on the estimand question: for a **frozen** policy
`theta`, `Jhat_{C,m}^i = (1/R_m) sum_{r in B_m} G_r^i` over disjoint
trajectory batches `B_m` is a scientifically coherent source estimator of
the paper's policy-level `J_C^i(theta)` (main_iclr.tex Eq. 1), and
disjoint batches are provably (not merely empirically) uncorrelated.
G7's own "what this does not establish" section named the standing gap
verbatim: it does not establish that `J_C^i(theta_k)` stays fixed long
enough, inside the 250-round nonstationary loop, for `M` sequentially
collected batches to estimate the SAME target.

G8 closes exactly that gap and nothing else. The distinction it must keep
clean throughout:

```
source randomness   !=   policy drift
```

Two batches drawn under two different policies can be perfectly
statistically independent and still be estimating two different numbers.
A low empirical correlation between sequentially collected sources is
therefore **not** evidence that they estimate a common `J_C`, and this
report is forbidden (gate G8e-ii) from using it that way.

## What G8 does not touch

No change to: the training algorithm, the source layer
(`safelie.sources`), the dual update, PPO, GAE, RCE, the attack module,
the environment, `budget_d`, or `M` in any config. No attack runs. No RCE
runs. No final factorial. No paper edit. Every G8 rollout is a frozen-
policy `env.step` plus `policy.distribution().sample()` loop, identical in
structure to `scripts/g7_collect_estimand_dataset.py::collect_one_round`
(which is itself identical to G3's). The only NEW code is diagnostic
scripts under `scripts/g8_*`.

## Notation, fixed before any number exists

* `G_r^i = sum_t gamma^t C_{r,t}^i` -- one trajectory's realized
  discounted cost return, computed by
  `safelie.training.constraint_return.discounted_window_return` on the
  round's `reported_cost` stream, i.e.
  `AgentRollout.finalize()["mc_cost_return"]`. Identical functional to
  G3-G7's. **This equals the withheld oracle's `true_cost_return`
  exactly** in this pipeline -- verified over all 1500 (round, agent)
  pairs of `results/runs_constraint_mc_g2/pilot_A_clean_seed0/oracle.jsonl`,
  where `max |true_cost_return - episodic_reported_cost_return| = 0.0`,
  which follows from `safelie.envs.dual_cost`'s stated contract (reported
  cost is numerically identical, per step, to what the environment's
  internal true-cost stream produces). So G8's `J_C` quantities ARE
  true-cost quantities; the task's requested "true-cost change" and
  `Delta J_C` are the same object here, not two.
* `theta_k` -- the policy state that GENERATES round `k`'s rollout, i.e.
  `ExperimentRun.round_index == k`, snapshotted BEFORE `run_round()` for
  that `k`. (`theta_250` is the existing G2/G7 checkpoint.)
* `Jhat_{C,m}^i(theta)` -- source `m`'s sample mean of `G_r^i` over its
  own batch of `R_m` fresh trajectories drawn under frozen `theta`.
* `Jhat_ref^i(theta)` -- a high-`R` sample mean under frozen `theta` from
  a pool disjoint from every source batch; the best available stand-in
  for the unknown `J_C^i(theta)`, always reported with its own `SE_ref`
  and never called exact.
* `sigmahat_i(theta)` -- per-owner, PER-POLICY sample sd of `G_r^i`,
  pooled over that policy's own source-plus-reference rounds only.
  Pooling across policies would fold drift into the variance estimate and
  is not done.
* `SE_m = sigmahat_i(theta)/sqrt(R_m)`; `SE_ref = sigmahat_i(theta)/sqrt(R_ref)`.

## Step 1 -- checkpoint grid (task sections 2, 10)

`results/runs_constraint_mc_g2/pilot_A_clean_seed0/` stores ONE checkpoint
(`theta_250`), because `safelie.experiment.run_experiment_with_oracle`
writes `checkpoint.pt` to a single path every round. The intermediate
policies G8 needs do not exist anywhere on disk, in any run, for any seed
(checked: all 36 `.pt` files under `results/` are final checkpoints).
Retraining is therefore **necessary** in the task's own sense, and is done
as a *bitwise reproduction*, not a new run: `ExperimentRun.checkpoint`
captures env/attack/replica/report-head/torch-global/numpy-global RNG
state and `seed_everything(0)` fixes the start, so the identical code path
and config reproduce the identical trajectory. Verified, not assumed --
`scripts/g8_collect_training_checkpoints.py` compares all four of
`mc_window`, `lambda_after`, `approx_kl`, `true_cost_return` for all 6
agents at all 250 rounds against the committed logs (6000 comparisons,
tol 1e-9) and aborts on the first mismatch. A three-round pre-check
already returned exact equality (max abs diff 0.0) on all four fields.

Anchors, matching the task's suggested grid exactly:

| phase | anchor k | stride Delta | policies snapshotted |
|---|---|---|---|
| early | 25 | 30 | 25, 26, 27, 55 |
| mid-early | 75 | 30 | 75, 76, 77, 105 |
| middle | 125 | 30 | 125, 126, 127, 155 |
| mid-late | 175 | 30 | 175, 176, 177, 205 |
| late | 225 | **24** | 225, 226, 227, 249 |

`Delta = 24` at the late anchor because `225 + 30 = 255` exceeds the
250-round run; 249 is the last policy that exists. This is the "closest
available saved checkpoint, documented exactly" case the task allows, and
every stride-Delta number for `k=225` is labelled `Delta=24` in the
report, never silently reported as 30.

## Step 2 -- pools (all fresh, all disjoint, task sections 3, 4, 5)

Per anchor, four policies are rolled out under total freeze. Per policy,
two DISJOINT pools:

| policy | source pool R | reference pool R_ref | role |
|---|---|---|---|
| theta_k | 90 | 120 | FROZEN M=3 (3 x 30 blocks); anchor reference |
| theta_k+1 | 30 | 120 | SEQ-1 source 2 |
| theta_k+2 | 30 | 120 | SEQ-1 source 3 |
| theta_k+Delta | 30 | 120 | SEQ-Delta source 2 |

660 rounds per anchor, 3300 total, 2000 steps each. Every pool draws its
round seeds from its own `np.random.default_rng`, with a seed family
(`970000 + 1000*anchor_idx + 10*policy_idx + pool_idx`) disjoint from
every seed used anywhere in G0-G7 (G3-G6: 777; G7: 8801/8802/8803;
training: 0 and derived offsets). Pairwise round-seed disjointness across
all 40 pools is **asserted in code** (G8g), not inspected afterwards. The
torch global RNG is re-seeded per pool from the same family, so two pools
under the same restored policy share neither environment randomness nor
action-sampling randomness.

`R_m = 30`, `M = 3`, total source budget 90 -- **identical to G7's S2-M3
operating point, deliberately unchanged**. Task section 13: do not expand
to M=5; `M >= 2f+1 = 3` at `f=1` is what the primary proof-of-concept
needs. Holding `R_m` at G7's value also keeps `SE_m` at G7's value, which
matters because a smaller `R_m` would inflate `SE_m` and make every drift
ratio below look *better* for free. The bar is not allowed to move in the
favourable direction.

## Step 3 -- the three conditions (task sections 4, 5, 6, 8)

* **FROZEN** -- `Jhat_{C,1..3}^i(theta_k)`, the three disjoint 30-round
  blocks of theta_k's 90-round source pool (`np.array_split`, G7's
  convention). All three under exactly the same theta_k. This is G7's
  S2-M3, re-run at five points of a training trajectory instead of one.
* **SEQ-1** -- source 1 = FROZEN block 0 (theta_k), source 2 =
  theta_k+1's 30-round pool, source 3 = theta_k+2's. Exactly the task's
  section-5 prescription: one REAL training update (the actual PPO plus
  dual update from the reproduced trajectory, not a simulated one)
  between consecutive source batches. Source 1 is shared with FROZEN by
  construction and this is stated wherever SEQ-1 is reported -- it IS the
  same batch under the same policy in both conditions, so sharing it
  isolates the drift as the only difference between the two conditions.
  **SEQ-1 is a lower bound on realistic sequential drift**: it assumes a
  whole 30-trajectory batch can be gathered inside one training round.
* **SEQ-Delta** -- source 1 = FROZEN block 0 (theta_k), source 2 =
  theta_k+Delta's 30-round pool. The realistic schedule for THIS training
  loop, which collects one rollout per round: a 30-trajectory source batch
  consumes 30 training rounds, so source 2 is drawn a full Delta updates
  after source 1. A third source at that stride would sit at
  theta_k+2*Delta, which does not exist inside the 250-round run for every
  anchor (`225 + 48 = 273`). Source 3's drift under this schedule is
  therefore reported as an **extrapolation** from the measured per-round
  rate, labelled as such in every table, never as a measurement.

None of these feed the dual update, the aggregator, the attack, or any
policy update. They are read-only rollouts under restored, frozen
policies.

## Step 4 -- policy-drift scale D_theta (task section 7)

Reported per anchor, per stride, for stride in {1, 2, Delta}:

1. **KL divergence** (preferred, per the task). Exact closed-form KL
   between diagonal-Gaussian pre-tanh policies, KL(pi_theta_k(.|s) ||
   pi_theta'(.|s)), averaged over a FIXED evaluation state set `S_k` (all
   2000 observations of theta_k's source-pool round 0, per agent, saved at
   collection time) and over the 6 agents. The tanh squash is a fixed
   deterministic bijection applied identically by both policies at the
   same `s`, so its log-Jacobian cancels exactly in the KL and the
   pre-tanh Gaussian KL IS the KL of the squashed policies -- stated here
   as the reason the closed form is legitimate, not assumed silently.
   Reported alongside PPO's own `approx_kl` from the reproduced trajectory
   (which measures the same theta_k to theta_k+1 step from inside the
   optimizer), as a cross-check that two independent measurements of the
   same step agree in order of magnitude.
2. **Parameter distance**: `||theta' - theta_k||_2 / ||theta_k||_2` over
   policy-network parameters only (critics excluded -- they do not define
   the trajectory distribution).
3. **Task-return change**: the difference of `mc_window_task_return`
   reference means between theta' and theta_k, from the same reference
   pools.
4. **Cost-return change**: `Delta J_C^i(Delta) = Jhat_ref^i(theta_k+Delta)
   - Jhat_ref^i(theta_k)`, with `SE = sqrt(SE_ref(theta_k+Delta)^2 +
   SE_ref(theta_k)^2)` and a 95% CI. Because reported cost equals true
   cost exactly here, this IS the true-cost change.

## Step 5 -- calibration bar (task section 11), inherited unchanged from G7

`z_m = (Jhat_{C,m}^i - Jhat_ref^i(theta_k)) / sqrt(SE_m^2 + SE_ref(theta_k)^2)`.

Note the denominator's reference term is **always the ANCHOR's** theta_k
reference, in every condition, because the question G8 asks is whether
source `m` estimates `J_C^i(theta_k)` -- the quantity the dual update at
round `k` is about. Scoring a drifted source against its own policy's
reference would answer a different, easier question.

* **good**: `|z_m| <= 2`
* **poor**: `|z_m| > 3`
* between: reported as-is, called neither (G3-G7 convention)

Thresholds are G7's, verbatim, chosen there before any G7 number existed
and re-used here without adjustment. Per owner, per anchor, per condition.
`|bias_m|` as a fraction of `budget_d = 25.0` is reported for context only
and gates nothing.

Also reported per owner/anchor/condition: bias, MAE, RMSE, SE, 95% CI,
pairwise source differences, and the **false-safe probability** relative
to `d = 25.0` -- `P(Jhat_{C,m}^i <= d)` under the source's own sampling
law `Normal(Jhat_ref^i(theta_k) + drift_m, SE_m^2)`, i.e. the probability
that a source of this design reports the owner as within budget when the
anchor's best-estimate `J_C` is not. Reported for FROZEN (drift 0) and for
each sequential slot (drift = the measured `Delta J_C`), so the report
shows what drift does to the safety-relevant tail, not only to the mean.

## Step 6 -- drift-to-sampling-error ratio (task section 9), bands declared before analysis

```
R_drift(anchor, owner, stride) = |Delta J_C^i(stride)| / SE_m ,
        SE_m = sigmahat_i(theta_k)/sqrt(30)
```

Bands, fixed now:

* `R_drift < 0.5` -- **negligible**: drift is small compared with source
  sampling noise.
* `0.5 <= R_drift < 2.0` -- **material**: drift measurably affects source
  comparability.
* `R_drift >= 2.0` -- **dominant**: sequentially collected sources are
  estimating substantially different policies.

Secondary, non-gating: the same ratio against the **aggregate's** standard
error `SE_agg = SE_m/sqrt(M) = SE_m/sqrt(3)`, reported separately because
a systematic drift that is tolerable for one source is `sqrt(3)x` larger
relative to the mean of three. This is reported, not gated, so that the
pre-declared bar cannot be quietly tightened after seeing the numbers.

## Step 7 -- the central criterion (task section 16), fixed before analysis

> Can three source estimates reasonably be treated as estimates of the
> same `J_C^i(theta_k)` at the time the dual update is made?

Operationalised as: **yes for a given (anchor, owner, schedule) iff**

```
max_m |Delta J_C^i(s_m)|  <=  z * SE_m ,   z = 2 ,  SE_m = sigmahat_i(theta_k)/sqrt(30)
```

where `s_m` is the number of policy updates between source 1's collection
and source `m`'s (`s = (0,1,2)` for SEQ-1; `s = (0, Delta, 2*Delta)` for
SEQ-Delta, the last extrapolated).

Why `z = 2`, argued before the data: a source whose systematic offset from
the anchor's `J_C` is inside its own two-standard-error interval is, at
the deployed batch size `R_m = 30`, statistically indistinguishable from
an unbiased estimator of `J_C^i(theta_k)` -- which is precisely what
Assumption 1(ii) asks of an honest source, expressed at the batch size
actually used. `z = 1` would demand drift smaller than the estimator can
resolve at all; `z = 3` would admit drift a single source could itself
detect. `z = 2` also matches the calibration bar's own `|z_m| <= 2`, so
one number, not two, governs "this source is consistent with the anchor."

## Step 8 -- schedule / batch-size analysis, derivation fixed before the data

The training loop collects ONE rollout per round and applies one PPO plus
dual update per round (`safelie.training.loop.run_round`,
`safelie.experiment.run_experiment_with_oracle`). Under a **sequential**
schedule where each source's `R_m` trajectories consume `R_m` training
rounds, source `m`'s batch spans rounds `[k+(m-1)R_m, k+m R_m)`, so its
target is centred `(m - 1/2) R_m` updates after theta_k. With a locally
linear drift rate `rho_i = dJ_C^i/dround` (estimated per anchor per owner
as `Delta J_C^i(Delta)/Delta`, the best-conditioned available estimate,
cross-checked against `Delta J_C^i(2)/2`):

```
R_drift(m) = rho_i (m - 1/2) R_m / (sigmahat_i / sqrt(R_m))
           = rho_i (m - 1/2) R_m^(3/2) / sigmahat_i
```

which GROWS with `R_m` -- a larger batch buys `sqrt(R_m)` precision but
pays `R_m` in drift. Setting the worst source (`m = M`) equal to the
`z = 2` tolerance gives the **maximum admissible batch size under
sequential collection**:

```
R_m_max = ( 2 sigmahat_i / ( rho_i (M - 1/2) ) )^(2/3)
```

reported per anchor per owner for `M = 3`. Under a **parallel** schedule
(`M` independent simulators/replicas stepping the same frozen theta_k) the
drift term is identically zero and `R_m` is limited only by compute. Both
are reported. This derivation is written down here, before the data, so
that the recommended design in the report is a consequence of a
pre-declared formula rather than a curve fitted to the observed numbers.

## Step 9 -- source diversity (task section 12)

Kept in two strictly separated halves, as the task demands:

* **Sampling independence** -- for FROZEN and for each sequential
  condition: theoretical pairwise covariance is **exactly 0** by the
  disjoint-draw construction (G7's proof carries over unchanged: batches
  are functions of disjoint, independently seeded rollouts, and under the
  sequential conditions they are additionally functions of different
  policies' rollouts). Reported alongside the empirical 3-point
  covariance/correlation, which is explicitly labelled unstable at n=3 and
  is never the evidence for anything. Effective source count `M_eff = M =
  3` by construction.
* **Target drift** -- reported separately, as `Delta J_C`, `R_drift` and
  the Step-7 criterion. The report must state, at the point where the
  sequential correlation is shown, that a correlation of zero under drift
  does NOT mean the sources estimate the same `J_C`. (Gate G8e-ii.)

## Gates (pre-declared)

* **G8a -- frozen calibration.** At every anchor `k in {25,75,125,175,225}`,
  the FROZEN M=3 disjoint sources are calibrated: per owner, all 3 sources
  `|z_m| <= 2`. Anchor passes iff `>= 5` of 6 owners pass. G8a passes iff
  all five anchors pass. The 5-of-6 bar is G7d's, verbatim, not re-tuned.
* **G8b -- owner coverage.** Every table in report sections C-J carries a
  row for each of the 6 owners; no aggregate-only table anywhere. Owners
  are never pooled in a way that could hide a weak individual agent.
  Structural; asserted in code over the emitted report object.
* **G8c-1 -- stride-1 drift tolerance.** At every anchor, the SEQ-1
  schedule satisfies the Step-7 criterion (`max_m |Delta J_C^i(s_m)| <=
  2 SE_m`, `s = (0,1,2)`) for `>= 5` of 6 owners. G8c-1 passes iff all
  five anchors pass.
* **G8c-Delta -- realistic-stride drift tolerance.** The same criterion at
  `s = (0, Delta, 2*Delta)`, the one-rollout-per-round schedule, with the
  `2*Delta` term extrapolated from the measured rate. Gated separately
  from G8c-1 because the two answer different deployment questions and
  must not be allowed to average each other out.
* **G8d -- temporal robustness.** G8a and G8c-1 both hold at the EARLY
  anchor (`k = 25`) as well as at the late anchor (`k = 225`). A
  construction that works only at the converged checkpoint fails G8d, even
  if the pooled-over-anchors numbers look acceptable.
* **G8e -- source diversity.** (i) Round-seed disjointness holds across
  every pair of the 40 pools, asserted in code; the theoretical pairwise
  source covariance is exactly 0 for every owner and condition. (ii) The
  report separates sampling independence from target drift and nowhere
  uses a low correlation as evidence of a common estimand. Both required.
* **G8f -- practical cost.** Environment steps, wall-clock seconds, number
  of training rounds consumed, and induced policy drift are quantified
  explicitly for both the sequential and the parallel collection schedule,
  at `M = 3`, for `R_m in {1, 5, 10, 30}` plus the derived `R_m_max`.
  Descriptive: passes iff the numbers are reported, fails iff the report
  hand-waves the cost.
* **G8g -- integrity / no-leakage.** Asserted in code, not by inspection:
  (i) every `Jhat` is a plain sample mean of `mc_cost_return` computed by
  `discounted_window_return` on `reported_cost` -- never GAE, `ret_c`, a
  critic forward pass, or a fitted head; (ii) no PPO, GAE, dual-update,
  attack, RCE, or head-refit code path executes during G8 collection;
  (iii) all pool round-seed sets are pairwise disjoint; (iv) the
  checkpoint reproduction is bitwise-identical to the committed G2 logs.
  Failure invalidates every downstream G8 number.

Raw correlation is not the sole gate for anything (task section 15).

## Pre-registered expectations (so the reading of the result cannot drift)

Stated now, from the ALREADY-PUBLIC training log
(`results/runs_constraint_mc_g2/pilot_A_clean_seed0/rounds.jsonl`, whose
per-round single-trajectory `mc_window` values were used only to SIZE the
pools, never to set a threshold): a 21-round local linear fit gives
`|dJ_C/dround|` of order 1.35 at `k = 25` and 0.12-0.27 at
`k in {75,125,175,225}`, against a frozen-policy `sigmahat ~ 5.9-6.3`
(G7). The pre-registered expectation is therefore **Outcome B** (early
drift material, late drift negligible) for the SEQ-Delta schedule and
**Outcome A/D** for SEQ-1. If the data contradicts this, the data wins and
the contradiction is reported.

## What G8 cannot establish, whatever the verdict (task section 17)

G8 does not establish Theorem 2. A pass says only that the disjoint-batch
source construction is compatible with the nonstationary training regime
**at the tested temporal scale, on this one trajectory
(`pilot_A_clean_seed0`), at this one `M = 3, R_m = 30` operating point**.
Theorem 2 additionally requires the honest-source assumption, the stated
concentration conditions, the attack model, `M >= 2f+1`, bounded
corruption, and the remaining assumptions of main_iclr.tex -- none of
which G8 touches. G8 also makes no claim about: attack stealth, RCE
behaviour, the numerical value of `epsilon(M,f,alpha)`, generalisation to
other seeds, other environments, other `d`, or other training horizons. It
does not authorise wiring the construction into
`safelie.training.loop.ExperimentRun`; that remains a separate decision
for the reviewer after this report.
