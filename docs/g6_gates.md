# G6 -- source-estimation ceiling diagnostic, pre-declared decision rules

**Status: DECLARED BEFORE ANY G6 NUMBER EXISTS.** Written and committed
while `results/g6_source_ceiling_diagnostic/` does not yet exist. Nothing
below may be edited after a G6 number has been read. If a rule turns out to
have been badly chosen, the correct response is to record that and say so,
not to move it.

## What G6 is for

`docs/g5_gates.md` recorded a decisive but sobering result: owner-specific
heads (`B1`) are the only semantically valid source family, bootstrap/
disjoint-data replication (`R1`/`R2`) buys real but small error-decorrelation
at a real calibration cost, no owner has three simultaneously well-calibrated
replicas, and `B1` itself is only "good" for 4/6 owners and "mixed"
(`corr` 0.47-0.49, just under the 0.5 floor) for `agent_4`/`agent_5`. Before
spending any more effort on source-replication mechanics, G6 asks the prior
question the whole `M >= 3` program has been assuming an answer to: **how
much information about `J_C^i` is actually present in the source query at
all**, and is the current ceiling caused by (H1) estimator capacity, (H2)
missing temporal context, or (H3) a fundamental stochastic/information limit
of the query itself.

No PPO update, no GAE, no dual update, no attack, no RCE, and no change to
the environment, the safety budget, or the policy occurs anywhere in this
diagnostic. G6 trains only small regression models on the **same** fixed,
disk-persisted dataset G3 already collected
(`results/g3_source_diagnostic/dataset/agent_*.npz`) -- no new rollout is
collected, and G5's replica experiments are not rerun.

## Data reuse and its one hard constraint (pre-declared)

Reuses `results/g3_source_diagnostic/dataset/agent_{0..5}.npz` verbatim --
same frozen checkpoint (`pilot_A_clean_seed0`, round 249), same 60 rollout
rounds seeded from generator seed 777, same masked MC `cost_to_go` fit rows,
same `obs[0]`/`mc_cost_return` eval query/truth pairs. `fit_obs`/`fit_target`
for rounds 0-39 are stored **in temporal order within each round**
(`scripts/g3_collect_calibration_dataset.py` appends `obs[:n_mc]` and
`mc_cost_to_go[:n_mc]` per round before concatenating once), confirmed by
`61640 / 40 == 1541` exactly (G5 already relied on this fact for its R2
round-block split; G6 relies on it for temporal-history construction).

**The one hard constraint this imposes**: `eval_query_obs`/`eval_truth`
(rounds 40-59) store ONLY the round's `t=0` observation and the round's
total `mc_cost_return` -- no intermediate per-step observations were ever
written to disk for those 20 rounds. A temporal-history feature (F2/F3) is
therefore **not constructible against the original 40/20 eval split**. G6
resolves this with two separate, explicitly-labeled protocols rather than
silently working around it (see "Protocols" below); this is a dataset
limitation to document, not to hide.

## Target definition, restated precisely (pre-declared, task section 2)

Per `src/safelie/training/constraint_return.py`: `fit_target[row for step t
in round r] = discounted_cost_to_go(costs_r, gamma=0.99)[t]`, one **global**
discount clock over the round's fixed 2000-step window (no reset at internal
episode/auto-reset boundaries -- `MaMuJoCoDualCostEnv` auto-resets mid-round
on `terminated`/`truncated` without breaking the window's discount clock).
`eval_truth[round] = discounted_cost_to_go(costs_round, gamma)[0] ==
discounted_window_return(costs_round, gamma)`, i.e. **the same functional
evaluated at `t=0`** -- fit row 0 of every fit round IS, by construction,
one more sample of exactly the quantity `eval_truth` measures. The tail 459
of 2000 steps are masked from `fit_target`/`fit_obs` because their
cost-to-go is censored by more than 1% of its own discounted mass
(`complete_target_horizon(0.99, 0.01) == 459`); `eval_truth` itself is not
masked (its own censoring is `gamma**2000 ~ 1.9e-9`, nil). The target is
**not** claimed deterministic conditional on `obs`: it is a Monte-Carlo
sample of a stochastic sum over a stochastic policy's 2000-step rollout,
computed with zero critic in the path (this is what G0/G1 already validated
as unbiased). Whether it is *well-identified* by `obs` (H1/H2 territory) or
carries genuine residual stochasticity `obs` cannot resolve (H3) is exactly
what sections G6-noise below measure, not assumed either way here.

## Protocols (pre-declared)

### P1 -- "t0-eval": direct G3/G4/G5 comparison, F1 only

Fit on ALL of `fit_obs`/`fit_target` (rounds 0-39, every masked row, pooled)
exactly as G3/G5's `B1`; evaluate at `eval_query_obs`/`eval_truth` (rounds
40-59, `t=0` only). This is the literal production query and is the only
protocol usable for a byte-comparable check against G5's `B1_control`
numbers (task section M). It supports **F1 only** -- P1 cannot support F2/F3
per the hard constraint above.

### P2 -- "round-CV, general-t": the protocol that makes F2/F3 possible

5-fold cross-validation over the 40 fit ROUNDS ONLY (rounds 40-59 are never
touched by P2 -- P2 does not need them and this keeps P1 and P2 fully
independent evaluations), folds built via `numpy.array_split(arange(40), 5)`
(8 rounds/fold, contiguous, deterministic, matching G5's own
`round_blocks` convention). For fold `f`: the other 32 rounds are the
training rounds, fold `f`'s 8 rounds are held out and used ONLY for scoring
-- no held-out round's row is ever used to fit anything in that fold,
including feature normalization statistics (`RunningMeanStd` is fit on
training-fold rows only).

A single history length `L` defines the valid query-time population for a
given comparison: query index `t` must satisfy `t >= L - 1` within its
1541-row round so a full `L`-step causal window exists. **The same valid-`t`
population (same `L`) is used for F1, F2, and F3 within one comparison**,
even though F1's input ignores everything but `o_t` -- this holds the
evaluation population fixed across feature variants so a difference in
metrics is attributable to feature content, not to a different set of
scored timesteps. `L = 16` is the primary window for the core Comparisons
A/B/C (section "Core comparisons" below); `L in {4, 8}` are run for F2 only,
at `C3` only, as a secondary check that the qualitative conclusion is not an
artifact of the specific `L` chosen (task section 4, "small set of
predeclared history lengths").

Training rows: every valid-`t` row of the 32 training rounds (up to
`32 * (1541 - L + 1)` rows). Evaluation rows: **every 20th valid-`t` row**
of the 8 held-out rounds (stride pre-declared to bound, not eliminate, the
within-round autocorrelation that `G_t = c_t + gamma * G_{t+1}` induces
between adjacent targets -- consecutive-`t` rows are not independent draws,
and dense scoring would silently inflate apparent precision without
inflating actual information; the stride is a partial mitigation, not a
claim of independence, and section L of the report states this caveat
verbatim). Metrics are computed per fold, then reported per fold, pooled
across folds (concatenating all 5 folds' scored predictions), and per
owner within each -- never pooled-only.

## Feature variants (pre-declared, task section 4)

* **F1**: `x_t = o_t`, the current 63-dim normalized observation. Reproduces
  `B1` under P1.
* **F2**: `x_t = [o_{t-L+1}, ..., o_t]` flattened, `63*L`-dim, `L in {4, 8,
  16}` (16 primary). Constructed only from a round's own row block (a
  window is never allowed to span two different rounds -- rounds are
  independent resets under different seeds and concatenating across that
  boundary would manufacture a spurious "history" from unrelated
  trajectories). Internal auto-resets *within* a round are NOT treated as
  window boundaries, consistent with `discounted_cost_to_go`'s own
  single-global-clock convention for the same window.
* **F3**: causal, non-future diagnostic features computed from the same
  `L=16` window ending at `t`, deliberately not a production architecture:
  `[o_t (63); mean_{s in window}(o_s) (63); std_{s in window}(o_s) (63);
  o_t - o_{t-L+1} (63, crude window-displacement proxy); mean(c_s) (1);
  std(c_s) (1); max(c_s) (1); linear trend of c_s over the window (1)]`,
  256-dim total. Per-step reported cost `c_s` inside the window is
  **reconstructed**, not re-collected, via
  `c_s = fit_target[s] - gamma * fit_target[s+1]` for `s, s+1` both inside
  the unmasked prefix (exact by the `discounted_cost_to_go` recursion) --
  no new rollout, no future information (every `c_s` used for query time
  `t` satisfies `s <= t`). Labeled explicitly throughout the report as an
  information-ceiling probe, not a candidate production feature set.

## Capacity variants (pre-declared, task section 5)

All three share Adam, Xavier-uniform weight init / zero bias init,
full-batch-bootstrap sampling, `RunningMeanStd` normalization of input and
target fit on that fold/protocol's own training rows only (never on held
rows), denormalized before any metric.

* **C1**: `Linear(D,16) -> Tanh -> Linear(16,1)`, 2500 steps, lr=1e-3, batch
  `min(2048, n)`. Byte-identical recipe to G3/G4/G5's `B1` head when `D=63`
  (F1). This is the current production architecture.
* **C2**: `Linear(D,32) -> Tanh -> Linear(32,1)`, 2500 steps, lr=1e-3, batch
  `min(2048, n)`. Hidden width doubled, nothing else changed -- isolates a
  pure width increase from a training-budget increase.
* **C3**: `Linear(D,128) -> Tanh -> Linear(128,64) -> Tanh -> Linear(64,1)`,
  8000 steps, lr=1e-3, batch `min(4096, n)`. Substantially wider and one
  layer deeper, more than 3x the gradient steps of C1/C2, chosen a priori to
  make "did not converge" an implausible explanation for a remaining gap
  (mirrors G3's own stated rationale for its 2500-step budget).

No hyperparameter is tuned against P1's eval rounds or against any P2
held-out fold; C1-C3 and F1-F3's exact specifications above are the only
configurations run.

## Non-neural baseline (pre-declared, task section 6)

Ridge regression, closed form, on the SAME normalized features per
(F,protocol,fold): `w = (X^T X + alpha*I)^-1 X^T y` on standardized `X`/`y`
(same `RunningMeanStd` convention as the neural heads, fit on training rows
only), `alpha = 1.0` in standardized units, fixed a priori and never swept
against any held data (task section 3's "do not tune hyperparameters against
the final evaluation" applies to the baseline too). Plain OLS
(`alpha = 0`) is also reported as a second reference point, solved via
`numpy.linalg.lstsq` (minimum-norm least-squares) rather than a direct
solve -- the raw observation has confirmed zero-variance dimensions in this
dataset (checked empirically before any G6 fit), which makes `X^T X`
exactly singular at `alpha = 0`; `lstsq`'s pseudo-inverse is the correct,
standard resolution and is used uniformly for both `alpha` settings rather
than only patched in for the case that broke first. If a feature matrix is
rank-deficient (F2 at `L=16`, `D=1008`, can exceed the ~1541-row per-round
training population's effective rank once pooled across only a few rounds)
`alpha=1.0` further keeps the ridge solve well-conditioned; this is
reported as a property of the comparison, not smoothed over.

## Metrics (pre-declared, task section 8, extends G3/G4/G5's set with R2)

Per (feature, capacity, protocol, owner) and pooled: `bias = mean(pred -
truth)`, `MAE = mean(|pred-truth|)`, `RMSE = sqrt(mean((pred-truth)^2))`,
`corr = Pearson(pred, truth)`, `R2 = 1 - SS_res/SS_tot` (`SS_tot` computed
against that same slice's own truth mean -- a P2 fold's `R2` is therefore
relative to that fold's held population, not a global constant), `FSR =
P(pred <= d | truth > d)`, `d = 25` (`budget_d`, unchanged). Calibration
labels reused verbatim from G3/G4/G5: **good** (`|bias| <= 5.0` and `corr >=
0.5`), **poor** (`corr < 0.2` or `|bias| > 5.0`), else **mixed**. Per-owner
tables are primary, per the task's explicit instruction and G4/G5's own
standing lesson that pooled correlation can look far better than any single
owner's.

## Irreducible-noise / ceiling analysis (pre-declared, task section 7)

* **Target variance**: reported per owner, pooled over all 60 rounds'
  `t=0` values (40 fit-round `t=0` rows, which are the same functional as
  `eval_truth`, concatenated with the 20 `eval_truth` rows) -- this is a
  60-sample estimate of `Var(J_C^i)` at the actual production query
  population, not just the 20-round `eval_truth` sample G3/G4/G5 used.
* **Nearest-neighbor target dispersion**: for a feature representation `x`
  (F1 `o_0` at the P1 `t=0` population, and separately F1/F3 `o_t`/features
  at the P2 population), standardize `x`, find each point's `k=5` nearest
  neighbors by Euclidean distance (excluding itself) within the same
  train/eval population being scored, compute the variance of `truth` among
  those 5 neighbors, and average over all points -> `local_var`. Report
  `ratio = local_var / global_var` for that same population. `ratio ~ 1`:
  neighbors in feature space carry no more target agreement than random
  points -- the feature is uninformative locally. `ratio << 1`: nearby
  points in feature space have measurably more similar targets -- local
  structure exists for a sufficiently flexible/well-fit estimator to
  exploit. This is a property of the FEATURE REPRESENTATION, computed
  identically for F1 and F3, so a comparison of the two ratios is a direct,
  model-free probe of whether F3's richer causal features expose local
  structure that raw `o_t` does not -- independent of whether any C1-C3
  network actually learns to exploit it.
* **Ceiling flag (pre-declared)**: `ratio >= 0.85` for F3 at `L=16` on the
  pooled P2 population is read as "even the richest causal feature set
  tested leaves neighbors in feature space with target dispersion nearly as
  wide as the whole population" -- evidence toward H3 for that population.
  `ratio <= 0.5` for F3 (regardless of F1's ratio) is read as "richer causal
  features do expose real local structure" -- evidence against a pure H3
  reading, redirecting the question toward whether F1/F2's own C1-C3
  networks are extracting that structure (H1) or need it handed to them
  more explicitly (H2/feature-limited).

No claim of "irreducible noise" in the formal information-theoretic sense is
made from this analysis alone; it is a descriptive dispersion measure, per
the task's explicit instruction not to over-claim it.

## Interpretation bands (pre-declared, task section 11, before any run)

All three core comparisons below use the SAME numeric bar, chosen to match
the order of magnitude G3/G5 already used for "materially" language
(G3: corr gap `>=0.3` for "both defects present"; G5: `>=0.08`/`>=0.28`
absolute correlation-reduction bars for replica diversity) rather than
invented fresh for G6:

* **No improvement**: pooled-P2 `corr` gain `< 0.10` absolute AND fewer than
  2 of 6 owners flip from "not good" to "good".
* **Minimal improvement**: pooled-P2 `corr` gain in `[0.10, 0.15)`, at most 1
  owner flips.
* **Meaningful improvement**: pooled-P2 `corr` gain `>= 0.15` absolute, OR
  `>= 2` of 6 owners flip from "not good" to "good" without any previously
  "good" owner flipping to "poor".
* **Decisive improvement**: pooled-P2 `corr` gain `>= 0.30` absolute, OR
  pooled-P2 `corr` reaches `>= 0.7` AND `>= 5` of 6 owners are "good".

`B1`/G5's own numbers (pooled `corr = 0.79`, per-owner `corr` `0.47-0.64`,
4/6 owners "good") are the empirical reference for what "already achieved"
looks like under P1; they are NOT the P2 population (different query-time
distribution, `t >= 15` general states vs. `t = 0` reset states only) and
G6's own P2 `F1,C1` number is the correct zero-shift baseline for judging
Comparisons A/B/C internally consistently. Both are reported; only the
latter is used to score the bands above.

## Core comparisons (pre-declared, task section 10)

* **Comparison A (capacity)**: P1, `F1,C1 -> F1,C2 -> F1,C3` (direct G5
  comparison) AND P2, `F1,C1 -> F1,C2 -> F1,C3` (apples-to-apples population
  with B/C below). Banded per "Interpretation bands" above, using the P2
  numbers as the scored comparison and the P1 numbers as a confirmatory
  cross-check against G5.
* **Comparison B (temporal)**: P2, `F1,C3 -> F2,C3` (`L=16` primary).
  Banded identically.
* **Comparison C (rich causal features)**: P2, `F2,C3 -> F3,C3` (`L=16`).
  Banded identically.

## Gates (pre-declared, task section "Predeclared Gates")

* **G6a (no leakage)**: for every P2 fold, assert (in code, not by
  inspection after the fact) that no held-round index appears in that
  fold's training row set, and that `RunningMeanStd` statistics used to
  score a fold were fit only on that fold's training rows. Pass/fail on the
  assertion; a failure here invalidates every P2 number in the run.
* **G6b (capacity)**: Comparison A's band, evaluated per owner AND pooled
  (a "meaningful" pooled band achieved by improving 1-2 easy owners while
  leaving the rest flat does not pass G6b -- the same anti-pooling
  discipline G4/G5 already applied).
* **G6c (temporal)**: Comparison B's band, per owner and pooled, same
  anti-pooling discipline.
* **G6d (rich features)**: Comparison C's band, per owner and pooled, same
  anti-pooling discipline.
* **G6e (nonlinearity)**: pass iff the best neural capacity (`C3`) beats
  ridge (`alpha=1.0`) on pooled MAE by `>= 10%` relative, for at least one
  of F1/F2/F3 under P2 -- otherwise the network is not demonstrated to be
  finding structure a linear map misses, and any capacity story (H1) is
  undermined regardless of the raw corr/MAE numbers.
* **G6f (ceiling flag)**: per "Ceiling flag" above -- `ratio >= 0.85` for F3
  pooled-P2 sets the flag toward H3; `ratio <= 0.5` clears it.
* **G6g (stability)**: every trained head (P1: 3 capacities x 1 fold x 6
  owners pooled-fit = one head per capacity, no per-owner refit needed
  since P1 heads are the SAME pooled `B1`-style construction as G5's B1;
  P2: 3 features(F1 at 3 L's for F2's secondary check + F1/F2/F3 main at
  L=16) x 3 capacities x 5 folds; ridge/OLS have no training instability to
  check) completes with finite loss at every logged step and finite
  predictions on its scored rows. Fail if any head produces NaN/inf
  anywhere.
* **G6h (owner coverage)**: every table in sections G-L of the report has a
  per-owner row; no aggregate-only table is permitted, mirroring G5b.

## Verdict rule (pre-declared, task section 13/O)

* **CAPACITY-LIMITED**: G6b passes at "meaningful" or better AND G6c/G6d do
  not each independently ALSO pass at "meaningful" or better (i.e. capacity
  alone accounts for materially more of the gap than temporal/feature
  richness do).
* **TEMPORAL-LIMITED**: G6c passes at "meaningful" or better and either G6b
  fails or G6c's gain is clearly larger than G6b's own gain; G6d does not
  independently add a further "meaningful" jump beyond what G6c already
  captured.
* **FEATURE-LIMITED**: G6d passes at "meaningful" or better beyond whatever
  G6b/G6c already captured -- i.e. F3's causal features move owners that
  F2's raw history could not.
* **FUNDAMENTAL-LIMITED**: G6b, G6c, and G6d all land in "no improvement" or
  "minimal improvement" AND G6f's ceiling flag is set (F3 `ratio >= 0.85`).
* **MIXED**: per-owner classifications genuinely disagree on which of the
  above applies (e.g. some owners resolve under F2/F3, `agent_4`/`agent_5`
  remain flat under every condition) -- reported as MIXED rather than
  forced into one global label, with the per-owner breakdown carrying the
  actual finding, exactly as the task's Outcome E anticipates.

If gate outcomes conflict with each other (e.g. G6b passes pooled but fails
per-owner while G6c fails both), the per-owner, anti-pooled reading is
authoritative, per every prior gate doc's own standing rule.

## Final scientific-continuation decision (pre-declared, task section 19)

Decided from the verdict above, not chosen freehand at write-up time:

* **YES** iff the verdict is CAPACITY-LIMITED or a MIXED verdict where a
  strict majority (`>= 4/6`) of owners individually resolve to "good" under
  SOME tested configuration (any F/C combination) without requiring a
  redesigned query.
* **NO** iff the verdict is FUNDAMENTAL-LIMITED, or MIXED with `<= 2/6`
  owners resolving under any tested configuration.
* **CONDITIONAL** iff the verdict is TEMPORAL-LIMITED or FEATURE-LIMITED, or
  a MIXED verdict strictly between the YES and NO bars above (3/6 owners
  resolve, or resolution requires F2/F3 rather than F1) -- continuation is
  reasonable only if the production source query is redesigned to carry
  temporal/richer information, which is a scoped architecture decision for
  separate review, not authorized by this diagnostic.

## What this diagnostic explicitly does not establish

Regardless of verdict: no claim about Assumption 1(ii) (source independence
/ honest-majority) -- G6 is entirely about whether ONE estimator's
prediction validity ceiling can be raised, never about whether multiple
estimators of it would be independent. No claim about Theorem 2. No claim
about attack stealth or RCE. No claim generalizing beyond
`pilot_A_clean_seed0`'s one frozen policy snapshot and one 60-round dataset
-- same single-checkpoint, single-dataset limitation G3/G4/G5 already
declared. A "good" P2 result does not by itself certify that the SAME
feature/capacity choice, re-embedded in the full nonstationary 250-round
training loop with fit data accruing online rather than fixed in advance,
would reproduce this diagnostic's numbers -- P2's own cross-validation
measures robustness within one frozen dataset, not across training
dynamics. F3 is explicitly a diagnostic probe, not a proposed production
feature set, per the task's instruction; a TEMPORAL-LIMITED or
FEATURE-LIMITED verdict identifies a redesign target, it does not itself
authorize implementing that redesign inside `constraint_report_heads` --
that remains a separate, scoped decision for after this report is reviewed.
