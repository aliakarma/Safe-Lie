# G5 -- owner-specific replica diversity, pre-declared decision rules

**Status: DECLARED BEFORE ANY G5 NUMBER EXISTS.** Written and committed
while `results/g5_source_diagnostic/` does not yet exist. Nothing below may
be edited after a G5 number has been read. If a rule turns out to have been
badly chosen, the correct response is to record that and say so, not to
move it.

## What G5 is for

`docs/g3_gates.md` recorded FAIL (Outcome B, H2 supported): a physical
peer's own head, queried at a foreign agent's observation, is
scientifically invalid as a `J_C^i` estimator. `docs/g4_gates.md` recorded
CONDITIONAL PASS at best: a single identity-conditioned shared network
(B2) can be calibrated (Outcome A/C), but is unconditionally `M_model = 1`
(G4d) -- one shared parameterization, however many identity vectors are
fed into it -- and three independent-initialization replicas of that
shared network showed pairwise residual correlation `~0.988-0.991`
(`participation_ratio = 1.014` of a possible 3): independent init alone
does not create meaningful source diversity.

G5 asks the next, narrower question: starting from the ONE estimator
family G3/G4 already validated as calibrated in-distribution -- the
owner-specific B1 head, `h_i(o^i)`, one physical network per owner, no
peer query, no shared parameterization -- can we train **multiple**
independent replicas of that head per owner, `h_{i,1..k}`, that (a) stay
individually calibrated and (b) have estimation errors that are
*meaningfully less correlated* than G4's init-only replicas, by
introducing diversity in the replica's **training data**, not just its
initialization seed?

No PPO update, no GAE, no dual update, no attack, no RCE, and no change to
the environment, the safety budget, or the policy occurs anywhere in this
diagnostic. G5 trains only small regression heads on the **same** fixed,
disk-persisted dataset G3 already collected
(`results/g3_source_diagnostic/dataset/agent_*.npz`) -- no new rollout is
collected.

## Data reuse (pre-declared)

* Reuses `results/g3_source_diagnostic/dataset/agent_{0..5}.npz` verbatim
  -- same frozen checkpoint (`pilot_A_clean_seed0`, round 249), same 60
  rollout rounds seeded from generator seed 777, same masked MC
  `cost_to_go` fit targets, same `obs[0]`/`mc_cost_return` eval
  query/truth pairs.
* **Fit split**: rounds 0-39, exactly as G3/G4 (61,640 masked rows/agent,
  `61640 / 40 = 1541` rows/round exactly -- confirmed from `meta.json`
  before any replica is trained). Because `scripts/g3_collect_calibration_
  dataset.py` appends each round's masked rows to `fit_obs`/`fit_target`
  in round order and only concatenates once at the end, row range
  `[r*1541, (r+1)*1541)` of the saved array IS round `r`'s data. This fact
  (not a re-collection) is what makes the round-block disjoint split in
  R2 below possible without touching the environment.
* **Eval split**: rounds 40-59, `eval_query_obs`/`eval_truth`, 20 rounds/
  agent, untouched by every fit step of every replica.

## Estimator family (pre-declared): owner-specific B1 only

Every estimator in G5 is `Linear(63,16) -> Tanh -> Linear(16,1)`,
Xavier-uniform init, Adam, lr=1e-3, 2500 full-batch-bootstrap gradient
steps, batch = min(2048, n_rows) -- identical architecture and training
budget to G3's normalized condition and G4's B1. `RunningMeanStd`
normalizes both input observation and regression target, fit on that
*specific replica's own* training rows (never the full owner dataset,
never pooled across owners) and denormalized before any metric is
computed. No peer heads, no identity-conditioned input, no pooled/shared
network anywhere in G5.

### B1 -- full-data owner-specific control (positive control, reused)
One head per owner, trained on ALL 61,640 fit rows, seed `6000 + i`
(`i` = owner index 0-5) -- byte-identical recipe to G3's normalized
condition and G4's B1, retrained here (not read off disk) so its
per-sample predictions sit in the same report as the replicas. Tells us
the best calibration achievable when one model sees all available
owner-specific data; every replica below is compared against it.

### R1 -- bootstrap-resampled replicas
For owner `i`, replica `j` (0-indexed) of a `k`-replica condition: draw
`n = 61640` row indices **with replacement** from owner `i`'s full fit
array using `np.random.default_rng(seed_R1 + 1_000_000)`, where
`seed_R1 = R1_SEED_BASE[k] + i*10 + j` (`R1_SEED_BASE = {3: 21000, 5:
22000}`). Train a head on the resulting resampled `(obs, target)` pairs
with `seed_R1` driving both Xavier init and the training loop's own
bootstrap batch sampling. Each replica sees the same 61,640-row budget
and the same marginal row distribution as B1, but a different resample
draw -- this tests whether resampling noise alone (holding the
underlying 40-round data-generating process fixed) is enough to
decorrelate errors.

### R2 -- disjoint round-block replicas
For owner `i`, a `k`-replica condition partitions the 40 fit ROUNDS
(not rows) into `k` contiguous, non-overlapping blocks via
`numpy.array_split(arange(40), k)` (`k=3`: 14/13/13 rounds; `k=5`: 8/8/
8/8/8 rounds), each block owning exactly its rounds' `1541`-row slices
(`13*1541 = 20033` rows at `k=3`; `8*1541 = 12328` rows at `k=5`).
Replica `j` trains on block `j`'s rows only -- no bootstrap resampling
on top, no cross-block leakage, `RunningMeanStd` fit only on that
block. Seed `seed_R2 = R2_SEED_BASE[k] + i*10 + j`
(`R2_SEED_BASE = {3: 23000, 5: 24000}`) drives init and the training
loop's within-block bootstrap batch sampling. This tests genuine
data-source diversity: each replica never sees another replica's rows
at all, at the cost of a smaller fit set per replica.

### k (pre-declared, section 5 of the task)
Primary condition: `k = 3` (the theoretical floor for `f = 1`,
`M >= 2f+1 = 3`), for both R1 and R2. Secondary condition: `k = 5`, for
both R1 and R2, run because it is cheap (small heads, seconds per fit)
-- not required for the verdict, reported only to show whether
diversity continues to scale.

### Reference: G4 initialization-only replicas (not retrained)
`results/g4_identity_source_diagnostic/g4_report.json ->
independent_replicas_B2`: three replicas of the **shared
identity-conditioned B2 network**, same architecture/data, differing
only in Xavier-init/bootstrap-sampling seed. Reused verbatim as the
empirical "diversity floor" reference: pairwise residual correlation
`0.988-0.991`, participation ratio `1.014` of 3. G5 does not retrain an
owner-specific init-only condition -- the G4 number is architecturally
different (shared vs. owner-specific network) and is used only as the
pre-declared bar for "does data diversity beat init-only diversity",
per the task's explicit instruction to use the G4 result as the
reference rather than invent a new cutoff.

## Metrics (pre-declared, identical definitions to G3/G4)

Per (owner, replica): `bias = mean(pred - truth)`, `MAE = mean(|pred -
truth|)`, `RMSE = sqrt(mean((pred-truth)^2))`, `corr = Pearson(pred,
truth)`, `FSR = P(pred <= d | truth > d)`, `d = 25` (`budget_d` from the
reused dataset's `meta.json`), all over the 20 shared eval rounds for
that owner. Errors for the diversity analysis are
`e_{i,a,t} = pred_{i,a,t} - truth_{i,t}` (never raw predictions).

## Calibration bar (pre-declared, reused verbatim from G3/G4/G2)

* **Good**: `|bias| <= 5.0` AND `corr >= 0.5`.
* **Poor**: `corr < 0.2` OR `|bias| > 5.0`.
* Anything between is reported as-is and called neither ("mixed").
* **FSR acceptable**: `<= 0.20`.

## Diversity measures (pre-declared)

For every (condition, k, owner): the `k x k` Pearson correlation matrix
and covariance matrix of `{e_{i,a}}_{a=1..k}` over the 20 shared eval
rounds, plus `participation_ratio = (sum(lambda))^2 / sum(lambda^2)` of
that covariance matrix's eigenvalues (`PR = 1`: all variance on one
axis; `PR = k`: `k` equal uncorrelated axes). Reported per owner (G5c
requires this -- pooling across owners must not hide a per-owner
failure) AND pooled across owners by concatenating each replica's
6-owner x 20-round error vector into one 120-length vector per replica,
exactly mirroring G4's `independent_replicas_B2` construction, so the
two numbers are directly comparable (section J of the task).

**Effective source count**, reported alongside PR: `M_eff =
round(PR)`, capped at `k`. This is a descriptive summary, not a
re-derivation of the RCE theorem's `M` (same caveat G4 already applied
to its own PR numbers).

## G5a -- individual calibration (pre-declared gate)

Every (owner, replica, condition) cell reports bias/MAE/RMSE/corr/FSR
individually -- no pooled-only number is allowed to stand in for a
per-cell result. **Pass** iff, for a given (condition, k), at least
5 of 6 owners have EVERY replica in that condition individually
classified "good"; **fail** otherwise. The 5-of-6 (not 6-of-6) bar is
set because G4's own B1 control already left 2 of 6 owners (`agent_4`,
`agent_5`) in "mixed", not "good" -- G5a must not silently raise the
bar for replicas beyond what G5's own full-data control (B1) achieves;
see G5f' cross-reference below and section F of the report.

## G5b -- owner coverage (pre-declared gate)

All six owners are evaluated and reported separately in every table in
this diagnostic; no aggregate statistic (pooled bias, pooled corr, mean
PR) may be reported without its accompanying per-owner breakdown in the
same section. **Pass** iff every table in section F/G of the report has
a per-owner row; **fail** if any table only reports a pooled number.

## G5c -- replica diversity (pre-declared gate)

For each (condition, k), report the per-owner pairwise error-correlation
matrix, covariance matrix, eigenvalues, participation ratio, and
`M_eff`, for all six owners individually, plus the pooled-across-owners
version. No pass/fail threshold is attached to G5c itself -- it is a
measurement gate (did the analysis get produced, per-owner, for every
condition) rather than a bar, since the bar is G5d below.

## G5d -- diversity beyond initialization (pre-declared gate)

**Pass** iff, for the pooled-across-owners construction, at least one of
R1-k3 or R2-k3's max pairwise replica correlation is `<= 0.90` (a
`>= 0.08` absolute reduction from G4's measured `0.988-0.991` range,
chosen because it is the smallest round number below G4's observed
minimum of `0.988` -- not an arbitrary cutoff invented after seeing G5's
own numbers, per the task's explicit instruction). **Materially better**
(stronger claim, reported separately) iff max pairwise correlation
`<= 0.7`. **Fail** iff every condition's max pairwise correlation stays
`>= 0.90` -- data splitting would not have solved what G4 already showed
initialization splitting could not.

## G5e -- three-source viability (pre-declared gate)

For `k = 3` only (R1 and R2 separately): **pass** iff G5a passes for
that condition (>=5/6 owners individually "good" on every replica) AND
G5d passes using that condition's OWN pooled max pairwise correlation
(not the other condition's). Passing G5e is explicitly NOT "honest
majority" or Assumption 1(ii) -- see "What G5 does not establish" below
-- it only says three owner-specific, differently-trained estimators of
the same `J_C^i` can be simultaneously calibrated and non-redundant on
this fixed dataset.

## G5f -- stability (pre-declared gate)

**Pass** iff every one of the `6 owners x (1 B1 + 2 k-values x 2
conditions x k replicas) = 6 x (1 + 2*2*3 + 2*2*5) = ...` -- concretely,
every trained head (B1: 6; R1-k3: 18; R1-k5: 30; R2-k3: 18; R2-k5: 30;
total 102 heads) completes its 2500 training steps with finite loss at
every logged step and produces finite (non-NaN, non-inf) predictions on
its 20 eval rows. **Fail** if any head produces NaN/inf anywhere.

## Accuracy-diversity tradeoff (pre-declared, section 11 of the task)

For R2 (disjoint data) only: compare each owner's per-replica calibration
at `k=3` (20,033-row blocks) and `k=5` (12,328-row blocks) against that
owner's B1 full-data (61,640-row) calibration. **Tradeoff observed** iff
any owner's R2 classification degrades from "good" (under B1) to "mixed"
or "poor" (under R2 at either k) per the bar above. **No tradeoff** iff
every owner that is "good" under B1 stays "good" under both R2
conditions. This is reported descriptively (section I of the report),
not as a pass/fail gate -- the task explicitly says the tradeoff, if
found, is itself a scientifically important result, not a failure to
paper over.

## Decision tree (pre-declared, task section 13)

* **Outcome A (strong candidate)**: G5a passes AND G5d passes with
  "materially better" (`<=0.7`) for at least one `k=3` condition (R1 or
  R2).
* **Outcome B (accurate but redundant)**: G5a passes for a condition but
  G5d fails for that same condition (max pairwise corr stays `>=0.90`).
* **Outcome C (diverse but inaccurate)**: G5d passes for a condition but
  G5a fails for that same condition.
* **Outcome D (neither)**: G5a and G5d both fail for every condition
  tested.
Different conditions (R1 vs R2, k=3 vs k=5) may land in different
outcomes; section L of the report states the outcome per condition, not
one global label, then gives one overall verdict per the rule below.

## G5 verdict rule (pre-declared)

* **PASS** -- at least one `k=3` condition (R1 or R2) reaches Outcome A.
* **CONDITIONAL PASS** -- at least one `k=3` condition reaches Outcome A
  or B (calibration holds) but none reaches Outcome A with "materially
  better" diversity, OR diversity/calibration holds for some owners but
  not others (owner-level split, e.g. the same `agent_4`/`agent_5`
  weakness G4 already flagged).
* **FAIL** -- every `k=3` condition lands in Outcome C or D: owner-
  specific replication cannot simultaneously deliver calibration and
  non-redundant errors on this fixed dataset.

## What this diagnostic explicitly does not establish

Regardless of verdict: no claim about Assumption 1(ii) (source
independence / honest-majority) -- that requires a statistical
independence argument (e.g. conditional independence given the true
`J_C^i`, or an explicit error-generating-process model) beyond "measured
pairwise correlation on 20 eval rounds is below a threshold". No claim
about Theorem 2. No claim about attack stealth or RCE. No claim
generalizing beyond `pilot_A_clean_seed0`'s one frozen policy snapshot
and one 60-round dataset -- same single-checkpoint, single-dataset
limitation G3/G4 already declared. No claim that `M = 7` or any specific
production `M` is licensed by this diagnostic; `M_effective` for any
future run must still be recomputed by counting only sources/replicas
whose calibration clears "good" above, per G3/G4's own standing rule.
And even a full G5 PASS establishes only that construction is possible
on 20 held-out eval rounds from one frozen checkpoint -- not that it
would survive being re-embedded in the full nonstationary 250-round
training loop, where fit data accrues online rather than being fixed in
advance.
