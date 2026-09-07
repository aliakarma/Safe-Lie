# A2 — Does RCE mitigate persistent safety-cost corruption?

**Status: PRE-DECLARED. Written and committed before any condition-C or
condition-E run existed.** Every threshold in this document is fixed here.
No bar below was chosen, moved, or softened after seeing a C or E result.

A1 established the attack effect. A2's only job is to establish the
**defense** effect, and specifically to separate two things that the raw
`C − B` improvement confounds:

1. RCE being *generically conservative* — it would change the policy even
   with no attacker present;
2. RCE providing *attack-specific* protection.

The decomposition is fixed in advance and is not renegotiable:

```
attack effect               =  B − A
clean RCE effect            =  E − A
attack effect under RCE     =  C − E
attack x RCE interaction    = (C − E) − (B − A)      <- the primary quantity
RCE main effect (raw)       =  C − B                 <- reported, never primary
```

---

## 1. Conditions

| Condition | Attack | RCE | Purpose | Runs |
|---|---|---|---|---|
| **A** | no | no | clean reference | **exists** (G9/G10) |
| **B** | yes | no | attack reference | **exists** (A1) |
| **C** | yes | yes | defense | **new**, seeds 0/1/2 |
| **E** | no | yes | RCE conservatism | **new**, seeds 0/1/2 |

Six new 250-round runs. A and B are reused verbatim and are **not** re-run.

Existing and new run directories:

```
A: seed 0  results/runs_constraint_batch_g9/g9_batch_clean
   seed 1  results/runs_constraint_batch_g10/seed1
   seed 2  results/runs_constraint_batch_g10/seed2
B: seed s  results/runs_a1/B_seed<s>
C: seed s  results/runs_a2/C_seed<s>     (new)
E: seed s  results/runs_a2/E_seed<s>     (new)
```

## 2. What is frozen

The A1/G10 architecture, unchanged and unmodified:
`manyagent_ant` N=6, ring topology, `d = 25`, `velocity_threshold = 0.75`,
`cost_mode = per_agent_velocity`, `M = 3`, `R_m = 30`, parallel
trajectory-batch sources under a pinned `theta_k`, `constraint_estimator =
mc_window`, PPO (clip 0.2, gamma 0.99, GAE-lambda 0.95, lr 3e-4, 4 epochs,
4 minibatches, hidden 64, entropy 0.001), `eta_lambda = 0.035`,
`lambda_max = 25`, 250 rounds x 2000 steps, `R_ref = 120` at rounds
{25, 75, 125, 175, 225}, `workers: 12`, `chunks_per_worker: 4`.

The **only** field that changes between a new A2 config and its frozen
counterpart is the `defense:` block (plus `run_id` / `output_dir`).

```yaml
defense:
  name: rce
  f: 1
  beta: 1.5          # [SPEC]
  sigma_min: 0.001
  min_retained: 3
  calibration_rounds: 20     # shipped default, unchanged
  calibration_alpha: 0.05
```

`C_seed<s>` = `configs/experiment/a1/b_seed<s>.yaml` + that block.
`E_seed<s>` = the frozen clean config for seed `s` + that block.

## 3. Seed pairing and common random numbers

Balanced attacked-source mapping, inherited from A1 section 3 unchanged:

| seed | attacked source (B and C) | `seed_entropy` |
|---|---|---|
| 0 | `batch_1` | 286314957402113664887331205920951063913 |
| 1 | `batch_2` | 52021175099534868945312500562751741008 |
| 2 | `batch_3` | 335268198726974212397240672597355197200 |

Within a seed, all four conditions share: `seed`, `seed_entropy`, hence the
identical 23,100 `(env, torch)` source-seed pairs in the identical order;
the identical policy initialisation; and the identical oracle evaluation
seed stream (`SeedBundle.eval` derives from `cfg.seed` alone). Condition E
carries no attack and therefore no attacked source; it is paired to A.

**RCE must not alter source-generation randomness.** This is gated, not
assumed — see A2-G1-ii.

## 4. Exact RCE map as implemented (audited before launch)

`safelie/training/loop.py` calls, independently for **each owner `i`**,
after the attack hook and before the dual update:

```
values^i   = { Jhat_C^{i,m} + delta_m }_{m=1..M}   delta_m = -B for the corrupted m, else 0
agg        = rce_aggregate(values^i, f, beta, sigma_min, min_retained)
residual^i = agg.pessimistic_estimate - d
lambda    <- Proj_[0,lambda_max]( W lambda + eta_lambda * residual )
```

`rce_aggregate` = `trimmed_mean_aggregator` (drop the `f` largest and `f`
smallest), then MAD over the **retained** set, then `Jbar = Jhat +
beta*spread`.

**At M = 3, f = 1 the retained set has exactly one element.** Therefore:

* `Jhat = median(values)` exactly;
* `MAD({single point}) == 0` structurally, for every possible input;
* `retained_n = 1 < min_retained = 3`, so `degenerate = True` every round;
* `0 < sigma_min`, so the spread is floored: `spread == sigma_min = 1e-3`;
* `applied_margin == beta*sigma_min = 1.5e-3` — a **constant**, 0.006 % of `d`.

so the implemented map is exactly, and provably for all inputs,

```
Jbar_RCE^i  =  median_m ( Jhat_C^{i,m} + delta_m )  +  beta * sigma_min
```

Verified numerically over 50,000 random triples: max deviation from
`median + beta*sigma_min` is exactly 0.0. **The MAD/pessimism term of
Algorithm 1 is structurally inert at this operating point.** A2 therefore
tests the *trimming* half of RCE, never the *margin* half. This is recorded
here, before launch, so it cannot later be presented as a discovery of the
results, and so no A2 outcome can be read as evidence for Theorem 2's
margin condition.

**Consequence for `guarantee_in_force`.** `epsilon_offline` is a quantile
of clean-run `spread` values, all of which are exactly `sigma_min`; so
`epsilon_offline == sigma_min` and `guarantee_in_force = (1.5e-3 >= 1e-3)`
is `True` unconditionally, for reasons that have nothing to do with whether
Theorem 2's precondition holds. **`guarantee_in_force` is vacuous in A2 and
is not evidence of anything.** It is reported only so this is on the record.

## 5. Pre-registered mechanism prediction

Between-source dispersion measured from the three **existing condition-A**
runs (no C/E data involved): pooled mean sd across the 3 sources per
(round, owner) is `sigma_src = 1.053` (per-seed 1.124 / 1.003 / 1.031;
median 0.963, p95 2.100).

`B = 12.5 = 11.9 sigma_src`, so the corrupted report is essentially always
the smallest of the three. The trimmed set then retains **the smaller of
the two honest sources** instead of the middle of three, giving a predicted
attack-induced shift of the RCE aggregate of

```
E[min of 2] - E[median of 3] = -sigma_src/sqrt(pi) = -0.59
```

against the mean aggregator's exact `-B/M = -4.1667` (confirmed in A1).
Predicted suppression of the *estimator* corruption: **about 7.0x**.

## 6. Gates

### A2-G1 — implementation integrity (gating)

* **G1-i.** For every seed: `C` vs `B` and `E` vs `A` differ only in
  `run_id`, `output_dir`, and `defense.*`. Nothing else. Checked on the
  fully-resolved Pydantic model by `scripts/a2_verify_frozen.py`.
* **G1-ii.** **CRN preserved.** At round 0, the pre-attack per-source
  values of `C` equal those of `B`, and of `E` equal those of `A`, to
  <= 1e-9 on all 3 sources x 6 owners. RCE must not perturb source
  generation.
* **G1-iii.** Source seed audit: 23,100 env seeds issued, 0 duplicates,
  0 replica-pairwise overlap; spawn keys equal to the paired run's.
* **G1-iv.** PPO env steps exactly 250 x 2000 = 500,000; source env steps
  exactly 45,000,000. RCE adds no environment interaction to the run.
* **G1-v — disclosed deviation.** Selecting `defense.name: rce` causes
  `safelie.eval.calibration.run_clean_calibration` to execute a 20-round
  attack-disabled clone **before round 0**. It runs before
  `seed_everything(cfg.seed)`, feeds only the logged Boolean
  `guarantee_in_force`, and (per section 4) can only ever return
  `epsilon_offline = sigma_min`. Gate: `guarantee_calibration.json` reports
  `epsilon_offline == 0.001` exactly in all six runs, **and** G1-ii and
  G1-iii hold. Its cost (about 0.6 h per run) is disclosed, not hidden.

### A2-G2 — attack replication (gating)

* **G2-i.** `C` and `B` carry a byte-identical `attack` block:
  `name: primary`, `f: 1`, `budget_ratio: 0.5`, `direction: negative`,
  `support: persistent`, `adaptivity: static`, `consistency: consistent`,
  and `corrupted_source_ids` equal to the section 3 mapping for that seed.
* **G2-ii.** Injection identity on 100 % of (round, owner) cells of `C`:
  reconstructing `values` from the logged pre-attack `reports`, applying
  `-12.5` to the mapped source, and re-running `rce_aggregate` reproduces
  the logged `point_estimate`, `spread`, `retained_n`, `applied_margin`
  and `pessimistic_estimate` to <= 1e-9.
* **G2-iii.** `E` carries `attack.name: none`, `attack.f: 0`, and the same
  identity holds with no shift applied.

### A2-G3 — clean RCE effect (reported, not gated on its value)

`E − A` on whole-run network-average true cost, paired by seed: per-seed
differences, mean, 95 % CI, sign consistency, `d_z`. Also `E − A` on
final-50 true cost, task return, lambda, violation rate, per-agent cost.

Gated only for run validity: all three E runs complete 250 rounds, all
quantities finite. **The value itself is a measurement, not a hypothesis** —
a large `E − A` is the "RCE is intrinsically conservative" finding, not a
failure.

### A2-G4 — residual attack effect (reported)

`C − E` on whole-run network-average true cost, paired by seed, reported
identically to `B − A` and next to it.

### A2-G5 — interaction (gating; the primary result)

`I_cost = (C − E) − (B − A)` on whole-run network-average true cost.
Higher true cost = worse, and `B − A = +3.986 > 0` is established, so
`I_cost < 0` means the attack costs less under RCE than without it.

Pre-registered decision rule, fixed now:

| Outcome | Rule |
|---|---|
| **PASS — defense supported** | mean `I_cost < 0`, `I_cost < 0` in **all 3** seeds, **and** mean `(C − E) <= 0.5 x (B − A)` — RCE removes at least 50 % of the attack-induced cost increase |
| **CONDITIONAL PASS** | mean `I_cost < 0` but the sign is not unanimous across seeds, **or** the reduction is above 0 % but below 50 % |
| **FAIL** | mean `I_cost >= 0`, i.e. `C − E >= B − A` |

The 50 % bar is the lenient form of section 5's mechanistic prediction (a
7x reduction in estimator corruption predicts about 14 % of `B − A`
surviving). It is set here, before any C or E round has been run.

`C − B` is reported alongside and is **never** substituted for `I_cost`.

### A2-G6 — learning health (gating, inherited from G10-C verbatim)

No NaN/Inf in mechanism estimate, residual, lambda, KL, entropy, true cost
or spread; PPO KL median <= 0.0069 and p95 <= 0.01548; lambda saturation
fraction < 0.05; entropy declines (last-10 mean < first-10 mean) and stays
finite and positive; theta checksum distinct in 100 % of rounds.

`G10-C-ii` (task-return gain >= +75) is **reported but not gated in C or E**,
for the same reason A1 did not gate it in B or D: a defense that degrades
learning is Outcome E, a real finding. Gating it would make Outcome E
unreachable.

### A2-G7 — RCE mechanism behaves as the audited math says (gating)

* **G7-i.** `retained_n == 1` and `degenerate == True` on 100 % of cells
  in C and E.
* **G7-ii.** `spread == sigma_min` and `applied_margin == beta*sigma_min ==
  0.0015` on 100 % of cells. **This is the expected result, not a failure**:
  it is the section 4 audit confirmed on live data, and it is what makes
  "the MAD condition is never active" a measurement rather than a claim.
* **G7-iii.** In C, the attacked source is **trimmed** (absent from the
  retained set) on >= 99 % of cells, and the retained value equals the
  smaller of the two honest sources on >= 99 % of cells.
* **G7-iv.** Within-run counterfactual (zero sampling noise): for every
  cell of C, `RCE(post-attack) - RCE(pre-attack)`. Predicted mean
  `-0.59`; gate band `[-1.20, -0.20]`. Compare against B's exact `-4.1667`.
* **G7-v.** Source ordering stability: fraction of consecutive rounds in
  which the argsort of the three source values is unchanged, reported per
  condition (descriptive, not gated).

## 7. Stop conditions (halt the queue; do not continue spending compute)

The queue halts immediately if any of these fire on a finished run:

1. any G1 or G2 sub-gate fails — RCE reached outside the aggregation
   stage, or the attack differs between B and C;
2. round-0 source values differ between C and B, or between E and A
   (RNG pairing broken);
3. the attacked source in C is not the section 3 mapping's source;
4. any NaN/Inf in an RCE output;
5. the seed audit reports a duplicate or a wrong issued-seed count;
6. `rounds.jsonl` round count != 250, or derived env steps != the G1-iv
   values;
7. oracle/reference leakage (`tests/isolation`, plus no learner-visible
   field equal to the round's oracle `true_cost_return`).

## 8. Statistics

`n = 3`. Paired seed-level differences throughout. Primary contrast is
`I_cost`; secondary are `E − A`, `C − E`, `C − B`. Reported for each: per
seed difference, mean, sd, 95 % CI, `d_z`, sign consistency.

Per decision D6 (`MIN_SEEDS_FOR_INFERENCE = 5`) the paired t-test is
emitted **labelled as sensitivity only** and no significance claim is made
at three seeds, for any contrast, regardless of the p-value. No
non-pre-declared contrast will be promoted on the basis of a small p-value.
No seeds will be added after seeing results — the same rule that made
A1's `B − D` permanently descriptive.

## 9. Reporting (mandatory, every condition, every seed)

**Task:** whole-run return, first-20, first-50, final-50, learning gain.
**Safety:** whole-run and final-50 network-average true cost; mechanism
reported estimate; aggregate estimate; residual; lambda; **all six
per-agent `J_C^i`, agents 4 and 5 never omitted**.
**Attack diagnostics:** attacked-source shift, between-source spread, RCE
output, detection gap.
**Constraint behaviour:** expected network-average cost (primary), episode
violation rate and peak violation (reported, **not** primary — the
constraint is an expected-value constraint and zero clean violation is not
required).
**RCE diagnostics (sections 4/5):** per round, the three source values, the
trimmed/retained value, MAD, beta*MAD, `epsilon_offline`, whether the
theoretical condition is active, and which source was trimmed.

## 10. What A2 can and cannot establish

**Can, at most:** "RCE reduces the empirical attack-induced increase in
true network-average cost under the tested `M=3, f=1, B/d=0.5` condition,
in this environment, at three seeds."

**Cannot, and will not be claimed:** Theorem 2's numerical bound (its MAD
condition is structurally inert here, section 4); robustness for `f > 1`;
topology generality; a second environment; dose response in `B`;
over-reporting liveness; superiority over other robust aggregators
(median, Krum, trimmed mean are not run); and "RCE guarantees safety" in
any form.

## 11. Compute

6 x 250 rounds. Measured A1/G10 cost is 6.5–10.1 h per run (mean about
8.5 h), plus about 0.6 h per run for the disclosed G1-v calibration phase.
Expected total about 55 h, sequential. Source env steps: 6 x 45,000,000.
