# A1 — clean vs magnitude-matched generic perturbation vs persistent directional under-reporting

**Status: pre-declared. Written and committed before any A1 training run
existed. Every threshold below is derived from the *already committed* G0–G10
artifacts and the sampling law, and from nothing else. The commit that adds
this file adds no A1 result.**

A1 is the first experiment in this repository whose purpose is to test the
paper's central scientific claim rather than to validate the measurement
apparatus. G0–G10 asked "can this pipeline measure anything?". A1 asks:

> Does persistent directional corruption of one valid safety-feedback source
> increase true safety cost, beyond what a magnitude-matched generic
> perturbation produces?

---

## 0. What A1 is *not*

No RCE. No robust aggregation. No defense of any kind — A1 is deliberately the
**attack-without-defense** experiment, so `defense.name = mean`, `defense.f = 0`
in every condition, exactly as in the clean G10 runs. No topology sweep, no β
sweep, no `f = 2`, no `M = 5`, no second environment, no baseline comparison,
and no attack-strength dose–response: `B/d = 0.5` only.

---

## 1. What is frozen

The G10-validated implementation, unmodified. `scripts/a1_verify_frozen.py`
proves this mechanically rather than by assertion: it loads each A1 config and
its seed's clean config, flattens both resolved Pydantic models, and fails if
any key outside `{run_id, output_dir, attack.*}` differs. Its output is
committed to `results/runs_a1/a1_frozen_verification.json`.

Inherited verbatim: parallel trajectory-batch source construction, `M = 3`,
`R_m = 30`, pinned-policy collection, PPO, GAE, the dual update, the ring
consensus matrix, budget `d = 25`, `η_λ = 0.035`, `λ_max = 25`, the environment
(`manyagent_ant`, N=6, `velocity_threshold = 0.75`,
`cost_mode = per_agent_velocity`), the task reward, the source estimator, the
oracle evaluator, 250 rounds at `rollout_length = 2000`, the withheld
`R_ref = 120` reference at rounds {25, 75, 125, 175, 225}, and the worker/chunk
scheduling knobs.

**The only new scientific factor is the corruption treatment.**

### 1.1 The one code change, and why it was necessary

`AttackConfig.corrupted_source_ids` (`src/safelie/utils/config.py`) names which
sources the adversary holds; `select_corrupted_sources`
(`src/safelie/training/loop.py`) honours it. `None` — the value in every
pre-A1 config — preserves the historical rule exactly, so no committed config
changes meaning.

It is necessary because §3 balances the attacked source across seeds, and the
obvious alternative — reordering `cfg.sources.sources` so the target is first —
would silently destroy the §5 pairing. Under `parallel_trajectory_batch`,
`ParallelBatchSourceCollector` zips `cfg.sources.sources` against the
`SeedSequence(seed_entropy).spawn(M+1)` children **by position**. Moving
`batch_2` to the front would hand it `batch_1`'s stream, changing which 7,500
trajectories every replica draws. Naming the source leaves all M streams where
they were. `tests/unit/test_corrupted_source_selection.py` pins both properties.

---

## 2. The three conditions

All three share one training seed, one policy initialisation, and one set of
source RNG streams (§5).

### A — CLEAN (matched control)

No perturbation. **Already run**; not re-run. Seed 0 is the committed G9 run,
seeds 1 and 2 the committed G10 runs (§4). The attack hook still executes and
is the identity map, so the plumbing is exercised in condition A too, not
bypassed.

### D — MAGNITUDE-MATCHED GENERIC PERTURBATION

`safelie.attacks.static.benign_control`, unchanged from the pilot. Applied to
the same single source condition B attacks.

| property | value |
|---|---|
| distribution | Gaussian |
| mean | **0**, exactly — not a sample-zero-mean construction |
| sd | `σ = budget_ratio · d = 12.5` — the **same nominal magnitude** as B's `B` |
| temporal structure | i.i.d., redrawn every round; no persistence, no autocorrelation, no drift |
| clipping | **none** |
| source | one source, per the §3 mapping — the same one B attacks |
| duration | every round 0…249, same as B |
| draw granularity | one draw per (round, owner); the hook runs inside the per-owner loop |
| RNG | `SeedBundle.rng("attack")`, a stream conditions A and B never draw from |

**This is not an "adversarial" control and is never described as one.** It is a
generic/matched perturbation control. See §9.

### B — PERSISTENT DIRECTIONAL UNDER-REPORTING

`safelie.attacks.static.static_attack` with `direction="negative"`,
`support="persistent"`:

```
J_hat_{C,m}^attack = J_hat_{C,m} − B ,      B = 0.5 · d = 12.5
```

on exactly one source, on every round, with constant sign.

`B/d = 0.5` is confirmed by code inspection to be the same parameterisation the
pilot used: `apply_attack` computes `B = cfg.budget_ratio * d`, and
`static_attack` adds `−1.0 · B` to the **residual** `g = J_hat_C − d`, which is
arithmetically identical to subtracting `B` from `J_hat_C`. So
`budget_ratio: 0.5` with `d = 25` gives `B = 12.5`, and

```
predicted mean-aggregate shift = −B/M = −12.5/3 = −4.16667
```

**Injection point.** Immediately after the source registry produces the
legitimate value and immediately before mean aggregation and the dual update —
`src/safelie/training/loop.py`, the `apply_attack` call inside the per-owner
loop. The attack does not touch the oracle, the environment cost, the PPO
rollout, or the withheld `R_ref` reference.

---

## 3. Which source is attacked — balanced, and declared before running

Three sources, three seeds. Attacking `batch_1` every time would make source
identity a perfect confound with the treatment. The mapping is fixed here:

| seed | attacked / perturbed source |
|---|---|
| 0 | `batch_1` |
| 1 | `batch_2` |
| 2 | `batch_3` |

**The same source is used for B and for D at a given seed**, so the B−D
contrast is not confounded by source identity either.

---

## 4. Seeds and runs

`seed ∈ {0, 1, 2}` — the same three training seeds G10 validated.
3 conditions × 3 seeds = **9 runs, of which 3 already exist.**

| condition | seed | config | output |
|---|---|---|---|
| A | 0 | (already run as G9) | `results/runs_constraint_batch_g9/g9_batch_clean` |
| A | 1 | (already run as G10) | `results/runs_constraint_batch_g10/seed1` |
| A | 2 | (already run as G10) | `results/runs_constraint_batch_g10/seed2` |
| B | 0 | `configs/experiment/a1/b_seed0.yaml` | `results/runs_a1/B_seed0` |
| B | 1 | `configs/experiment/a1/b_seed1.yaml` | `results/runs_a1/B_seed1` |
| B | 2 | `configs/experiment/a1/b_seed2.yaml` | `results/runs_a1/B_seed2` |
| D | 0 | `configs/experiment/a1/d_seed0.yaml` | `results/runs_a1/D_seed0` |
| D | 1 | `configs/experiment/a1/d_seed1.yaml` | `results/runs_a1/D_seed1` |
| D | 2 | `configs/experiment/a1/d_seed2.yaml` | `results/runs_a1/D_seed2` |

Analysis is **paired by seed**: `A_s ↔ D_s ↔ B_s`. No seed is selected or
dropped. Runs are sequential — the source collector already saturates all 12
logical CPUs.

**Condition D is a genuine independent training run, not a replay.** A replay
is impossible here in principle: the perturbation changes `λ`, which changes the
policy, which changes every subsequent trajectory. Nothing in A1 is obtained
without retraining, and nothing in A1 is presented as if it were.

**Cost.** 6 new runs. From the committed G9/G10 metadata, 7.6–10.1 h each
(45 M source env steps, 0.5 M PPO steps, 0.5 M oracle steps per run):
**≈ 46–60 h sequential wall clock**, plus 135 M source environment steps.

---

## 5. Randomness: what is shared across conditions, what stays independent

### 5.1 Shared (common random numbers — deliberate)

For a given training seed, conditions A, B and D use:

* the **same** `seed`, hence the same `SeedBundle` and therefore the same
  `policy_init`, `env`, `eval` and `torch` streams;
* the **same** `source_collection.seed_entropy`, hence the **identical**
  23,100 `(env_seed, torch_seed)` pairs drawn in the identical order by the
  M+1 spawned streams.

Consequence, and the reason this matters: at **round 0**, before the policies
can diverge, the underlying source values in A, B and D are **bit-identical**,
and the only difference is the injected corruption. That is what makes §7's
mechanism identity an exact check rather than a two-sample comparison.

### 5.2 Independent (must remain so)

* The three replica streams stay mutually disjoint **within** every run — 0
  shared env seeds, 0 duplicates, audited per run by the existing
  `source_seed_audit`. No source stream is ever reused across the three
  sources inside one run.
* Condition D's noise is drawn from `SeedBundle.rng("attack")`, a stream A and
  B never touch. Adding it therefore cannot perturb the env, policy-init,
  torch or source streams that D shares with its clean control.

### 5.3 A consequence of D's draw granularity, stated in advance

D draws **one perturbation per (round, owner)**, because the corruption hook
sits inside the per-owner loop. This is the pilot's behaviour and A1 does not
change it. It has an arithmetic consequence that must not be discovered later
and presented as a finding:

* B shifts every owner's aggregate by a **constant** `−B/M = −4.167`, so after
  `K` rounds the multiplier has integrated a drift of `η_λ · B/M · K`.
* D shifts owner `i`'s aggregate by `N(0, (σ/M)²)` with `σ/M = 4.167`,
  **independently** across owners and rounds, so the multiplier integrates a
  random walk of sd `η_λ · σ/M · √K` — smaller by a factor `√K ≈ 15.8` at
  `K = 250`.

D is therefore matched to B in **per-round magnitude** and differs in
**persistence and direction**. That is the intended contrast — persistence and
direction are precisely the properties under test — but it means a B−D
separation is evidence about *persistence and direction*, not about intent.
§9 governs how this may be described.

---

## 6. Which constraint is primary — resolved before running, with a flag

### 6.1 The finding

`paper/main_iclr.tex:109` states the objective as **per-agent**:
`J_C^i(θ) ≤ d^i for all i ∈ N`, and adds that "the shared-constraint case
`(1/N)Σ_i J_C^i ≤ d` is recovered by taking `W = (1/N)11ᵀ`".

The implementation is the paper's Eq. 2 verbatim,
`λ_{k+1} = Π_[0,λmax][W λ_k + η_λ g_k]`, with a doubly-stochastic ring `W`.
Verified from the committed clean logs, all three seeds: dual reconstruction
error **0.0**, mixing error **1.8e-15**. The implementation is faithful.

But Eq. 2's stationary point satisfies `(I − W)λ* = η_λ g*`, so `g*` is pinned
only up to `range(I − W)`. Measured on the actual matrices:

| `W` | `rank(I − W)` | what the dual fixed point pins |
|---|---|---|
| `identity` (no communication) | 0 | `g^i = 0` **per agent** |
| `ring` (used here) | 5 | `1ᵀg = 0` only — the **network average** |
| `complete` | 5 | network average |
| `star` | 5 | network average |

Confirmed in the data. Last-50 **sum** of reported residuals is
+0.33 / −0.66 / +0.20 across seeds 0/1/2 — zero to within noise — while the
per-agent residuals stay stratified across a ~10-unit range
(−2.9 … +7.2) and `λ` never equalises (cross-agent spread 0.88–0.96).

### 6.2 Flag: this is a paper-level gap, not an implementation defect

**The paper's per-agent claim is not delivered by the paper's own algorithm**,
and the parenthetical in `main_iclr.tex:109` has it backwards: *every* connected
doubly-stochastic `W` yields the shared/average constraint at the dual fixed
point. `W = (1/N)11ᵀ` is not the special case that produces it; `W = I` is the
only case that produces per-agent enforcement. Consensus mixing is exactly what
converts per-agent enforcement into network-average enforcement.

This is raised here, before A1 runs, and is **flagged for the paper**, not
silently absorbed. It is not created by the source architecture and not created
by A1; G10 §3.1 recorded the same behaviour.

**It does not block A1**, for a reason stated in advance: A1's outcome is a
*paired difference between conditions that share identical enforcement
semantics*. Whatever Eq. 2 enforces, it enforces it identically in A, B and D,
so `Δ_attack = B − A` is well posed either way.

### 6.3 The decision

* **Primary safety outcome: the network-average expected true cost**
  `(1/N) Σ_i J_C^i`, because that is what the dual demonstrably enforces.
* **Per-agent expected true costs `J_C^0 … J_C^5` are mandatory co-reported**
  for every condition and seed, with CIs, never pooled away (§10). G10
  established a reproducible structural imbalance — agents 0–3 near or below
  budget, agents 4–5 above it — in every clean campaign across three source
  architectures. A1 does not hide it and does not describe a result using the
  pooled average alone.
* The per-agent **attack increment** `J_C^i(B) − J_C^i(A)` is well posed for
  every `i` regardless of §6.1, and is gated (A1-G3).

---

## 7. Structural gates — the stop conditions, checked per run

**Every one must pass exactly.** A failure halts A1; the implementation is
fixed and A1 restarts. These are correctness checks, not statistics.

| id | check | how |
|---|---|---|
| **A1-S1** | every G10 structural gate (G10-S1…S10) still passes, per run | re-run the G10 structural analyser against each A1 run |
| **A1-S2** | the attack is applied at the intended point | `rounds.jsonl` records the **pre-attack** source values in `constraints.<agent>.reports[].value` and the **post-attack** aggregate in `mechanism_reported_cost_return`; the identity below must hold on **every round and every owner** |
| **A1-S3** | the measured source shift is the pre-declared `−B` | §7.1 |
| **A1-S4** | the attacked source never enters PPO | PPO env steps exactly `250 × 2000 = 500,000`, independent of source steps; `TestNoNeuralPathInTheBatchSourcePipeline`, `TestPpoIsUnaffected` |
| **A1-S5** | source streams independent within the run | 23,100 unique env and torch seeds, 0 duplicates, 0 replica-pairwise overlap, 0 replica-vs-reference overlap |
| **A1-S6** | policy pinning holds | `worker_checksums_all_match` for all 48 chunks every round; 250 distinct θ checksums |
| **A1-S7** | oracle isolation holds | `tests/isolation/test_oracle_isolation.py`; no learner-visible field equals the round's oracle `true_cost_return` |
| **A1-S8** | the attack does not change the environment cost | the oracle's `true_cost_return` path contains no attack term — the corruption acts only on `residuals` inside the per-owner loop; and the withheld `R_ref` reference is collected before the hook and must be **un**shifted (§7.2) |
| **A1-S9** | CRN integrity | B and D each share all 23,100 source seed pairs with their seed's clean run, in order |
| **A1-S10** | nothing but the treatment differs from G10 | `scripts/a1_verify_frozen.py` exits 0 |

### 7.1 The mechanism identity (A1-S2/S3) — exact, not approximate

Within any attacked run, on every round `k` and every owner `i`:

```
mean_m( reports[m].value )  −  mechanism_reported_cost_return  =  +B/M  =  +4.16667
```

because `reports[].value` is logged pre-attack and the aggregate is computed
post-attack. Required to hold to **≤ 1e-9** on **100%** of the 250 × 6 = 1500
(round, owner) cells. This is far stronger than a first-round check: it
verifies the injection point on every round of the run.

Across the clean/attacked pair at **round 0**, where §5.1 makes the underlying
draws bit-identical:

```
J_hat_m^attack  − J_hat_m^clean   =  −B      =  −12.5     (attacked source, ≤1e-9)
J_hat_m'^attack − J_hat_m'^clean  =   0                   (the other two, ≤1e-9)
mean^attack     − mean^clean      =  −B/M    =  −4.16667  (≤1e-9)
```

### 7.2 The reference must be un-shifted (A1-S8)

`validation_reference.jsonl`'s `reference_mean` and `source_means` are written
by the collector **before** the attack hook. In an attacked run they must be
**identical to the clean run's** at the same round if the policies have not yet
diverged, and in all cases must show **no** `−B` offset. A shifted reference
would mean the corruption had reached the measurement apparatus.

---

## 8. Scientific gates

`d = 25`. Every gate is evaluated on **paired** per-seed differences.

### A1-G1 — the primary causal contrast: does the attack increase true cost?

`Δ_attack-cost = J_C^true(B) − J_C^true(A)`, network-average, paired by seed.

* **G1-i.** The paired difference is **positive in all 3 seeds** on whole-run
  network-average true cost.
  *Bar:* sign consistency across all seeds is the strongest claim n=3 licenses
  under decision D6 (`safelie.eval.protocol.consistent_across_seeds`). Under a
  true null the probability of 3/3 same-sign is 0.25.
* **G1-ii.** The paired mean difference exceeds **+1.0 cost units** (4% of `d`)
  on whole-run network-average true cost, with its 95% CI reported.
  *Bar:* the three clean runs' whole-run network averages are 25.214, 24.955,
  24.978 — between-seed sd **0.143**. +1.0 is 7 clean sd, and is the same
  4%-of-`d` bar G10-A-ii used for aggregate bias. The pilot's observed B−A was
  +2.98.
* **G1-iii.** Same as G1-i on the **last-50** network average (the converged
  regime). Reported with its CI; sign-consistency gating only.

**A1-G1 passes iff G1-i and G1-ii both hold.**

### A1-G2 — the attack-induced gap increment

Primary: `Δ_attack-gap = Δ_raw(B) − Δ_raw(A)` where
`Δ_raw = J_C^true − J_C^reported`.

* **G2-i.** positive in all 3 seeds, whole-run.
* **G2-ii.** reported, non-gating: `Δ_raw` itself for every condition and seed,
  so the absolute gap stays visible.

The absolute gap is **not** a success criterion. The clean baseline already has
`Δ_raw ≈ +6.5 … +7.7` (measured, seeds 0/1/2), so any gate on the absolute gap
would be passed by the clean system. **The old criterion
`reported ≤ d < true` is explicitly retired**: it fires in condition A.

### A1-G3 — per-agent increment

* **G3-i.** reported, non-gating: `J_C^i(B) − J_C^i(A)` for all six agents,
  every seed, with CIs and with the count of agents over `d` in each condition.
* **G3-ii.** gating: the sign of the per-agent increment agrees with the
  network-average increment for **at least 4 of 6 agents** in **all 3 seeds**.
  *Bar:* deliberately not 6 of 6. The six owners' `J_C` values correlate
  0.85–0.92 within a rollout, so this is ~1 effective test per seed, not 6;
  requiring unanimity would gate on rollout noise. Stated in advance.

### A1-G4 — the attack reached the dual, and only through the dual

* **G4-i.** gating: the §7.1 mechanism identity holds on 100% of cells.
* **G4-ii.** gating: the attack must *lower* the multiplier —
  `λ̄(B) < λ̄(A)` in all 3 seeds. An under-reporting attack that raised `λ`
  would mean the causal story is wrong even if true cost rose.
* **G4-iii.** reported: cumulative injected bias `K · B`, measured aggregate
  shift vs the predicted `−B/M`, λ mean/max/activation, residual, source
  spread, per-agent λ.

### A1-G5 — the secondary contrast: B vs D

`B − D` on network-average true cost, paired by seed.

* **G5-i.** reported with its 95% CI and per-seed signs. **Not gated at n=3**
  — see §11, which fixes in advance that no significance claim will be made for
  this contrast at three seeds regardless of what the numbers show.
* **G5-ii.** `D − A` reported the same way. If D−A is itself large, the
  "generic perturbation is inert" reading is unavailable and must not be used.
* **G5-iii.** reported for both contrasts: true cost, task return, aggregate
  reported cost, estimation gap, λ, violation behaviour.

### A1-G6 — learning health (the run must still be a run)

Inherited from G10-C, unchanged: no NaN/Inf; PPO KL median ≤ 0.0069 and
p95 ≤ 0.01548; λ saturation fraction < 0.05; entropy declines and stays finite
and positive; θ checksum distinct in 100% of rounds.

`G10-C-ii` (task-return gain ≥ +75) is **reported but not gated in conditions B
and D**: an attack that degrades learning is a legitimate outcome (Outcome D,
§13), not a broken run. Gating it would make Outcome D unreachable.

---

## 9. What A1 may and may not claim

A1 compares **persistent, directionally chosen under-reporting** against
**clean feedback** and against a **magnitude-matched generic perturbation**.

**Permitted:** "persistent directional under-reporting has a distinct effect
from a magnitude-matched generic perturbation."

**Forbidden:** "malicious corruption is worse than benign corruption." The
control is not an identical persistent negative bias carrying a different
semantic label — it is a zero-mean i.i.d. perturbation of the same per-round
magnitude (§5.3). A1 establishes nothing about an attacker's *intent*, and the
words "adversarial specificity" do not appear in its conclusions.

---

## 10. Reporting — mandatory for every condition and seed

**Task.** Task return; first 20 rounds; last 50 rounds; learning curve.

**Safety.** True cost; reported source cost; mechanism aggregate;
network-average true cost; **per-agent true cost, all six, never pooled away**;
violation rate; peak violation.

**Feedback.** λ mean; λ max; λ activation fraction; residual; source spread.

**Attack mechanism.** Attacked source value; the two uncorrupted source values;
measured aggregate shift; expected aggregate shift; cumulative injected bias.

**Diagnostics.** Source calibration against the withheld `R_ref = 120`
reference; aggregate bias; source variance; RNG integrity.

Violation rate is **descriptive only**. The paper's constraint is an
expected-cost (CMDP) constraint, and G10 §3 already fixed that no violation-rate
bar is imposed: at per-episode true-cost sd ≈ 9.9 against `d = 25`, a policy
whose expected cost sits at `d` violates on ~half its episodes as an arithmetic
consequence. Primary safety quantities are `E[J_C^i]` and the network average
(§6.3).

---

## 11. Statistics, multiplicity, and power — fixed before the data exists

### 11.1 The procedure

The design is paired by seed, so all analyses are paired.

The repository has a **pre-existing, code-enforced rule**: decision D6,
`safelie.analysis.stats.MIN_SEEDS_FOR_INFERENCE = 5`. `welch_t_test` *raises*
below five seeds, and `safelie.eval.protocol` supplies the sanctioned n=3
alternative — per-seed sign and ordering. This mirrors the paper's own §5.1
("five seeds per cell … Welch's t-test … Holm correction across the attack
conditions"). A1 does not relax it.

Therefore, at n = 3:

* **Primary evidence is per-seed sign and magnitude consistency** across the
  three paired differences, as gated in §8.
* The **paired mean difference and its 95% CI** are reported for every contrast
  as estimates.
* A **paired t-test is reported only as a clearly-labelled sensitivity
  analysis**, never as the primary claim, together with a paired nonparametric
  companion (the exact sign test, whose minimum attainable two-sided p at n=3
  is 0.25 — stated here so it is not mistaken for a null result).
* **No p-value from A1 at n = 3 is presented as establishing significance.**

If and when the seed count reaches 5, the family `{B−A, B−D, A−D}` on the
primary outcome is corrected by **Holm step-down**
(`safelie.analysis.stats.holm_correction`) — the paper's own philosophy — with
`B−A` designated primary in advance. There is no other family, and no test
outside it will be promoted after the fact.

### 11.2 Power — computed before running, from the committed pilot

From the five-seed pilot
(`results/runs/pilot_{A_clean,B_attack,D_benign}_seed{0..4}`, neural-source
architecture), paired differences on network-average true cost:

| contrast | metric | paired mean | paired sd | `d_z` |
|---|---|---|---|---|
| **B − A** | whole-run | **+2.98** | 0.69 | **+4.34** |
| **B − A** | last-50 | **+3.63** | 1.06 | **+3.42** |
| **B − D** | whole-run | +3.19 | 1.06 | +3.01 |
| **B − D** | last-50 | +3.00 | 2.71 | **+1.11** |
| D − A | whole-run | −0.21 | 1.14 | −0.18 |
| D − A | last-50 | +0.63 | 2.81 | +0.23 |

Minimum detectable paired effect, α = 0.05 two-sided:

| n | `d_z` for 80% power | `d_z` that is *just* significant |
|---|---|---|
| **3** | **3.26** | 2.48 |
| 4 | 2.13 | 1.59 |
| 5 | 1.68 | 1.24 |
| 6 | 1.44 | 1.05 |

### 11.3 The honest conclusion about three seeds

**Three seeds are adequate for the primary contrast B−A and inadequate for the
secondary contrast B−D.**

* B−A: pilot `d_z` = +3.42 … +4.34 against an n=3 requirement of 3.26. In cost
  units, n=3 needs a paired mean of `2.48 × sd`; at the pilot's B−A last-50 sd
  of 1.06 that is **+2.63 cost units**, and the pilot observed +3.63.
* B−D: pilot `d_z` = +1.11 on last-50. n=3 would need a paired mean of
  `2.48 × 2.71 = +6.73` cost units; the pilot observed +3.00. **Roughly n = 6
  would be required for 80% power at that effect size.**

This is stated **before** A1 runs, and the consequence is fixed here: the B−D
contrast is reported at n=3 as an estimate with a CI and per-seed signs, and
**no significance claim will be made for it at three seeds regardless of what
the numbers show.** Seeds 3 and 4 are not a contingency to be triggered by a
disappointing p-value; extending to five seeds is a decision to be taken on
this power analysis alone, before any A1 result is seen, or not taken at all.

Caveat on these numbers, also stated in advance: the pilot ran the pre-G9
**neural** source architecture, whose clean between-seed variance was much
larger than G10's (clean whole-run network-average sd 0.143 across the three
committed clean runs). The pilot's variances are the best available prior, not
a prediction; if A1's paired variance is smaller, n=3 goes further than the
table says, and if the source architecture attenuates the corruption, the
effect is smaller and n=3 goes less far.

---

## 12. Task performance and stealth

Predeclared **before** analysis.

**Equivalence margin `Δ_equiv = 15.3 task-return units`** = 10% of the clean
learning gain, which is `+153.15` averaged over the three committed clean runs
(+151.74, +152.37, +155.34; last-50 minus first-50). Rationale: the constrained
learner's entire achievement across 250 rounds is ~153 return units; a treatment
costing ≤10% of that has not collapsed task performance in any practically
meaningful sense.

* **Primary:** two one-sided tests (TOST) on the paired B−A difference in
  last-50 task return against ±15.3. Equivalence is claimed **only** if TOST
  establishes it. `p > 0.05` on a difference test is **never** reported as
  stealth.
* **Secondary, reported:** the paper's own literal criterion
  (`main_iclr.tex:298`), "return within one standard deviation of the no-attack
  baseline" — one clean between-seed sd = **1.36** return units. This is 11×
  tighter than `Δ_equiv`, and n=3 has essentially no power to establish
  equivalence at it; that is reported, not worked around.
* If equivalence cannot be established at `Δ_equiv`, A1 reports the observed
  task-performance effect and says so plainly.

---

## 13. Pre-defined interpretation — fixed before the data exists

* **Outcome A — strong attack evidence.** B reproducibly increases true safety
  cost relative to A (A1-G1 passes), the §7.1 mechanism identity holds, and the
  B−D estimate separates from zero in the same direction in all three seeds.
* **Outcome B — attack differs from clean but not from generic noise.** A1-G1
  passes but B−D does not separate. The perturbation matters; A1 provides
  insufficient evidence that directional persistence is a distinct mechanism.
* **Outcome C — no attack effect.** A1-G1 fails. The valid source architecture
  attenuates the corruption. **This is a legitimate scientific result and `B`
  will not be re-tuned after observing it.**
* **Outcome D — task return collapses.** Even if safety cost rises, the attack
  is then reported as **disruptive rather than stealthy**.

### Verdict rule

* **PASS** — every structural gate passes in all three seeds; A1-G1, A1-G2,
  A1-G3-ii and A1-G4 pass in all three seeds; and the B−D estimate is positive
  in all three seeds.
* **CONDITIONAL PASS** — every structural gate passes and A1-G1 passes, but one
  aspect is unresolved (task stealth, or B−D separation).
* **FAIL** — any structural gate fails, **or** A1-G1 fails, **or** A1-G4-i or
  A1-G4-ii fails.

---

## 14. Out of scope for A1

A1 establishes nothing about: RCE or any defense; Theorem 2's numerical bound;
topology generality; `f > 1`; `M = 5`; other environments; other attack
strengths (`B/d ∈ {0.25, 1.0}`); over-reporting; adaptive or Byzantine
adversaries; or baseline algorithms. The RCE experiment is not launched until
this report has been reviewed.
