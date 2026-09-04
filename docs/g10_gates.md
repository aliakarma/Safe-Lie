# G10 — clean three-seed validation of the integrated source architecture

**Status: pre-declared. Written and committed before seed 1 or seed 2 was
launched. Every threshold below is justified from the *already committed*
G0/G1/G2/G9 artifacts, and from the sampling law, and from nothing else.**

Seed 0 of G10 **is** the committed G9 run
(`results/runs_constraint_batch_g9/g9_batch_clean`, git
`73b1ae09473b3964a07c0d4347e425ff2896c5b2`). It is re-analysed by
`scripts/analyze_g10.py`, not re-run.

---

## 0. What G10 is asking

G9 answered "does the parallel trajectory-batch source architecture survive
integration into a live training loop?" in **one** run. G10 asks only whether
that answer is **reproducible across independent training seeds**. It adds no
architecture, tunes no hyperparameter, and runs no attack.

G10 keeps two questions strictly apart, and every gate below is labelled with
the one it belongs to:

| | Question | What it is about |
|---|---|---|
| **A** | Does the aggregate source estimator remain calibrated across seeds? | The **mechanism**. Answered by comparing the mechanism aggregate against a withheld high-precision reference collected **under the same pinned θ_k**, so it is independent of where the policy happens to sit. |
| **B** | Does the constrained learner behave consistently across seeds? | The **policy**. Answered by learning curves, dual dynamics, and the true-cost distribution. |

A calibrated source does not imply a safe policy, and a variable policy does
not imply a broken source. No gate below is allowed to be read as evidence for
the other question.

---

## 1. What is held fixed

The exact committed G9 implementation, at git
`73b1ae09473b3964a07c0d4347e425ff2896c5b2`. Unchanged: source architecture,
`M = 3`, `R_m = 30`, PPO, GAE, dual update, `η_λ = 0.035`, `λ_max = 25`,
budget `d = 25`, environment (`manyagent_ant`, N=6, `velocity_threshold=0.75`,
`cost_mode=per_agent_velocity`), ring topology, mean aggregation, 250 rounds,
`rollout_length = 2000`, the `R_ref = 120` withheld reference at rounds
{25, 75, 125, 175, 225}, and the worker/chunk scheduling knobs (which
`tests/unit/test_source_batch.py::TestWorkerCountIsNotAScientificParameter`
pins as value-invariant).

The only permitted differences are the training seed, the source RNG streams
derived from it (§2), and the output directory.

**`M` stays at 3.** G7/G8/G9 established M=3 as the operating point for
`f = 1`, `M ≥ 2f+1`. No M=5 analysis in G10.

---

## 2. Source RNG streams must be derived from the seed — and are not, by default

`SourceCollectionConfig.seed_entropy` (`src/safelie/utils/config.py:72`) is a
**fixed constant that does not depend on `cfg.seed`**, and the M replica
streams plus the reference stream are spawned from it alone
(`src/safelie/training/source_batch.py:331`). Re-running the G9 config with
`--seed 1` would therefore train a different policy while drawing the
**identical 23,100 source seeds** as seed 0.

That would invalidate G10 before it started: the source sampling noise would be
common-mode across all three seeds, and "the aggregate is calibrated in every
seed" would be one observation repeated three times rather than three
independent ones. The very excursions that made G9b fail (§4) would recur at
the same checkpoints in every seed, and would look like a reproducible defect
rather than the sampling noise they are.

G10 therefore fixes an explicit derivation, published here in advance:

```
seed 0 : seed_entropy = 286314957402113664887331205920951063913     # the G9 constant, verbatim
seed k : seed_entropy = int(sha256(b"safelie/g10/source-entropy/seed=<k>").digest()[:16])
```

giving

```
seed 1 : 52021175099534868945312500562751741008
seed 2 : 335268198726974212397240672597355197200
```

This changes **which seeds the architecture draws**, which is exactly what an
independent training seed is supposed to change. It does not touch the
architecture. It is set in the seed's config file, so the G9 implementation is
byte-for-byte unmodified.

`scripts/g10_derive_source_entropy.py` computes these and audits the streams
**offline, before any run**, by replaying the collector's exact draw order.
Pre-launch result (`results/g10_entropy_audit.json`):

* the offline replay reproduces all **22,500** replica env seeds recorded in
  G9 seed 0's `source_seeds.jsonl` **exactly** — which is what licenses
  trusting the same replay for seeds that have not run yet;
* every seed: 23,100 env seeds, 23,100 unique, 0 within-run duplicates, 0
  replica-vs-replica overlap, 0 replica-vs-reference overlap;
* every seed pair: **0 shared env seeds** (chance expectation 0.25), sequences
  not identical.

---

## 3. The clean-policy validity criterion, stated before it is measured

`paper/main_iclr.tex:104-109` defines

```
J_C^i(θ) = E_{π_θ}[ Σ_t γ^t C^i(s_t,a_t) ]      subject to   J_C^i(θ) ≤ d^i  for all i ∈ N
```

This is an **expected-cost (CMDP) constraint**, not a chance constraint. G10
does not silently convert one into the other. Concretely:

* **`E[J_C^i] ≤ d` is the constraint.** It is what the dual ascent drives and
  what the gates in §8 test.
* **`P(J_C^i > d)` is a reported quantity, not the constraint.** The same
  paper lists violation rate and peak violation as *metrics*
  (`main_iclr.tex:298`), and uses "true cost > d" only in the **attack**
  success criterion. G10 reports both and never calls the policy "safe" on the
  strength of a mean.
* **No violation-rate bar is imposed.** G9 seed 0 has per-episode true-cost
  sd ≈ 9.9 against d = 25. A policy whose expected cost sits exactly at d then
  violates on roughly half its episodes *as an arithmetic consequence*
  (observed: 0.473). Requiring "violation rate < 10%" would be requiring a
  chance constraint the algorithm never claimed to enforce, and would fail a
  correctly-behaving learner.

### 3.1 Per-agent versus network-average — declared in advance

The paper states a **per-agent** constraint but also notes
(`main_iclr.tex:109`) that "the shared-constraint case
`(1/N)Σ_i J_C^i ≤ d` is recovered by taking `W = (1/N)11ᵀ`".

The implemented dual is `λ_{k+1} = Π_[0,λmax][ W λ_k + η_λ(J̄_C − d) ]` with a
**doubly-stochastic ring `W`** applied to λ every round — verified against G9
seed 0's log to a reconstruction error of `1.8e-15`. Mixing λ over a doubly
stochastic `W` drives the multipliers toward consensus, so the fixed point
equalises multipliers rather than residuals, and what is actually enforced is
the **network-average** expected-cost constraint. G9 seed 0 shows exactly this:
last-50 network-average true cost 25.82 (95% CI [24.06, 27.59], contains d),
while per-agent expected costs stratify as

```
agent_0 23.36   agent_1 23.18   agent_2 22.57   agent_3 24.17   agent_4 28.15   agent_5 33.50
```

with agents 4 and 5 above `d` at 120-trajectory reference precision (26.81 and
31.51 at round 225, SE ≈ 0.6 — far outside sampling noise).

This ordering is **not a seed effect and not a source effect**: it reproduces
in every committed clean campaign, across three different source
architectures.

| campaign | last-50 per-agent true cost (0…5) | net-avg |
|---|---|---|
| G0 | 26.06 26.08 26.69 29.62 34.64 40.55 | 30.61 |
| G1 | 29.67 29.68 30.87 34.75 40.83 48.00 | 35.63 |
| G2 | 26.27 26.24 26.88 29.71 35.23 41.48 | 30.97 |
| G9 | 23.36 23.18 22.57 24.17 28.15 33.50 | 25.82 |

It is a property of the ManyAgent-Ant geometry and the ring topology.

**Therefore G10 gates the network-average expected cost (§8, G10-F) and
reports the per-agent expected costs, with CIs and with the count of agents
over `d`, as a first-class non-gating finding.** Gating per-agent would fail
G10 for a property of the decentralized algorithm that has nothing to do with
the source architecture G10 exists to validate. Hiding it would misstate what
"the clean learner is constraint-controlled" means.

**Pre-declared prediction:** the ordering `0 ≈ 1 ≈ 2 < 3 < 4 < 5` will
reproduce in seeds 1 and 2. If it does not, that is a finding and will be
reported as one.

---

## 4. Why the G9 conditional false-safe gate and the G9b source gate are demoted

Both are still **computed and reported verbatim** for all three seeds. Neither
is a primary G10 gate. The reasons are stated here, before seeds 1 and 2 exist.

### 4.1 Conditional false-safe rate — demoted to secondary (as instructed)

`P(Ĵ ≤ d | J_C > d)` is not comparable across policies whose true-cost
distributions sit at different distances from `d`. G9's own data shows the
mechanism: the conditional rate is 0.59 in the `|J_C − d| < 1` band and 0.25 in
the `> 6` band, and it ranges from 0.028 (agent_5, mean cost 34) to 0.775
(agent_0, mean cost 21.6) *within a single run*. The G9 bar of 0.353 was tied
to the G2 policy regime and is not invariant to the base rate. G10's primary
false-safe measure is the **joint** rate (§8, G10-E), always reported beside
`P(J_C > d)`.

### 4.2 G9b (per-source |z| ≤ 2) — demoted to secondary, on a multiplicity argument

G9b required all 3 individual sources within `|z| ≤ 2` of the reference for
≥5 of 6 owners in ≥4 of 5 checkpoints. It scored 3/5 and failed. That verdict
does not survive inspection:

1. **The six owners are not six tests.** All six owners' costs are read off
   the *same* 90 trajectories, so a source's excursion is one number reported
   six times. Round 125, source `batch_2`, z by owner:
   `+3.28 +3.21 +2.99 +3.12 +3.41 +2.87`. Round 225, source `batch_1`:
   `−2.69 −2.61 −2.83 −2.56 −2.54 −2.68`. (Consistent with the recorded
   0.85–0.92 within-rollout correlation of owner `J_C` values.)
2. **So a checkpoint is ~3 tests, not 18.** Under a perfectly calibrated
   source, `P(a checkpoint fails) = 1 − (1 − 0.0455)³ ≈ 0.130`, and
   `P(≥2 of 5 checkpoints fail) ≈ 0.129`.

G9's G9b outcome is therefore **~13% likely under a perfectly calibrated
source**. It is a gate with no multiplicity correction, not evidence of
miscalibration — a reading reinforced by the fact that both excursions were
single, *different*, non-repeating replicas (`batch_2` at 125, `batch_1` at
225), and that the **aggregate** stayed within 1.4 SE at both.

G10's primary calibration gate is at the **aggregate** level (§8, G10-A) —
which is also the only quantity that actually enters the dual update.

**This is a demotion with a stated statistical reason, decided before the new
data exists. It is not a threshold relaxed after seeing a failure: G9b is
still computed and its verdict still reported for every seed.**

---

## 5. Effective sample size — a standing correction applied to every gate

Because the six owners share one rollout, any statistic averaged over owners
has close to the standard error of **one** owner, not `1/√6` of it. G10 never
divides an owner-averaged standard error by `√6`, and flags any G9-inherited
interval that did. (G9f's bootstrap CI `[1.029, 1.144]` treated 1500
owner-cells as independent; with ~250 effective cells the interval is roughly
`±0.063` rather than `±0.057`, and the observed 1.085 is ~1.4 SE from 1.0, not
~3 SE. G10 reports the corrected reading.)

---

## 6. Runs

| seed | config | output |
|---|---|---|
| 0 | (already run as G9) | `results/runs_constraint_batch_g9/g9_batch_clean` |
| 1 | `configs/experiment/g10_batch_clean_seed1.yaml` | `results/runs_constraint_batch_g10/seed1` |
| 2 | `configs/experiment/g10_batch_clean_seed2.yaml` | `results/runs_constraint_batch_g10/seed2` |

Sequential, not concurrent: the machine has 12 logical CPUs and the source
collector already uses all 12: two concurrent runs would not finish sooner and
would prevent the §7 stop rule from being applied.

**Stop rule.** Seed 2 is launched **only** if seed 1 passes every structural
gate in §7. A structural failure in seed 1 halts G10; the implementation is
fixed and G10 restarts.

---

## 7. Structural gates — verified independently for each seed, never inherited

These are correctness checks, not statistics. **Every one must pass exactly.**
They are re-verified for seed 1 and seed 2 from that seed's own logs; nothing
is inferred from seed 0.

| id | check | how |
|---|---|---|
| **G10-S1** | 3 sources per owner per round; `M=3`, `R_m=30`, 90 trajectories, 250 rounds | `rounds.jsonl.source_batch`, and source ids agree with the per-owner report set (G9a, verbatim) |
| **G10-S2** | every source replica ran the pinned θ_k | `worker_checksums_all_match` for all 48 chunks each round; `collect()` raises otherwise |
| **G10-S3** | θ_k unchanged by collection, and 250 distinct θ checksums | `theta_k_checksum_stable_through_dual`, distinct-checksum count (G9h, verbatim) |
| **G10-S4** | source RNG streams distinct; no trajectory reused | 23,100 unique env and torch seeds, 0 duplicates, 0 replica-pairwise overlap, 0 replica-vs-reference overlap (G9g, verbatim) |
| **G10-S5** | source streams disjoint **across seeds** | 0 shared env seeds between this seed and every other seed's recorded log (new in G10; pre-verified offline in §2) |
| **G10-S6** | source trajectories never enter PPO | PPO env steps exactly `250 × 2000 = 500,000` and independent of source steps; `BatchSourceResult` carries only scalars and per-trajectory floats — no transitions; enforced statically by `tests/unit/test_source_batch.py::TestNoNeuralPathInTheBatchSourcePipeline` and `::TestPpoIsUnaffected` |
| **G10-S7** | no oracle leakage | `tests/isolation/test_oracle_isolation.py` (greps the training package for oracle handles); plus no learner-visible field in `rounds.jsonl` equals the round's oracle `true_cost_return` |
| **G10-S8** | withheld reference never reaches the dual | reference appears only in `validation_reference.jsonl`; `reference_mean` absent from `rounds.jsonl` (`::TestValidationReferenceIsWithheld`) |
| **G10-S9** | dual arithmetic exact, mixing included | (a) `λ_mixed_before = W λ_{k−1}` for the ring `W`, max abs error ≤ 1e-9; (b) `λ_after − λ_mixed_before = η_λ · residual` on every unclipped cell, max abs error ≤ 1e-9 |
| **G10-S10** | checkpoint/resume correctness | a dedicated short-config resume test: run `n` rounds, checkpoint, resume, and require the resumed round's source seeds, source means and policy checksum to be **identical** to the uninterrupted run's |

---

## 8. Scientific gates

Each is evaluated per seed. `d = 25`.

### G10-A — aggregate calibration *(Question A: the source)*

The mechanism aggregate versus the withheld `R_ref = 120` reference, both under
the same pinned θ_k, at the 5 checkpoints. `z̄_k` is the **owner-mean** of
`z_agg = (aggregate − reference) / sqrt(SE_agg² + SE_ref²)`; `b̄_k` is the
owner-mean aggregate bias in cost units.

* **A-i.** `|z̄_k| ≤ 3.0` at **all 5** checkpoints.
  *Bar:* with 5 checkpoints × 3 seeds ≈ 15 near-independent tests, a 3σ
  per-test bar gives a family-wise false-alarm rate of ≈ 15 × 0.0027 = **4%**.
  A 2σ bar would give ≈ 51% — which is precisely how G9b failed (§4.2).
* **A-ii.** the checkpoint-pooled bias `|mean_k b̄_k| ≤ 1.0` cost units.
  *Bar:* 1.0 is **4% of `d`**, an order of magnitude below the −5.85 / −6.10 /
  −7.10 biases of the G1 / G0 / G2 neural-source campaigns, and small enough
  that it cannot by itself flip the constraint at any material margin. On the
  statistics it is ≈ 2.4 SE of the pooled estimate (per-checkpoint SE ≈ 0.94,
  pooled over 5 ≈ 0.42), so noise alone trips it ≈ 1.7% of the time per seed.
  Pooling across checkpoints, rather than gating each one, is deliberate: a
  per-checkpoint 1.0 bar would re-commit G9b's error, since G9's own
  checkpoint biases (+0.15, −0.78, +1.27, −0.93, −1.28) are alternating-sign
  sampling noise averaging to **−0.31**.

**Seed passes G10-A iff A-i and A-ii both hold.**
**Meta:** G10-A passes iff all three seeds pass **and** the 3-seed pooled bias
satisfies `|b̄_all| ≤ 0.6` (≈ 2.5 SE of the 15-checkpoint pooled estimate),
reported with its 95% CI.

*Seed 0 (already known):* `z̄` = +0.17, −0.86, +1.24, −0.91, −1.36 → A-i ✓;
pooled bias −0.31 → A-ii ✓.

### G10-B — source precision follows the sampling law *(Question A)*

* **B-i.** the between-source variance ratio
  `mean(s²_between) / mean(σ̂²/R_m)` lies in **[0.80, 1.25]** (G9f's band,
  verbatim).
* **B-ii.** reported, non-gating: observed `SE_agg` versus the predicted
  `SE_m/√3`, per seed and pooled, with the §5 effective-sample-size
  correction applied to any interval.

*Seed 0:* ratio 1.085 ✓; `SE_agg` observed 0.738 vs predicted 0.709 (ratio
1.042).

### G10-C — learning health *(Question B: the policy)*

* **C-i.** no NaN/Inf in mechanism, residual, λ, KL, entropy, true cost, spread.
* **C-ii.** `mean(task return, last 50) − mean(task return, first 50) ≥ +75`.
  *Bar:* every committed clean campaign delivered +128.6 (G0), +150.1 (G1),
  +139.5 (G2), +151.7 (G9). +75 is cleared by ≥1.7× by all of them and is not
  cleared by a run that fails to learn.
* **C-iii.** PPO KL median ≤ 0.0069 and p95 ≤ 0.01548 (3× the committed G2
  clean reference — G9c's band, verbatim).
* **C-iv.** λ saturation: fraction of cells at `λ_max` < 0.05.
* **C-v.** entropy declines (last-10 mean < first-10 mean) and stays finite and
  positive; θ checksum distinct in 100% of rounds.

*Seed 0:* +151.7; KL 0.00231 / 0.00538; saturation 0.0; entropy 3.65 → 1.61 ✓.

### G10-D — dual dynamics *(Question B)*

* **D-i.** the §7 G10-S9 arithmetic (gating there, restated here for reading).
* **D-ii.** sign: `Δλ > 0` wherever residual > 0 and `Δλ < 0` wherever
  residual < 0, on every unclipped cell.
* **D-iii.** responsive: `frac(λ > 0) ≥ 0.10` and `λ_max < λ_max_config = 25`.
* **D-iv.** reported, non-gating: residual→Δλ relationship, Δλ sign-flip rate
  against the 0.5 pure-noise reference, λ–true-cost correlation at lags
  {0,1,5,10}, per-agent λ means, λ cross-agent spread.

*Seed 0:* wiring error 8.7e-16; frac(λ>0) 0.925; λ_max 9.33; flip rate 0.112 ✓.

**No exact numeric agreement with G9 is required by any part of G10-D** —
only that the feedback stays correctly wired and stable.

### G10-E — joint false-safe rate *(system-level: A **and** B together)*

* **E-i.** the 95% Wilson **upper** bound on
  `FSR_joint = P(Ĵ_mech ≤ d ∧ J_C > d)` is **< 0.2433** — the joint rate of
  G1, the *best* of the three committed pre-G9 neural-source campaigns
  (G0 0.2640, G1 0.2433, G2 0.2673). Pre-declared from committed baselines,
  not from G9.
* **E-ii.** `P(J_C > d)` is reported beside it in every table. The joint rate
  is uninterpretable without the base rate.
* **E-iii.** declared in advance: if E-i fails in any seed, the rate is
  decomposed as `joint = P(Ĵ ≤ d | J > d) × P(J > d)` and the report states
  **which factor moved**, so a base-rate-driven failure is attributed to the
  policy (Question B) and not to the source (Question A).
* **E-iv.** the conditional rate `P(Ĵ ≤ d | J > d)` with its Wilson CI is
  reported as a **secondary diagnostic only** (§4.1), stratified by owner, by
  training phase, and by `|J_C − d|` margin band.

*Seed 0:* joint 0.1753, CI [0.1569, 0.1954] → ✓; base rate 0.4733; conditional
0.3704.

### G10-F — constraint control *(Question B)*

The constraint under test is the expected-cost constraint of §3, enforced at
the network average per §3.1.

* **F-i.** network-average true cost over the last 50 rounds ≤ **28.0**
  (= `d + 3`).
  *Bar:* `d + 3` is 12% of `d`, ≈ 3.3 SE of the last-50 network-average
  estimate (SE ≈ 0.90), and below the tightest pre-G9 campaign (G0 at 30.61),
  so it discriminates the new architecture rather than rubber-stamping it.
  G1 (35.63), G0 (30.61) and G2 (30.97) all fail it.
* **F-ii.** no upward divergence: last-50 network average minus the rounds
  51–150 network average ≤ **+3.0**.
* **F-iii.** reported, non-gating, per §3.1 and the task's §15: mean, median,
  sd, quantiles, violation rate, peak violation, first- vs second-half,
  50-round blocks, per-agent expected cost **with 95% CIs**, per-agent
  violation rate, the count of agents whose expected cost exceeds `d`, and the
  `R_ref = 120` reference estimate of `J_C^i(θ_k)` at every checkpoint.

*Seed 0:* last-50 net-average 25.82 ✓; rounds 51–150 24.88 → divergence +0.94 ✓;
2 of 6 agents over `d` in expectation (reported, not gated).

---

## 9. Three-seed meta-result

G10's question is whether the clean architecture behaves consistently enough
that **an attack experiment would be interpretable**. All three seeds are used;
none is selected or dropped.

| | meta-criterion |
|---|---|
| Source calibration | G10-A passes in all three seeds and the 3-seed pooled bias meets the meta bar |
| Source precision | G10-B-i holds in all three seeds |
| Learning | G10-C passes in all three seeds |
| Dual | G10-D and G10-S9 pass in all three seeds |
| Constraint | G10-F passes in all three seeds |
| Structure | every §7 gate passes in all three seeds |
| Reproducibility | no seed contradicts the architecture qualitatively |

**Verdict rule, fixed in advance:**

* **PASS** — every §7 structural gate and every §8 scientific gate passes in
  all three seeds, and the §9 meta bar holds.
* **CONDITIONAL PASS** — every structural gate passes and no seed contradicts
  the architecture, but at most one scientific gate misses in at most one seed,
  and the miss is attributable and does not concern aggregate calibration
  (G10-A) or structural correctness. Exactly one issue to fix is then named.
* **FAIL** — any structural gate fails in any seed, **or** G10-A fails in any
  seed, **or** two or more scientific gates miss, **or** a seed behaves in a
  way that contradicts the architecture.

"Do not require identical trajectories" is honoured: nothing above compares a
G10 point estimate to a G9 point estimate for equality.

---

## 10. Computational cost — recorded, not hidden

Per run, from the G9 seed-0 measurement: **45,000,000 source environment
steps** (180,000 per dual update, 90× the PPO rollout, 46× the total
environment interaction of the G2 configuration), 500,000 PPO steps, 500,000
oracle steps, and ≈ **10.1 hours** wall clock on the 12-CPU run machine
(36,390 s; 30,588 s of it source collection, 84%). Two new seeds ≈ **20 hours**
sequential.

`env_steps`, `wall_clock_s`, `source_s`, `oracle_s`, `s_per_round`, workers and
chunk count are recorded per run in `run_metadata.json` and reproduced in the
G10 report. For a publication this cost is a material practical limitation of
the architecture and is reported as one.

---

## 11. Out of scope for G10

G10 runs **no** attack, benign perturbation, RCE, over-report, Byzantine or
adaptive-attacker condition; no topology, β, `f`, defense-ablation, baseline or
second-environment experiment; and no `M = 5`. G10 establishes nothing about
attack effectiveness, stealth, Theorem 2's numerical bound, other environments,
other `f`, or other topologies. It is clean replication only, and the attack
study is not launched until this report has been reviewed.
