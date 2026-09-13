# Submission evidence sheet

**Not prose. This is the numbers sheet the manuscript is written from.**
Every value here is traceable to a committed artifact named in the
section. Nothing is projected, rounded up, or reconstructed from memory.

Compiled 2026-09-12. Statistical policy: n = 3, paired by seed, effect
sizes and paired 95 % CIs, per-seed values always shown. Decision D6
(`safelie.analysis.stats.MIN_SEEDS_FOR_INFERENCE = 5`) forbids
significance inference below five seeds, so every paired t in the source
reports is labelled SENSITIVITY ONLY and none is quoted as a p-value in
the paper.

**Metric convention.** `J_C` is the whole-run network-average true cost
(higher is worse), budget d = 25.0. "Detection gap" in this sheet always
means the **mechanism-level** gap
`Δ = true_cost_return − mechanism_reported_cost_return`
(the logged field `detection_gap_vs_aggregate`), not the own-critic gap
`detection_gap` that `analyze_a{1,2}.py` summarise. See §A2-R below.

---

## 0. The metric correction (zero-compute reanalysis)

`scripts/analyze_a{1,2}.py` summarise `oracle.jsonl:detection_gap`, which
`experiment.py` defines as `true_cost − own_critic_reported` and whose own
comment says it "measures critic estimation error, not the attack". The
quantity the paper studies is how far the value the **constraint
mechanism** consumes sits below the truth. Both are logged on every oracle
row of every committed run; only the wrong one was analysed.

Recomputed by `scripts/reanalyze_mechanism_gap.py` →
`results/mechanism_gap_reanalysis.json`. No run was repeated and no log
was modified.

**Verification of the recomputation** (all runs, all rounds, all agents):

| check | result |
|---|---|
| `detection_gap_vs_aggregate == true_cost − mechanism_reported` | max abs error **0.0** |
| `oracle.jsonl` mechanism value == `rounds.jsonl` value fed to the dual | max abs error **0.0** |
| `mean(gap) == mean(true_cost) − mean(mech_reported)` | max abs error **5.0e-15** |
| all finite | yes |

**Why it matters.** Clean runs carry an own-critic gap of ≈ +7.5, because
the GAE cost critic reads far below the truth. That offset is present in
every condition and swamps the attack. The mechanism gap is ≈ +0.21 when
clean, so the same attack shows as a ~20× separation rather than a 13 %
increment.

| contrast | own-critic gap (superseded) | **mechanism gap (use this)** |
|---|---|---|
| B−A | +0.97 | **+4.2168** |
| D−A | ≈ 0 | **−0.0157** |
| (C−E) − (B−A) | −0.31 | **−3.5022** |

---

## G10 — Instrument Validation

**What it is.** Three clean constrained runs establishing that the
measuring instrument works before any attack is applied: that the
parallel trajectory-batch source architecture is calibrated, that the
dual is wired correctly, and that the learner learns.

**Artifacts.** `results/runs_constraint_batch_g10/g10_report.json`;
runs `runs_constraint_batch_g9/g9_batch_clean` (seed 0),
`runs_constraint_batch_g10/seed{1,2}`. Gates: `docs/g10_gates.md`.

**Verdict: PASS.** All six scientific gates (G10-A..F) and all structural
checks (S1–S9) pass on all three seeds. `scientific_misses: []`.

### Per-seed values

| | seed 0 | seed 1 | seed 2 | mean (sd) |
|---|---|---|---|---|
| `J_C` whole-run | 25.2143 | 24.9550 | 24.9782 | **25.0492** (0.1435) |
| `J_C` final-50 | 25.8218 | 26.2197 | 24.5748 | 25.5388 (0.8582) |
| mech. reported cost | 24.7660 | 24.9694 | 24.7851 | 24.8402 (0.1124) |
| mechanism gap Δ | +0.4483 | −0.0144 | +0.1932 | **+0.2090** (0.2318) |
| task return, first-20 | −181.9909 | −180.7243 | −174.6102 | −179.1084 (3.9468) |
| task return, last-50 | −44.1922 | −42.5286 | −45.2154 | −43.9787 (1.3561) |
| learning gain | 151.7374 | 152.3701 | 155.3443 | 153.1506 (1.9260) |
| violation rate, whole | 0.4733 | 0.4607 | 0.4647 | 0.4662 (0.0065) |
| λ mean | 1.5097 | 1.6000 | 2.1777 | 1.7625 (0.3624) |
| λ max reached | 9.3282 | 9.3558 | 12.7492 | — (cap is 25.0) |
| λ fraction saturated | 0.0 | 0.0 | 0.0 | **0.0** |
| agents above d, final-50 | 2 / 6 | 3 / 6 | 2 / 6 | — |
| wall clock (s) | 36,390 | 35,058 | 27,211 | — |

### Key gate values (seed 0 shown; all three pass)

- **G10-A (source calibration).** Per-checkpoint |z| max 1.361, bar 3.0.
  Pooled bias −0.3126, 95 % CI [−1.2147, +0.5895], bar 1.0.
  Cross-seed meta-pooled bias **+0.1083**, 95 % CI [−0.3177, +0.5344],
  bar 0.6, over 15 checkpoints.
- **G10-B (variance).** Between-cell variance ratio 1.0851, band
  [0.80, 1.25]. Observed/predicted aggregate s.e. 1.0417.
- **G10-C (learning health).** Non-finite counts all 0. Task return by
  blocks of 50: −195.93 → −158.61 → −117.48 → −82.22 → −44.19
  (Δ = 151.74, bar 75.0). PPO KL median 0.00231 (bar 0.0069), p95
  0.00538 (bar 0.01548). λ saturation 0.0 (bar 0.05).
- **G10-D (dual wiring).** Consensus mixing reproduces the ring W to
  1.78e-15; dual ascent reproduces to 8.74e-16 over 1,388 unclipped
  cells. λ positive on 92.5 % of cells.
- **G10-E (false-safe rate).** Joint false-safe rate 0.17533, Wilson CI
  [0.15693, 0.19540], bar: upper limit below 0.2433. Base violation rate
  0.47333, Wilson CI [0.44817, 0.49864].
- **G10-F (cost level).** Final-50 network average 25.8218, 95 % CI
  [24.0560, 27.5875], bar 28.0. No divergence: final-50 minus rounds
  51–150 = +0.9453, bar 3.0.
- **S5 (seed independence).** Pairwise shared env seeds across the three
  seeds: **0, 0, 0** of 22,500 (chance expectation 0.236).

### What G10 establishes

That the instrument is sound: sources calibrated, dual arithmetic exact,
learner learning, seeds independent. It also fixes the operating point
the rest of the paper sits at — **the clean constrained system runs at
J_C = 25.05 against d = 25.0**, i.e. essentially exactly on the budget.

### What G10 does NOT establish

Nothing about attacks, defenses, or the constraint's causal role. It is
calibration, not a result. It also does **not** show per-agent safety:
2–3 of 6 agents sit above d in every clean seed, and the violation rate
is ~47 % of rounds. Only the network average is controlled — which is
Proposition 1, not a defect.

---

## A1 — Persistent Directional Corruption

**What it is.** Condition B — one of three trajectory-batch sources
under-reports by a constant B = 12.5 = 0.5 d on every round, persistently,
statically, consistently — against the paired clean condition A.

**Artifacts.** `results/runs_a1/a1_report.json`, runs
`results/runs_a1/B_seed{0,1,2}`. Gates: `docs/a1_attack_gates.md`.
Mechanism gap from `results/mechanism_gap_reanalysis.json`.

**Design.** Paired by seed under common random numbers (identical source
`seed_entropy` per seed). Attacked source rotates by seed:
seed 0 → `batch_1`, seed 1 → `batch_2`, seed 2 → `batch_3`.

**Verdict: PASS.** A1-G1, G2, G3, G4, G6 all pass. G5 (B−D) is reported
as a descriptive estimate only, by pre-declaration.

### Per-seed condition values

| | A_s0 | A_s1 | A_s2 | B_s0 | B_s1 | B_s2 |
|---|---|---|---|---|---|---|
| `J_C` whole | 25.2143 | 24.9550 | 24.9782 | 29.0559 | 29.2728 | 28.7761 |
| `J_C` final-50 | 25.8218 | 26.2197 | 24.5748 | 29.8166 | 28.6216 | 28.4191 |
| mech. reported | 24.7660 | 24.9694 | 24.7851 | 24.4921 | 24.7697 | 24.5656 |
| mechanism gap Δ | +0.4483 | −0.0144 | +0.1932 | +4.5638 | +4.5031 | +4.2105 |
| task return last-50 | −44.1922 | −42.5286 | −45.2154 | −41.7024 | −44.8224 | −43.8197 |
| violation rate | 0.4733 | 0.4607 | 0.4647 | 0.6573 | 0.6360 | 0.6147 |
| λ mean | 1.5097 | 1.6000 | 2.1777 | 1.0346 | 1.2968 | 1.6891 |
| agents above d, final-50 | 2 | 3 | 2 | **6** | **6** | **5** |

Condition means: A `J_C` = 25.0492 (0.1435); B `J_C` = 29.0349 (0.2490).

### Primary contrast B − A (paired, n = 3)

| metric | per seed (0, 1, 2) | mean | sd | 95 % CI | dz | signs |
|---|---|---|---|---|---|---|
| **`J_C` whole** | +3.8416, +4.3178, +3.7979 | **+3.9858** | 0.2884 | **[+3.2693, +4.7022]** | +13.82 | 3/3 + |
| `J_C` final-50 | +3.9948, +2.4018, +3.8443 | +3.4136 | 0.8795 | [+1.2288, +5.5985] | +3.88 | 3/3 + |
| **mechanism gap Δ** | +4.1155, +4.5175, +4.0173 | **+4.2168** | 0.2650 | **[+3.5584, +4.8752]** | +15.91 | 3/3 + |
| mech. reported cost | −0.2739, −0.1997, −0.2195 | −0.2310 | 0.0498 | [−0.3265, −0.1356] | −4.64 | 3/3 − |
| violation rate whole | +0.1840, +0.1753, +0.1500 | +0.1698 | 0.0177 | [+0.1259, +0.2137] | +9.61 | 3/3 + |
| λ mean | −0.4752, −0.3033, −0.4886 | −0.4223 | 0.1033 | [−0.6790, −0.1657] | −4.09 | 3/3 − |
| task return last-50 | +2.4899, −2.2939, +1.3957 | +0.5306 | 2.5065 | [−5.6958, +6.7570] | +0.21 | **mixed** |

### Gates

- **A1-G1** (attack raises true cost): all 3 seeds positive whole-run and
  final-50; paired mean +3.9858 > bar 1.0. **PASS**
- **A1-G2** (stealth): mechanism gap positive in all 3 seeds. **PASS**
- **A1-G3** (per-agent): the per-agent increment agrees with the network
  sign in **6 of 6 agents in all three seeds** (bar: ≥ 4 of 6). Seed 0
  increments: +3.06, +3.05, +3.39, +4.08, +4.42, +5.04. Seed 1:
  +4.32, +4.37, +4.23, +4.22, +4.39, +4.37. Seed 2: +3.60, +3.57, +3.35,
  +3.53, +3.99, +4.75. **PASS**
- **A1-G4** (mechanism identity): the §7.1 constant identity
  `mean(pre-hook sources) − post-hook aggregate = B/M = 4.166667` holds on
  **100 % of cells** of every B run, max abs error **9.77e-15**
  (tolerance 1e-9); the clean identity (= 0) holds on 100 % of cells of
  every A run at max abs error **0.0**. The attack also lowers λ in all
  3 seeds. **PASS**
- **A1-G6** (learning health): finite, λ saturation 0, KL within the G10
  bars, entropy declines, on all runs. **PASS**

### Stealth (task return equivalence)

Pre-declared margin δ = 15.3 task-return units (§12):
TOST p = 0.0047, **equivalent**, 90 % CI [−3.695, +4.756].
Against the secondary paper criterion δ = 1.36 (one clean between-seed
sd): p = 0.312, **not equivalent** — report both, and never report the
difference test's p = 0.749 as evidence of stealth.

### What A1 establishes

At this operating point, a persistent directional under-report on one of
three sources raises the true network-average cost by **+3.99**
(95 % CI [+3.27, +4.70]), pushes 5–6 of 6 agents above budget (from 2–3),
raises the violation rate by **+17.0 points**, and does so while the
**reported** cost the operator sees goes *down* by 0.23 and the task
return is statistically indistinguishable. The mechanism gap opens from
+0.21 to +4.42.

### What A1 does NOT establish

- Anything at f > 1, M ≠ 3, another topology, another budget, another
  environment, or another aggregator.
- Anything about adaptive or Byzantine adversaries — this attack is
  static and consistent by construction.
- Significance in the conventional sense. n = 3; the evidence is effect
  size plus 3/3 sign consistency.
- That the attack is undetectable by *any* monitor — only that it is
  invisible to one watching reported cost and task return.

---

## A1 — Zero-Mean Control (condition D)

**What it is.** The pre-registered falsification control. Condition D
replaces the adversary with a zero-mean i.i.d. Gaussian perturbation of
the same scale (σ = 0.5 d = 12.5) on the same source. If a
same-magnitude but undirected perturbation produced the same effect, the
phenomenon would be ordinary estimation noise and the adversarial framing
would be unnecessary.

**Artifacts.** `results/runs_a1/D_seed{0,1,2}`, gates A1-G5 in
`a1_report.json`.

**Status: descriptive by pre-declaration.** The 2026-09-05 power analysis
(before any A1 run existed) showed B−D at dz = +1.11 on final-50 needs
about n = 6; three seeds carry B−A and do not carry B−D. B−D is therefore
permanently an estimate with a CI and three signs, with no significance
claim, and **no seed may be added to A1 to change that** — doing so would
be exactly the data-dependent inflation the pre-registration forbids.

**Mechanism check on D** (A1-G4, the distribution check rather than a
constant identity, since D is a fresh zero-mean draw per cell): implied
per-source ε has mean +0.1272 / +0.1598 / −0.3735 against a 4-s.e.
tolerance, and sd 12.849 / 12.250 / 12.159 against the declared
σ = 12.5 (ratios 1.028 / 0.980 / 0.973, band [0.85, 1.15]). The
perturbation is live on every cell. **PASS** — D really is a same-scale,
zero-mean perturbation, which is what makes it a control.

### D − A (the control contrast, paired, n = 3)

| metric | per seed (0, 1, 2) | mean | 95 % CI | signs |
|---|---|---|---|---|
| **`J_C` whole** | +0.2511, −0.4961, −0.8475 | **−0.3642** | [−1.7579, +1.0296] | **mixed** |
| `J_C` final-50 | −1.2912, −0.0101, −2.1915 | −1.1643 | [−3.8875, +1.5589] | 3/3 − |
| **mechanism gap Δ** | +0.4737, −0.0700, −0.4508 | **−0.0157** | [−1.1700, +1.1385] | **mixed** |
| violation rate whole | −0.0013, −0.0407, −0.0593 | −0.0338 | [−0.1073, +0.0398] | 3/3 − |
| λ mean | −0.0700, −0.1170, −0.1858 | −0.1243 | [−0.2690, +0.0204] | 3/3 − |
| task return last-50 | −2.7120, −5.2281, −6.1764 | −4.7055 | [−9.1529, −0.2581] | 3/3 − |

### B − D (descriptive, n = 3)

| metric | per seed (0, 1, 2) | mean | 95 % CI | signs |
|---|---|---|---|---|
| `J_C` whole | +3.5905, +4.8139, +4.6454 | **+4.3499** | [+2.7029, +5.9970] | 3/3 + |
| `J_C` final-50 | +5.2861, +2.4119, +6.0358 | +4.5779 | [−0.1740, +9.3299] | 3/3 + |
| mechanism gap Δ | +3.6418, +4.5876, +4.4682 | +4.2325 | [+2.9531, +5.5119] | 3/3 + |
| violation rate whole | +0.1853, +0.2160, +0.2093 | +0.2036 | [+0.1635, +0.2436] | 3/3 + |

Condition D mean `J_C` = 24.6850 (0.6955), vs A 25.0492 and B 29.0349.

### What the control establishes

A zero-mean perturbation of the **same magnitude** on the **same source**
moves the true cost by −0.36 with a CI straddling zero and mixed signs,
and moves the mechanism gap by −0.02 with mixed signs. The directional
attack moves them by +3.99 and +4.22 with 3/3 signs. **Direction, not
magnitude, is what makes the corruption bite** — which is the prediction
Theorem 1 makes, since `1ᵀδ` is what is conserved and a zero-mean δ has
`E[1ᵀδ] = 0`.

### What it does NOT establish

- It is a *null* result on D−A, reported descriptively. At n = 3 we
  cannot exclude a small true D−A effect; the claim is about the
  **contrast in magnitude** between B−A and D−A, not that D−A is exactly
  zero.
- D's task return is 4.7 units lower than A's with 3/3 signs — the noise
  is not free, it just is not a *safety* attack. Report this.
- Nothing about a benign *persistent* bias (condition D′), which was not
  run.

---

## A2 — RCE Factorial Defense

**What it is.** The 2×2 attack × defense factorial.
A = clean/no-RCE, B = attack/no-RCE, E = clean/RCE, C = attack/RCE.
Primary quantity: the **interaction** (C−E) − (B−A) — how much of the
attack's effect survives when the aggregator is robust.

**Artifacts.** `results/runs_a2/a2_report.json`,
`results/runs_a2/{C,E}_seed{0,1,2}`. Gates: `docs/a2_rce_gates.md`.

**Verdict: PASS — DEFENSE SUPPORTED.** All six gates pass. Sign
convention asserted at run time against A1's established B−A = 3.9858
(agrees to 1e-3).

### Per-seed condition values

| | C_s0 | C_s1 | C_s2 | E_s0 | E_s1 | E_s2 |
|---|---|---|---|---|---|---|
| `J_C` whole | 26.4550 | 25.8098 | 25.5997 | 25.6805 | 25.3636 | 24.4701 |
| mech. reported | 24.6507 | 24.9599 | 24.8041 | 24.7137 | 24.8136 | 24.6809 |
| mechanism gap Δ | +1.8043 | +0.8499 | +0.7956 | +0.9668 | +0.5500 | −0.2108 |
| task return last-50 | −39.4598 | −47.2753 | −45.4682 | −39.3810 | −49.0706 | −48.5858 |
| violation rate | 0.5260 | 0.4813 | 0.5000 | 0.4873 | 0.4760 | 0.4387 |
| agents above d, final-50 | 3 | 3 | 2 | 2 | 2 | 2 |

Condition means: C `J_C` = 25.9548 (0.4457); E `J_C` = 25.1714 (0.6277).

### The four contrasts on `J_C` whole-run (paired, n = 3)

| contrast | per seed (0, 1, 2) | mean | 95 % CI | dz | signs |
|---|---|---|---|---|---|
| B − A (attack, no defense) | +3.8416, +4.3178, +3.7979 | **+3.9858** | [+3.2693, +4.7022] | +13.82 | 3/3 + |
| **C − E (attack, with RCE)** | +0.7745, +0.4462, +1.1296 | **+0.7834** | [−0.0655, +1.6324] | +2.29 | 3/3 + |
| E − A (RCE's own cost, clean) | +0.4662, +0.4085, −0.5081 | +0.1222 | [−1.2357, +1.4802] | +0.22 | **mixed** |
| C − B (raw RCE effect) | −2.6009, −3.4630, −3.1764 | −3.0801 | [−4.1708, −1.9894] | −7.01 | 3/3 − |

### PRIMARY: interaction (C−E) − (B−A)

| metric | per seed (0, 1, 2) | mean | sd | 95 % CI | dz | signs |
|---|---|---|---|---|---|---|
| **`J_C` whole** | −3.0671, −3.8716, −2.6683 | **−3.2023** | 0.6129 | **[−4.7250, −1.6797]** | −5.22 | **3/3 −** |
| `J_C` final-50 | −2.8054, −1.5015, −3.2164 | −2.5078 | 0.8953 | [−4.7319, −0.2836] | −2.80 | 3/3 − |
| **mechanism gap Δ** | −3.2781, −4.2176, −3.0109 | **−3.5022** | 0.6338 | **[−5.0766, −1.9278]** | −5.53 | **3/3 −** |
| mech. gap Δ final-50 | −3.1535, −1.7898, −3.5491 | −2.8308 | 0.9230 | [−5.1236, −0.5379] | −3.07 | 3/3 − |
| task return whole | −7.1873, +3.2734, −3.0584 | −2.3241 | 5.2689 | [−15.4127, +10.7645] | −0.44 | mixed |

**Fraction of the attack effect removed: 80.3 % on `J_C`
(1 − 0.7834/3.9858), 83.1 % on the mechanism gap
(1 − 0.7146/4.2168).** Gate A2-G5 required the interaction negative in
all 3 seeds and (C−E) ≤ 0.5 × (B−A). Both hold.

### A2-R — the mechanism-gap reanalysis of the same runs

Recomputed on `detection_gap_vs_aggregate`
(`results/mechanism_gap_reanalysis.json`):

| contrast | per seed (0, 1, 2) | mean | 95 % CI | signs |
|---|---|---|---|---|
| C − E | +0.8375, +0.3000, +1.0064 | **+0.7146** | [−0.2018, +1.6310] | 3/3 + |
| E − A | +0.5185, +0.5644, −0.4039 | **+0.2263** | [−1.1308, +1.5834] | mixed |
| C − B | −2.7596, −3.6532, −3.4149 | **−3.2759** | [−4.4254, −2.1263] | 3/3 − |

### What the defended aggregator actually is (critical for scope)

**Verified from the logs, not asserted.** At the tested operating point
M = 3, f = 1:

- `M − 2f = 1`, so `retained_n = 1` in **every** cell of every C and E
  run — the trimmed set holds exactly one value.
- The MAD of a single value is 0, so `spread` is pinned at the floor
  `sigma_min = 1e-3` in **every** cell, and the applied margin is
  `β·σ = 1.5 × 0.001 = 0.0015` cost units — **0.006 % of d = 25**. The
  margin does no work.
- The point estimate equals the **median of the three post-attack
  reports**, bit for bit: max abs error **0.000e+00** over all
  250 rounds × 6 owners × 3 seeds, in all three C runs.

**Therefore the empirically tested defense is the trimming/median special
case of RCE, not RCE's margin machinery.** The paper must say so. The
general RCE formulation and Theorem 2's `ε(M,f,α)` remain valid theory,
but no experiment here exercises the margin, and none can at M = 3, f = 1.

### What A2 establishes

At M = 3, f = 1, replacing the mean aggregator with the trimming/median
special case of RCE removes ~80 % of the attack's true-cost effect
(interaction −3.20, 95 % CI [−4.73, −1.68], 3/3 signs) and ~83 % of its
mechanism-gap effect, while costing **essentially nothing** on clean runs
(E−A = +0.12, CI straddling zero, mixed signs) and nothing detectable on
task return (interaction mixed, CI ±13).

### What A2 does NOT establish

- **Not** that RCE's uncertainty margin works — it was inert (0.0015).
- **Not** aggregator superiority over coordinate-wise median, trimmed
  mean or Krum: at M = 3, f = 1 RCE *is* the median, so there is no
  comparison here. No aggregator ablation was run.
- **Not** anything at f = 2, M = 5, M = 7, or `2f ≥ M`. The breakdown
  study was not run.
- **Not** robustness to adaptive or Byzantine adversaries.
- Residual harm remains: C − E = +0.78 (3/3 positive). The defense
  reduces, it does not eliminate.

---

## Unconstrained MAPPO — Constraint-Binding Control

**STATUS: RUNNING, NOT COMPLETE.** Seed 0 launched 2026-09-12 19:57:48
UTC; seeds 1 and 2 are gated on its structural verification. ~8.3 h per
seed, ~25 h total. This section will be completed from
`results/runs_unconstrained/unconstrained_report.json`.

**Pre-declaration.** `docs/unconstrained_control.md`, written and fixed
**before** seed 0 started. `scripts/analyze_unconstrained.py` was written
before any control number existed.

**Question.** Does the safety constraint actually influence the
constrained learner? A1 and A2 are both differences *between* constrained
runs; neither contains a run in which the dual is absent, so neither can
show the dual is load-bearing.

**Treatment.** `dual.lambda_max: 25.0 → 0.0`. Verified on the **loaded
`ExperimentConfig`** (`scripts/verify_unconstrained_config.py`, all
checks pass, `results/runs_unconstrained/config_verification.json`): the
loaded-config diff against the paired clean config is exactly
`{run_id, seed, output_dir, dual.lambda_max}`; the PPO block, env,
topology, sources, source-collection, horizon, rollout length, estimator
and `eta_lambda` are identical. Config hashes (sha256 of the canonicalised
loaded config, first 16 hex): seed 0 `0aa231c9cd2dee6f`,
seed 1 `5aebd2b90f11b6e3`, seed 2 `805c2bf35c73ffc0`.
Launch sha `bed26a88edc714fbbd180c3478e351877c3eb33f`, no tracked file
modified, nothing under `src/` modified.

**Mechanism, verified on a 2-round pre-launch smoke run**
(`results/unconstrained_smoke/smoke`): λ is exactly 0.0 in 12 of 12 cells
(`max|λ| = 0.0`); the residual is still computed and live (round 0
per-agent: 13.83, 13.99, 15.78, 21.62, 29.24, 37.05); attack `none`,
`corrupted_source_ids: []`, defense `mean`; all quantities finite.
`np.clip(·, 0.0, 0.0)` makes the projection interval the single point
{0}, and `combined_adv = adv_r − λ·adv_c` reduces to `adv_r` identically.

**Frozen comparison target** (condition A, already committed — see G10):
`J_C` = 25.2143 / 24.9550 / 24.9782, mean **25.0492** (sd 0.1435),
against d = 25.0.

**Primary quantity.** `J_C,unconstrained − J_C,constrained` on whole-run
network-average true cost, paired by seed, common random numbers.

**No effect-size gate.** No pre-existing declaration fixes a bar, and
none was invented after the constrained side was in hand. Reported as an
estimate with per-seed values and a paired 95 % CI.

### Placeholders to fill on completion

| | seed 0 | seed 1 | seed 2 | mean (sd) |
|---|---|---|---|---|
| `J_C` whole / final-50 | — | — | — | — |
| per-agent `J_C` (×6) | — | — | — | — |
| task return first-20 / last-50 / gain | — | — | — | — |
| violation rate whole / final-50 | — | — | — | — |
| λ mean / max (must be 0.0) | — | — | — | — |
| residual final-50 sum | — | — | — | — |
| mech. reported cost | — | — | — | — |
| mechanism gap Δ | — | — | — | — |
| **U − A on `J_C` whole** | — | — | — | **— [CI]** |

### What it will establish

If U − A is clearly positive: the constrained learner's J_C ≈ d is caused
by the constraint, so A1's result is a claim about a live mechanism.

If U − A is near zero: the unconstrained optimum already satisfies the
budget, the dual is not load-bearing here, and A1/A2 must be re-read in
that light. **That outcome will be reported with equal prominence.**

### What it will NOT establish, in either direction

That the constraint is *sufficient* for safety (the constrained runs
violate on ~47 % of rounds and 2–3 of 6 agents sit above d); any
transfer to another environment, topology, budget, M or f; anything about
aggregators other than the mean.

---

## Theory

Proved in the appendix; all are independent of the experiments and stand
whatever the empirical outcome.

| result | statement | status |
|---|---|---|
| **Proposition 1** (what the consensus dual enforces) | For any connected doubly stochastic W, the dual fixed point pins `1ᵀg* ≤ 0`, i.e. the **network average** `J_C,avg ≤ d̄`, with equality at interior fixed points; individual residuals are otherwise unconstrained in sign and magnitude. The per-agent condition is the fixed point of `W = I` only. | Proved |
| **Theorem 1** (corruption mass conservation) | `1ᵀe_K = η_λ Σ_k 1ᵀδ_k`, independent of W, its connectivity and its spectral gap. | Proved |
| **Proposition 2** (spreading) | A persistent single-source bias gives `e_K = η_λδ[K/N·1 + r_K]` with `‖r_K‖₂ ≤ 1/(1−σ₂(W))`: every agent biased by `η_λδK/N + O(1)` instead of one agent by `η_λδK`. | Proved |
| **Proposition 3** (multiplicative amplification) | `‖Δ_R‖ = ‖∇θ ε‖` (zero for a constant reward offset) vs `‖Δ_C‖ = |e^i|·‖∇θ J_C^i‖` (nonzero for a constant cost offset). | Proved |
| **Theorem 2** (bounded violation) | `|Ĵ_C − J_C| ≤ ε(M,f,α)` under `M ≥ 2f+1`; if `βσ ≥ ε` then `J̄_C ≥ J_C`. | Proved |
| **Proposition 4** (transfer to the average) | Union bound gives the same guarantee for `J_C,avg` at level `1 − Nα`. | Proved |
| **Proposition 5** (liveness) | Over-reporting inflates by at most `ε` and drives λ to at most `λ_max`; RCE updates every round, whereas a gating defense can be frozen forever by `B > d^i`. | Proved |

### Numerical corroboration (in the appendix, cheap and already run)

Iterating the shipped dual update to convergence on heterogeneous
instances: for `ring`, `complete`, `star` and `W = (1/N)11ᵀ` at N = 6, the
fixed point satisfies `1ᵀg* ≤ 0` with `|1ᵀg*| < 1e-11` at interior fixed
points while individual agents sit above and below d; for `W = I` every
agent satisfies `Ĵ_C^i ≤ d`. The mean-multiplier trajectory agrees with
scalar dual ascent on `J_C,avg − d̄` to 1e-15 across all of these W.

**Empirical corroboration of Proposition 1** — this is worth stating,
because it is visible in the clean runs and costs nothing: the network
average sits at 25.05 against d = 25.0 while seed 0's per-agent whole-run
costs are (21.62, 21.57, 21.69, 23.92, 28.50, 33.99). The average is
pinned; the individuals are not. Seeds 1 and 2 show the same pattern.

### What the theory does NOT establish

- Theorem 1 assumes both dual trajectories stay interior to
  `[0, λ_max]`; the projected case gives an inequality, not equality.
  (Observed λ saturation is 0.0 in every committed run, so the interior
  assumption holds in practice here.)
- Theorem 2 assumes honest sources fail **independently**. Critic
  ensembles on shared trajectories violate this. The six constraint
  owners share one rollout: measured over the 250 rounds of clean seed 0,
  their pairwise `J_C` correlations run **min 0.836, median 0.931,
  max 0.998**. A "5 of 6 owners" statement is therefore not six
  independent tests.
- Theorem 2 is a per-round estimation guarantee. It is **not** a
  finite-time cumulative-violation bound and does **not** characterise
  the fixed point of the corrupted dynamics.
- No theory here covers adaptive corruption correlated with λ_k; that
  breaks the independence the proof uses.

---

## Limitations

State these in the paper, not in a footnote.

1. **One environment, one operating point.** ManyAgent Ant, N = 6,
   d = 25, ring topology, M = 3, f = 1, B/d = 0.5, 250 rounds,
   500k PPO steps. Nothing is varied. No HalfCheetah, no
   Safety-Gymnasium.
2. **n = 3 seeds.** Pre-declared, not chosen after the fact. Effect sizes
   and CIs; no significance claims. The repository enforces the floor in
   code (`MIN_SEEDS_FOR_INFERENCE = 5`).
3. **The tested defense is the median special case.** At M = 3, f = 1 the
   retained set has one element and the margin is 0.0015 cost units. RCE's
   distinguishing component is untested.
4. **No aggregator comparison.** Mean vs RCE-as-median is the only
   contrast. Krum, coordinate-wise median and trimmed mean were not run
   as separate arms.
5. **No baselines beyond the unconstrained control.** No MACPO, no
   Dec-PDO, no PID-Lagrangian. The PID controller is implemented and
   bitwise-reduces to the Lagrangian at `k_p = k_d = 0`, but no PID run
   exists.
6. **One attack point.** Persistent, static, consistent, negative. No
   adaptive attack, no Byzantine equivocation, no selective support, no
   positive direction, no dose-response over B/d.
7. **Per-agent safety is not achieved and is not claimed.** 2–3 of 6
   agents exceed d in clean runs; ~47 % of rounds record a violation.
   Proposition 1 says the consensus dual cannot deliver more than the
   average, so this is the formulation's property, not a bug.
8. **Source independence is imperfect.** The six owners share one
   rollout; measured pairwise `J_C` correlations over clean seed 0 run
   0.836 (min) / 0.931 (median) / 0.998 (max). Theorem 2's independence
   assumption is violated in the very runs that support it.
9. **Cost-critic bias is large and separate from the attack.** The
   own-critic gap is ≈ 7.5 under GAE even with no attack. This is why the
   paper reports the mechanism gap. Say so explicitly rather than letting
   a reader discover it.
10. **The 22,500 source env steps per round dominate compute.**
    45M source steps vs 500k PPO steps per run. This is a property of the
    trajectory-batch source architecture and bounds what could be run.

---

## Traceability

| claim | artifact |
|---|---|
| G10 | `results/runs_constraint_batch_g10/g10_report.json` |
| A1 | `results/runs_a1/a1_report.json`, `a1_frozen_verification.json` |
| A2 | `results/runs_a2/a2_report.json`, `a2_frozen_verification.json` |
| mechanism gap | `results/mechanism_gap_reanalysis.json` |
| control config | `results/runs_unconstrained/config_verification.json` |
| control runs | `results/runs_unconstrained/U_seed{0,1,2}/` |
| control analysis | `results/runs_unconstrained/unconstrained_report.json` |
| pre-declarations | `docs/g10_gates.md`, `docs/a1_attack_gates.md`, `docs/a2_rce_gates.md`, `docs/unconstrained_control.md` |
