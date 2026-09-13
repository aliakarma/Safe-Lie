# A2 — RCE defense experiment with conservatism control: final report

**Status: COMPLETE. Verdict: PASS — DEFENSE SUPPORTED.**
Six runs, three paired seeds, all gates pass. Analysis produced by
`scripts/analyze_a2.py` from `results/runs_a2/a2_report.json`; every number
below is read from committed run artifacts, none is hand-entered.

Gates were pre-declared in [`a2_rce_gates.md`](a2_rce_gates.md), committed
`f94cfe1`, **before any RCE run existed**.

---

## A. A2 Protocol

Two-factor factorial, attack × defense, on the frozen A1/G10 architecture.

| | no RCE | RCE |
|---|---|---|
| **clean** | A (reused from A1) | **E** (new) |
| **attacked** | B (reused from A1) | **C** (new) |

Six new runs (C and E × 3 seeds). A and B reused unmodified from A1, which
is what makes the design paired rather than merely matched.

**Frozen, not re-derived** — identical to A1/G10 in every field: M=3 parallel
trajectory-batch sources under a pinned policy, R_m=30 trajectories per
source, ring consensus topology, PPO+GAE, projected dual update, d=25,
eta_lambda, lambda_max, 250 rounds, 500,000 PPO env-steps, 22,500 source
trajectories = 45,000,000 source env-steps. `scripts/a2_verify_frozen.py` asserts this field-by-field; only
`run_id`, `output_dir`, `attack` and `defense` may differ.

**RCE block** (identical in C and E): `name: rce, f: 1, beta: 1.5,
sigma_min: 0.001, min_retained: 3, use_reliability_weights: false,
calibration_rounds: 20, calibration_alpha: 0.05`.

**Attack** (identical in B and C): primary, negative, persistent, static,
`budget_ratio = 0.5` so B = 12.5 = 0.5*d. One corrupted source, balanced
across seeds: seed 0 → `batch_1`, seed 1 → `batch_2`, seed 2 → `batch_3`.

---

## B. Exact RCE implementation audit

**Read before any run, from the code, not from the paper.**
`safelie/defenses/rce.py` → `rce_aggregate(values, f, beta, sigma_min, min_retained)`:

1. calls `trimmed_mean_aggregator`, which sets `retained_idx = order[f : m - f]`;
2. computes the MAD over the **retained set only**;
3. floors the spread at `sigma_min` when `retained_n < min_retained`;
4. returns `pessimistic_estimate = point_estimate + beta * spread`.

**At M=3, f=1 the retained set holds exactly one value.** `order[1:2]` is a
single element. The MAD of a single value is identically 0, so the floor
always fires, and the margin is the constant `beta * sigma_min = 0.0015`.
The trimmed mean of one value is the median. Therefore, for the whole of A2:

```
Jbar_RCE^i  =  median_m ( Jhat_C^{i,m} + delta_m )  +  0.0015
```

This is an algebraic identity at this operating point, independent of the
input values. It was written into the pre-declaration (`a2_rce_gates.md` §4)
before data existed, and then **verified numerically**: gate G2 recomputes
the entire map offline from the logged source reports and compares it against
what the learner actually consumed, over **1500 cells per run**. Maximum
absolute error, every run, every field — `point_estimate`, `spread`,
`retained_n`, `applied_margin`, `pessimistic_estimate`,
`mechanism_reported_cost_return`, `constraint_residual` — is exactly **0.0**.

The implementation differs from the paper's Algorithm 1 in one respect that
matters and one that does not:

- **Reliability weights** (Algorithm 1 lines 2, 10) are declared and
  initialised, and **no line reads them** (`use_reliability_weights: false`).
  This is the paper's own `[GAP]` G1 and was not invented here.
- **The MAD/margin half of Algorithm 1 is present and correct in code**, but
  is *structurally inert at this M*. That is a property of the operating
  point, not a defect of the implementation.

---

## C. Pre-declared gates

| gate | what it fixes |
|---|---|
| A2-G1 | implementation integrity; the audited map; CRN; derived env-steps |
| A2-G2 | attack replication — B reproduces A1 |
| A2-G3 | clean RCE effect (E−A), **measured, not hypothesised** |
| A2-G4 | residual attack effect (C−E) |
| A2-G5 | **the interaction (C−E)−(B−A)** — the primary quantity |
| A2-G6 | learning health |
| A2-G7 | source/RCE mechanism, including whether MAD is ever active |

A2-G5's PASS rule, fixed in advance: **mean I < 0, I < 0 in all three seeds,
and mean (C−E) ≤ 0.5 × (B−A)**. The original pilot's RCE gate was not reused.

---

## D. Integrity and seed pairing

| check | result |
|---|---|
| CRN round-0 identity vs paired A1/G9/G10 run | `max_abs_pre_attack_difference = 0.0`, **all 6 runs**, all 3 sources |
| underlying source draws identical | true, all 6 |
| duplicate source-seed events | **0**, all 6 |
| derived PPO env-steps (250 × rollout) | 500,000 exact, all 6 |
| derived source trajectories (from `rounds.jsonl`) | 22,500 exact, all 6 (250 × 3 × 30) |
| derived source env-steps (from `rounds.jsonl`) | 45,000,000 exact, all 6 (22,500 × 2,000) |
| config frozen vs A1/G10 | PASS, field-by-field |
| sign-convention check: B−A vs A1's published value | 3.98576 vs 3.9858, **agrees to 1e-3** |

That last row matters: it proves the analyser re-derives A1's headline number
from the reused runs, so A2's contrasts are on the same scale and sign
convention as A1's.

> **Correction on record (2026-09-09).** The two source-collection rows above
> previously read as a single row, *"derived source env-steps — 22,500,000"*.
> That was a mislabelling, not a miscount: 22,500 is the **trajectory** count
> (250 × 3 × 30), and each trajectory is `rollout_length = 2,000` env-steps,
> so the env-step figure is **45,000,000**. The run artifacts were always
> right — every round's `source_batch.env_steps` is 180,000 and
> `a2_report.json` records `source: 45000000` for all twelve runs — so only
> this prose was wrong and no A2 conclusion depends on it. Found while
> auditing A3's G1-iv derivation, which uses the correct convention
> (`5 × 30 × 2000 × 250 = 75,000,000`) and is unaffected.

---

## E. Condition matrix — whole-run network-average true cost (d = 25)

| seed | A clean | B attacked | E clean+RCE | C attacked+RCE |
|---|---|---|---|---|
| 0 | 25.214 | 29.056 | 25.681 | 26.455 |
| 1 | 24.955 | 29.273 | 25.364 | 25.810 |
| 2 | 24.978 | 28.776 | 24.470 | 25.600 |
| **mean** | **25.049** | **29.035** | **25.172** | **25.955** |

Whole-run averaging, per the repository's standing rule — a trailing window
samples a random phase of the lambda oscillation and previously produced a
sign-reversing artifact.

---

## F. Source-level RCE behaviour

Logged per round, per agent, for every run.

| quantity (attacked runs, C) | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| attacked source **trimmed** | 100 % | 100 % | 100 % |
| attacked source retained | 0 % | 0 % | 0 % |
| attacked source is the global **minimum** | 100 % | 100 % | 100 % |
| retained value = smaller of the two honest | 100 % | 100 % | 100 % |

**RCE removes the corrupted source; it does not bound its influence.** With
B = 12.5 ≈ 11.9 sigma_src, the corrupted report is the minimum in every
single round, so trimming the extremes discards it outright. The retained
value is then the *smaller* of the two honest reports — a downward-biased
survivor, which is the mechanism behind the residual effect in section I.

Within-run counterfactual (what the aggregate would have been without RCE):

| | seed 0 | seed 1 | seed 2 | predicted | band |
|---|---|---|---|---|---|
| mean RCE shift | −0.628 | −0.762 | −0.714 | −0.59 | [−1.2, −0.2] |
| mean-aggregator shift (−B/M) | −4.167 | −4.167 | −4.167 | | |
| **suppression factor** | **6.64×** | **5.47×** | **5.84×** | | |

All three land inside the pre-registered band.

In the clean runs (E) the retained value equals the median in 100 % of
rounds, as the algebra requires.

---

## G. Clean RCE effect — E − A

```
mean  = +0.122     95% CI [-1.236, +1.480]
per-seed = [+0.466, +0.409, -0.508]     sign-consistent: NO
Cohen's dz = 0.22    sign-test p = 1.0
```

**Clean RCE's effect on true cost is not distinguishable from zero and is
not sign-consistent.** This is exactly what the mechanism predicts: trimming
to a single retained value costs some precision, the constant 0.0015 margin
adds no meaningful conservatism, and the net is small and seed-dependent.

This was pre-declared as a *measurement*, never as a hypothesis to be passed
or failed — G3 gates only run validity.

> **Correction on record.** An interim two-seed read of +0.437 was quoted in
> `a3_gates.md` §2 as if final. Seed 2 came in at −0.508 and largely
> cancelled it. The final value is +0.122. See `a3_gates.md` §13.

---

## H. Attack effect without RCE — B − A

```
mean  = +3.986     95% CI [+3.269, +4.702]
per-seed = [+3.842, +4.318, +3.798]     sign-consistent: YES
Cohen's dz = 13.82
```

The attack replicates A1 exactly. **The corruption raises true cost by ~4.0
against a budget of 25 — a 16 % overshoot — while reported cost stays at
24.49–24.77, i.e. below d.** That is the stealth signature.

---

## I. Attack effect with RCE — C − E

```
mean  = +0.783     95% CI [-0.066, +1.632]
per-seed = [+0.775, +0.446, +1.130]     sign-consistent: YES
Cohen's dz = 2.29
```

**Positive in all three seeds: RCE does not fully neutralise the attack.**
19.7 % of the attack's effect survives. Mechanism from section F: because the
corrupted report is always the minimum, trimming leaves the *smaller* honest
value, which is itself biased low relative to the honest mean.

---

## J. RCE main effect — C − B

```
mean  = -3.080     95% CI [-4.171, -1.989]
per-seed = [-2.601, -3.463, -3.176]     sign-consistent: YES
```

**Reported for completeness only.** This contrast is confounded — it mixes
robustness against corruption with any conservatism RCE would impose on a
clean run. It is *not* the defense claim, and the pre-declaration forbids
calling it one. Section K is the claim.

---

## K. Two-factor interaction — (C − E) − (B − A)  ← PRIMARY

```
I     = -3.202     95% CI [-4.725, -1.680]
per-seed = [-3.067, -3.872, -2.668]     negative in 3 of 3 seeds
Cohen's dz = -5.22
fraction of attack effect removed = 80.3 %
```

Gate rule, fixed in advance: **I < 0 in all three seeds AND (C−E) ≤ 0.5 ×
(B−A)**. Observed: 0.783 ≤ 1.993. **Both conditions met.**

The interaction is the quantity that is *not* confounded by conservatism: it
asks whether the attack's effect is smaller under RCE than without it,
differencing out whatever RCE does on a clean run. It is negative, in every
seed, and large relative to its spread.

---

## L. Task performance

| condition | return (whole run) | learning gain |
|---|---|---|
| A | −119.7 / −119.2 / −124.3 | — |
| B | −112.9 / −116.0 / −118.6 | — |
| E | −115.6 / −124.8 / −125.8 | 158.4 / 147.0 / 151.3 |
| C | −116.1 / −118.3 / −123.2 | 157.3 / 142.1 / 157.2 |

Return spans −112.9 to −125.8 across **all twelve runs** — a range of 12.9
against seed-level noise of comparable size. **RCE costs no measurable task
performance**, and the attack buys none. Every run's learning gain clears the
G10-C-ii reference of +75 by a wide margin.

---

## M. Network-average safety

| run | reported | true | gap | viol % (whole) | viol % (last 50) | peak |
|---|---|---|---|---|---|---|
| A | 24.77 / 24.97 / 24.79 | 25.21 / 24.96 / 24.98 | 7.51 / 7.71 / 6.50 | 47.3 / 46.1 / 46.5 | 50.7 / 55.7 / 45.3 | 1.00 |
| B | 24.49 / 24.77 / 24.57 | 29.06 / 29.27 / 28.78 | 8.62 / 8.48 / 7.53 | 65.7 / 63.6 / 61.5 | 72.0 / 67.0 / 61.7 | 1.00 |
| E | 24.71 / 24.81 / 24.68 | 25.68 / 25.36 / 24.47 | 7.42 / 8.25 / 6.43 | 48.7 / 47.6 / 43.9 | 52.3 / 50.3 / 42.7 | 1.00 |
| C | 24.65 / 24.96 / 24.80 | 26.46 / 25.81 / 25.60 | 8.17 / 8.44 / 7.49 | 52.6 / 48.1 / 50.0 | 55.7 / 52.7 / 47.7 | 1.00 |

**Reported cost sits at 24.5–25.0 in every condition** — the constraint is
doing its job on the signal the learner can see, in all four cells. What
differs is the true cost behind it.

**Violation rate ordering is monotone and matches true cost**: B (61.5–65.7 %)
> C (48.1–52.6 %) > A (46.1–47.3 %) ≈ E (43.9–48.7 %).

**Two metric caveats, reported because they are unfavourable:**

1. **The detection gap does not isolate the attack at this scale.** It is
   6.4–8.7 in *every* condition, clean included, because the GAE-corrected
   cost critic is systematically biased low regardless of any adversary. It
   moves by ~1.0 under attack against a ~7.5 baseline. The paper's §5 treats
   the detection gap as operationalising stealth; **on this evidence it does
   not**, and the usable stealth signature is instead "reported cost stays
   ≤ d while true cost rises".
2. **Peak violation is 1.00 in all twelve runs** — saturated at the per-step
   cost cap. It carries no information here and should not be tabulated as
   though it does.

---

## N. Per-agent safety (whole-run true cost, d = 25)

| run | ag 0 | ag 1 | ag 2 | ag 3 | ag 4 | ag 5 | # > d (last 50) |
|---|---|---|---|---|---|---|---|
| A_seed0 | 21.62 | 21.57 | 21.69 | 23.92 | 28.50 | 33.99 | 2 |
| A_seed1 | 20.89 | 20.85 | 21.28 | 24.05 | 28.49 | 34.17 | 3 |
| A_seed2 | 20.67 | 20.73 | 21.49 | 24.25 | 28.62 | 34.11 | 2 |
| B_seed0 | 24.68 | 24.62 | 25.08 | 28.01 | 32.92 | 39.03 | 6 |
| B_seed1 | 25.22 | 25.23 | 25.51 | 28.27 | 32.87 | 38.54 | 6 |
| B_seed2 | 24.27 | 24.29 | 24.84 | 27.78 | 32.61 | 38.86 | 5 |
| C_seed0 | 22.60 | 22.58 | 22.82 | 25.32 | 29.86 | 35.55 | 3 |
| C_seed1 | 21.81 | 21.82 | 22.23 | 24.77 | 29.25 | 34.98 | 3 |
| C_seed2 | 21.58 | 21.53 | 21.90 | 24.52 | 29.24 | 34.82 | 2 |
| E_seed0 | 22.34 | 22.26 | 22.24 | 24.56 | 28.71 | 33.96 | 2 |
| E_seed1 | 21.41 | 21.43 | 21.86 | 24.42 | 28.71 | 34.34 | 2 |
| E_seed2 | 20.75 | 20.71 | 20.91 | 23.39 | 27.73 | 33.32 | 2 |

**Stratification is severe and structural, not adversarial.** Agents span ~21
to ~34 against d=25 in the *clean* condition; agent 5 is 36 % over budget
with no attacker present, and 2–3 of 6 agents exceed d in every clean run.
This is the predicted consequence of Proposition 1: the consensus dual pins
only the **network average**, so a heterogeneous cost stratifies underneath
it. It is a finding about the algorithm, reported here whether or not it is
favourable.

**The attack's effect is uniform across the stratification** — it shifts
every agent up by ~3–5 without changing the ordering — and **RCE's repair is
equally uniform**, returning the agent-above-budget count from 5–6 back to
2–3, i.e. to the clean baseline.

---

## O. Lambda / dual dynamics

| run | mean | max | frac > 0 | frac saturated |
|---|---|---|---|---|
| A | 1.510 / 1.600 / 2.178 | 9.33 / 9.36 / 12.75 | .925 / .950 / .929 | **0.0000** |
| B | 1.035 / 1.297 / 1.689 | 7.11 / 8.02 / 10.52 | .870 / .916 / .901 | **0.0000** |
| E | 1.334 / 1.924 / 1.971 | 8.74 / 10.67 / 11.35 | .919 / .923 / .907 | **0.0000** |
| C | 1.450 / 1.451 / 1.919 | 9.32 / 8.04 / 11.35 | .909 / .945 / .926 | **0.0000** |

**Lambda is never saturated in any of the twelve runs**, so lambda_max is not
binding and the dual is free to respond throughout.

The ordering is the causal mechanism made visible: **B has the lowest lambda
in every seed.** Under-reported costs produce a smaller residual, the dual
relaxes, the policy is under-constrained, and true cost rises. RCE restores
lambda toward the clean level (C > B in all three seeds), which is *why* true
cost falls. The multiplier and the true cost move in exactly inverse order.

---

## P. MAD / theoretical-condition diagnostics

**This is A2's most important negative result.**

| diagnostic | value, all six runs |
|---|---|
| `retained_n == 1` | **100 % of rounds** |
| `degenerate == true` | **100 % of rounds** |
| spread == `sigma_min` | **100 % of rounds** |
| margin == `beta*sigma_min` = 0.0015 | **100 % of rounds** |
| MAD ever non-zero | **false** |
| `theoretical_condition_ever_active` | **false** |
| `epsilon_offline` (all 6 calibrations) | **0.001** exactly |
| calibration disagreement sd | 4.3e-19 |
| `guarantee_in_force` true | **100 % of rounds** |

**Theorem 2's MAD condition was never active for a single round of A2.**

The last two rows are the trap. `guarantee_in_force` compares the empirical
margin `beta*sigma` against the offline reference `epsilon_offline`. The
margin is the constant 0.0015. The calibration, run on a clean clone,
measured disagreement on the *same degenerate path* and so returned exactly
`sigma_min` = 0.001, with a standard deviation of 4e-19 — it measured the
floor, not any disagreement. So the check reads `0.0015 >= 0.001` → **true,
in 100 % of rounds, vacuously**. A naive reading would report "the Theorem 2
guarantee held throughout." It held because both sides of the comparison were
constants derived from the same floor.

**A2 tested RCE's trimming. It did not test RCE's margin.** Any statement
about Theorem 2's precondition is out of scope for this experiment.

---

## Q. Statistical analysis

`n = 3` paired seeds. Decision D6 and `MIN_SEEDS_FOR_INFERENCE = 5` forbid
inference below five seeds; the code raises rather than emitting an
underpowered p-value.

**Primary evidence is per-seed sign consistency**, pre-declared:

| contrast | sign-consistent | detail |
|---|---|---|
| B − A | **yes** | 3/3 positive |
| C − E | **yes** | 3/3 positive |
| E − A | **no** | 2 positive, 1 negative |
| **I = (C−E)−(B−A)** | **yes** | **3/3 negative** |

Paired-t values are computed and stored but **labelled SENSITIVITY ONLY** in
the report JSON, and are not part of any gate. The exact sign test's minimum
attainable p at n=3 is **0.25** — no three-seed design can reach
significance, which is precisely why the gate is stated on sign and magnitude
rather than on p.

Confidence intervals are reported for scale, not for decisions. Note that
E−A's CI spans zero and C−E's touches it; only B−A and I are comfortably
clear of it.

---

## R. Gate results

| gate | verdict |
|---|---|
| A2-G1/G2 structural integrity | **PASS** (6/6 runs) |
| A2-G3 clean RCE effect | **PASS** (run validity; value is a measurement) |
| A2-G4 residual attack effect | **PASS** (reported) |
| A2-G5 **interaction** | **PASS** |
| A2-G6 learning health | **PASS** (6/6; KL, entropy, lambda, finiteness) |
| A2-G7 RCE mechanism | **PASS** (6/6) |

No gate was weakened, moved, or reinterpreted after seeing data. No extra
diagnostic runs were created — no structural gate failed.

---

## S. A2 verdict

### **PASS — DEFENSE SUPPORTED**

The pre-declared primary quantity, the two-factor interaction, is negative in
all three seeds with a mean of −3.202, and the residual attack effect under
RCE (0.783) is well under half the undefended attack effect (3.986). RCE
removes **80.3 %** of the corruption's effect on true cost, at no measurable
cost to task return and with no measurable conservatism penalty on clean
runs.

Against the pre-registered outcome taxonomy this is **OUTCOME C — genuine
interaction**, not OUTCOME A. A strong-defense reading is not available:
19.7 % of the attack effect survives, sign-consistently.

---

## T. What A2 establishes

1. **RCE genuinely mitigates persistent cost corruption** — established on
   the interaction, not on the confounded C−B, and sign-consistent across all
   three seeds.
2. **The mitigation is not conservatism in disguise.** E−A is
   indistinguishable from zero, so there is no clean-run safety margin for
   the C−B contrast to have been borrowing from.
3. **The mechanism is trimming, and specifically source removal.** The
   corrupted report is the global minimum in 100 % of rounds and is discarded
   in 100 % of rounds; suppression versus a mean aggregator is 5.5–6.6×.
4. **The mitigation is partial and the residual has a known cause.**
   Discarding the minimum leaves the *smaller* honest report, which is itself
   biased low — this is why C−E is positive in every seed.
5. **The causal path is visible in the dual.** The attack suppresses lambda;
   RCE restores it; true cost follows in inverse order. Lambda never
   saturates.
6. **At M=3, f=1, RCE is exactly `median + 0.0015`** — proven algebraically
   before the runs and verified to 0.0 absolute error over 1500 cells per
   run. **Theorem 2's MAD condition was never active.**
7. **`guarantee_in_force` can read true vacuously.** Both sides of the
   comparison collapsed to the same `sigma_min` floor. Any deployment reading
   that flag as evidence would be misled; this is a reportable defect of the
   diagnostic, not of the run.
8. **Per-agent stratification is structural**, present at 21–34 against d=25
   in clean runs, consistent with Proposition 1's network-average result. The
   attack and the defense both act uniformly across it.
9. **The detection gap does not operationalise stealth at this scale** — it
   is 6.4–8.7 in every condition including clean.

---

## U. What A2 does NOT establish

- **Anything about Theorem 2's margin.** The MAD was structurally zero in
  every round. A2 tested one half of Algorithm 1.
- **Anything about the numerical bound epsilon(M,f,alpha)**, which depends on
  an unobservable sub-Gaussian parameter.
- **Statistical significance.** n=3; the minimum attainable sign-test p is
  0.25. The evidence is sign consistency, not inference.
- **Superiority over other robust aggregators.** At M=3, f=1 RCE *is* the
  median plus a constant, so no comparison against median, Krum or trimmed
  mean was possible in principle here.
- **Generality**: f > 1; M ≠ 3; other topologies; a second environment; other
  attack strengths (B/d in {0.25, 1.0}); over-reporting; adaptive or
  Byzantine adversaries; reliability weights; baseline algorithms.
- **That the residual 19.7 % is irreducible** — it may be an artifact of the
  retain-the-smaller-honest-value mechanism at M=3 specifically.

---

## V. Exactly one recommended next experiment

**A3 — the same four-cell design at M = 5, changing nothing else.**

At M=5, f=1 the retained set holds |T| = M − 2f = **3** values, which is
exactly `min_retained`. Nothing is floored, the MAD is live, and `beta*MAD`
becomes a real quantity that can respond to corruption. This is the minimum
change that makes the half of Algorithm 1 A2 could not test become testable,
and it does not disturb any frozen component.

It is pre-declared in [`a3_gates.md`](a3_gates.md) with its own gates, which
**invert A2's**: where A2 required the margin to be provably dead, A3
requires it to be provably live (`retained_n == 3`, `degenerate == false`,
`epsilon_offline > sigma_min`, margin within a pre-registered band, and the
margin must *respond* to corruption). Twelve runs, ~2.0 days on the Mac mini
at three seeds, ~3.4 days at five.

Its precondition — that A2's interaction holds its sign — **is met**.
