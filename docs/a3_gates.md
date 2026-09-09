# A3 — Does RCE's pessimism margin do anything, at M=5 where it is finally live?

**Status: PRE-DECLARED. Written and committed before any A3 run existed, on
any machine.** Every threshold here is fixed now. None was chosen after
seeing an A3 result.

**A3 runs only if A2 passes.** If A2's interaction does not hold its sign,
the next question is not this one, and A3 is abandoned rather than
re-scoped. That decision belongs to the A2 verdict, not to this document.

---

## 1. Why A3 exists

A2 audited the shipped RCE implementation at the frozen `M=3, f=1`
operating point and found the trimmed set holds exactly **one** value.
`MAD({1 point}) = 0` structurally, the spread is floored to `sigma_min`,
and the margin collapses to the constant `beta*sigma_min = 0.0015` — 0.006 %
of the budget. A2 therefore tests RCE's **trimming** and can say nothing
whatever about its **margin**, which is the half of Algorithm 1 that
Theorem 2's condition is about.

At `M=5, f=1` the trimmed set holds `M - 2f = 3` values. `min_retained` is
satisfied, nothing is floored, and `beta*MAD` becomes a real, data-dependent
quantity. A3 is the smallest change to A2 that makes the margin live.

**This is the only difference.** Same environment, same attack, same PPO,
same dual, same topology, same `d`, same `R_m`, same horizon, same seed
definitions. `M` goes 3 -> 5 and nothing else moves. (How many of those
seeds are run, and on which machine, is the section-4 branch; it changes the
scope of A3, never its design.)

## 2. What the margin does at M=5 — derived from the real code, before any run

200,000 Monte-Carlo draws through `safelie.defenses.rce.rce_aggregate`
itself, at the measured source dispersion `sigma_src = 1.053` (pooled
between-source sd from the existing condition-A runs) and `B = 12.5`:

| quantity | M=3 (A2, measured) | M=5 (A3, predicted) |
|---|---|---|
| retained set size | 1 | 3 |
| `degenerate` | 100 % | **0 %** |
| MAD, clean | 0 (floored to 1e-3) | **0.270** |
| MAD, under attack | 0 (floored to 1e-3) | **0.363** |
| margin `beta*MAD`, clean | 0.0015 (constant) | **0.404** |
| margin `beta*MAD`, attacked | 0.0015 (constant) | **0.544** |
| clean aggregate bias | +0.003 | **+0.402** |
| attack shift of the aggregate | −0.594 | **−0.222** |
| mean-aggregator shift `−B/M` | −4.1667 | −2.5000 |
| suppression vs mean | 7.0x | **11.3x** |
| attacked source trimmed | 100 % | 100 % |

Three consequences, all pre-registered:

1. **The margin is live and responds to the attack.** It rises 0.404 ->
   0.544 (+34 %) when one source is corrupted, because the corrupted value
   widens the retained set's dispersion. At M=3 it was a constant that
   could not respond to anything.
2. **Clean RCE becomes genuinely conservative.** The margin biases the
   clean aggregate up by +0.402, so the reported cost estimate is higher,
   the residual is higher, `lambda` is higher, and the policy should be
   *safer* when no attacker is present.
3. **The sign of the conservatism channel should FLIP between A2 and A3.**
   A2 measured `E − A = +0.437` (RCE slightly *worse* when clean, because
   trimming costs precision and the margin was inert). A3 predicts
   `E' − A' < 0` (RCE *safer* when clean, because the margin now dominates
   the precision loss). This is the sharpest falsifiable prediction A3
   makes, and it is why the four-cell design is still required: with a live
   conservatism channel, `C' − B'` becomes genuinely confounded in a way it
   was not at M=3.

## 3. Conditions

| Condition | Attack | RCE | Purpose |
|---|---|---|---|
| **A'** | no | no | clean reference, M=5 |
| **B'** | yes | no | attack reference, M=5 |
| **C'** | yes | yes | defense |
| **E'** | no | yes | RCE conservatism |

**Nothing is reused from A2.** Changing `M` changes the source
architecture, so A3 collects its own `A'` and `B'`. All four conditions are
load-bearing: at M=5 the margin makes clean conservatism real (section 2),
so `C' − B'` is genuinely confounded and `E'` is what breaks the confound.
No condition may be dropped to save time. **12 runs under BRANCH-A, 8 under
BRANCH-B** (section 4).

The decomposition is unchanged from A2 and is not renegotiable:

```
attack effect             =  B' − A'
clean RCE effect          =  E' − A'
attack effect under RCE   =  C' − E'
attack x RCE interaction  = (C' − E') − (B' − A')     <- primary
RCE main effect (raw)     =  C' − B'                  <- reported, never primary
```

## 4. Two machines — the assignment, chosen by measurement

> **SUPERSEDED IN PART — see section 12.** The probe ran on 2026-09-09 and
> selected BRANCH-B, but also measured a ~2.9x speed asymmetry between the
> two machines that this section did not anticipate. Section 12 records the
> resulting amendment (**BRANCH-B'**), made after the probe and before any
> A3 run existed. The text below is left unchanged as the original
> pre-declaration.


A3's scope is **branched on a hardware property, measured before any A3 run
exists**. Both branches are fixed here. The selector is the section-9 probe,
which compares a 4-round run on the second machine against the copy this
repository already carries from the first. It measures whether two computers
agree on the same arithmetic; it cannot see an A3 result, so branching on it
does not weaken the pre-declaration.

### BRANCH-A — the probe shows bit-identity

Cross-machine paired contrasts are valid, so the 12 runs balance evenly.

| machine | assignment | runs |
|---|---|---|
| Windows | conditions `A'` and `B'`, all 3 seeds | 6 |
| Mac mini | conditions `C'` and `E'`, all 3 seeds | 6 |

**3 seeds, ~3.25 days**, full `n = 3` parity with A1 and A2. No reduced-claim
caveat is needed and none will be added.

### BRANCH-B — the probe shows any difference

Every contrast must then live entirely on one machine, so each machine runs
*all four conditions* of the seeds it owns.

| machine | assignment | runs |
|---|---|---|
| Windows | seed 0, conditions `A' B' C' E'` | 4 |
| Mac mini | seed 1, conditions `A' B' C' E'` | 4 |

**2 seeds, ~2.2 days.** The third seed is dropped, not because two is
adequate, but because 12 runs split 8/4 under this branch and the third seed
therefore costs **2.2 extra days of critical path, not 1.1**. A single-seed
A3 is never chosen: it costs the same 2.2 days as two seeds, since the
second machine runs four either way.

Under BRANCH-B the claims split, and the split is declared now:

* **Mechanism claims (A3-G3) are primary and fully powered.** They are
  per-cell facts measured over ~1,500 aggregation cells per run — that the
  retained set holds 3 values, that MAD is a real measurement rather than a
  floor, that `beta*MAD` lands where section 2 predicts, and that it rises
  under corruption. None of these depends on the seed count.
* **The downstream interaction (A3-G5) is reported as DIRECTIONAL ONLY.** At
  `n = 2` the minimum attainable sign-test p is 0.500 and the 95 % CI
  halfwidth is 8.98 x sd. A3-G5's PASS/CONDITIONAL/FAIL labels are still
  computed and reported, but under BRANCH-B they carry an explicit `n = 2,
  directional` marker everywhere they appear, and no significance claim is
  made for the interaction in the paper.

### The rule both branches share

Under BRANCH-B, a machine effect that shifts all four cells of a seed
equally cancels exactly in `B' − A'`, `E' − A'`, `C' − E'` and the
interaction, because each is a difference between two runs on the same
hardware. What that does **not** protect against is a machine x condition
*interaction* — implausible, but not proven, and therefore stated as a known
limitation rather than assumed away. Under BRANCH-A the probe has already
shown the machines agree bitwise, so the question does not arise.

**Cross-machine bit-identity is never assumed, only measured.** Source
*seeds* are pure integer arithmetic (`SeedSequence -> PCG64 -> integers`)
and are identical on any architecture; source *trajectories* run through
MuJoCo and PyTorch floats and may differ between x86-64 and ARM64. Under
BRANCH-B the A2-G1-ii style round-0 CRN check is applied **within a machine
only**, and a cross-machine mismatch is explicitly not a stop condition.

Whichever branch fires, the seed assignment and the machine that produced
each run are recorded in every results table, not only here.

## 5. Seed pairing and common random numbers

Fresh source entropy per seed, by the G10 convention
`int(sha256(b"safelie/a3/source-entropy/seed=<k>").digest()[:16])`:

| seed | attacked source (B' and C') | `seed_entropy` | used by |
|---|---|---|---|
| 0 | `batch_1` | 25873748826208450093657097358164192435 | both branches |
| 1 | `batch_3` | 224463726283861287585842441982236519383 | both branches |
| 2 | `batch_5` | 11901184105024366541245543018062542394 | BRANCH-A only |

Which machine produces which run is set by the section-4 branch, not by this
table. The seed -> entropy -> attacked-source mapping is identical under
either branch; BRANCH-B simply does not run seed 2. Seed 2 is the one
dropped (rather than seed 0 or 1) so that the two seeds retained are the
same two under both branches, and a BRANCH-B result is a strict subset of
what BRANCH-A would have produced.

New entropy, not A2's: A3 spawns `M+1 = 6` streams where A2 spawned 4, and
reusing A2's entropy would start A3's first three replicas at the same
points as A2's, correlating two campaigns that must be independent.

The attacked source is spread across the five (`1, 3, 5`) rather than
`1, 2, 3`. With three seeds and five sources a fully balanced mapping does
not exist; the sources are exchangeable by construction (independent
spawned streams, identical `R_m`), so spreading is a presentational choice,
declared here so it is not made later.

Within a seed, all four conditions share `seed` and `seed_entropy`.

## 6. Gates

### A3-G1 — implementation integrity (gating)

* **G1-i.** Each config differs from its frozen counterpart only in
  `run_id`, `output_dir`, and `defense.*`; and the M=5 architecture block
  (`sources`, `source_collection.M`) is identical across all four
  conditions of a seed. Machine-checked before launch.
* **G1-ii.** **Within-machine CRN.** At round 0, `C'`'s pre-attack source
  values equal `B'`'s, and `E'`'s equal `A'`'s, to <= 1e-9 on all 5 sources
  x 6 owners. Compared only against same-machine runs.
* **G1-iii.** Seed audit: `(M+1) x R_m x rounds` env seeds issued with 0
  duplicates and 0 pairwise overlap; spawn keys equal to the same-seed
  partner's.
* **G1-iv.** PPO env steps exactly 500,000; source env steps exactly
  `5 x 30 x 2000 x 250 = 75,000,000`.
* **G1-v.** `guarantee_calibration.json` present for every RCE run. Unlike
  A2, `epsilon_offline` is **not** expected to equal `sigma_min` here — the
  MAD is live, so it is a real quantile of a real dispersion distribution.
  Gate: `epsilon_offline > sigma_min` strictly, i.e. the calibration is no
  longer vacuous.

### A3-G2 — attack replication (gating)

* **G2-i.** `C'` and `B'` carry byte-identical attack blocks: `primary`,
  `f: 1`, `budget_ratio: 0.5`, `direction: negative`, `support: persistent`,
  and the section-5 attacked source.
* **G2-ii.** Injection identity on 100 % of cells: reconstructing values
  from the logged pre-attack reports, applying `−12.5` to the mapped
  source, and re-running `rce_aggregate` reproduces every logged aggregate
  field to <= 1e-9.
* **G2-iii.** `E'` and `A'` carry `attack.name: none`, `attack.f: 0`.

### A3-G3 — the margin is actually live (gating; A3's reason to exist)

* **G3-i.** `retained_n == 3` and `degenerate == False` on **100 %** of
  cells, in both `C'` and `E'`.
* **G3-ii.** `spread > sigma_min` on **>= 99 %** of cells — the MAD is a
  measurement, not a floor.
* **G3-iii.** Mean `applied_margin` in `[0.25, 0.60]` in `E'` (predicted
  0.404) and in `[0.35, 0.80]` in `C'` (predicted 0.544).
* **G3-iv.** The margin **responds to corruption**: mean `applied_margin`
  in `C'` exceeds that in `E'` at the same seed, in **every seed run**
  (3 under BRANCH-A, 2 under BRANCH-B). Predicted ratio 1.34; the gate is
  the direction only. Note this gate is a within-seed comparison of two
  per-cell means over ~1,500 cells each, so it is well determined at either
  seed count.

**If G3-i or G3-ii fails, A3 has not tested the margin and no conclusion
about Theorem 2's condition may be drawn from it** — the same discipline
that made A2 report its own degeneracy rather than bury it.

### A3-G4 — clean RCE effect (reported; one directional prediction)

`E' − A'` on whole-run network-average true cost, paired by seed.

Reported with per-seed values, mean, 95 % CI, sign consistency. The
**pre-registered prediction is `E' − A' < 0`** (RCE safer when clean, the
sign flip from A2's `+0.437`). This prediction is recorded so it can fail;
its failure does not by itself fail A3, but it must be reported as a failed
prediction and not reinterpreted after the fact.

### A3-G5 — interaction (gating; the primary result)

`I' = (C' − E') − (B' − A')` on whole-run network-average true cost.
Higher true cost = worse.

| Outcome | Rule |
|---|---|
| **PASS** | mean `I' < 0`, `I' < 0` in **all 3** seeds, **and** mean `(C' − E') <= 0.5 x (B' − A')` |
| **CONDITIONAL PASS** | mean `I' < 0` but not unanimous, **or** reduction between 0 % and 50 % |
| **FAIL** | mean `I' >= 0` |

Identical in form to A2-G5, deliberately: A3 must be comparable to A2 on
the same rule, and the 50 % bar is again the lenient form of the
mechanistic prediction (11.3x estimator suppression predicts ~9 % of
`B' − A'` surviving).

`C' − B'` is reported alongside and is **never** substituted for `I'`.

### A3-G6 — learning health (gating, G10-C verbatim)

No NaN/Inf; PPO KL median <= 0.0069 and p95 <= 0.01548; `lambda` saturation
fraction < 0.05; entropy declines and stays finite and positive; theta
checksum distinct in 100 % of rounds.

Task-return gain is **reported but not gated** — a defense that costs task
performance is a finding, and gating it would make that finding
unreachable. At M=5 with a live margin this matters more than it did in A2:
a persistently conservative policy is exactly the failure mode to watch.

### A3-G7 — mechanism (gating)

* **G7-i.** Attacked source trimmed on >= 99 % of cells in `C'`.
* **G7-ii.** Within-run counterfactual `RCE(post-attack) − RCE(pre-attack)`:
  predicted mean −0.222; gate band **[−0.70, −0.02]**. Compare against the
  mean aggregator's exact `−B/M = −2.5`.
* **G7-iii.** Retained set composition reported: which ranks survive, and
  how often the retained set under attack equals the three lowest of the
  four honest sources (predicted ~100 %).

## 7. Stop conditions

Halt that machine's queue immediately on: any G1/G2/G3-i/G3-ii failure;
round-0 CRN mismatch against a same-machine partner; the attacked source
not being the section-5 source; NaN/Inf in an RCE output; a duplicate or
miscounted source seed; round count != 250 or env steps != the G1-iv
values; oracle/reference leakage.

**A cross-machine CRN mismatch is NOT a stop condition** — it is expected
until the probe says otherwise, and no A3 gate depends on it.

## 8. Statistics

Paired seed-level differences, exactly as A2. `n = 3` under BRANCH-A,
`n = 2` under BRANCH-B. Primary is `I'`; secondary are `E' − A'`,
`C' − E'`, `C' − B'`.

Paired t-tests are emitted **labelled sensitivity-only** per decision D6
(`MIN_SEEDS_FOR_INFERENCE = 5`); no significance claim is made at either
seed count, regardless of p-value. The primary evidence is per-seed sign
consistency, as in A1 and A2.

What the seed count changes, stated now so it is not argued later:

| | BRANCH-A, `n = 3` | BRANCH-B, `n = 2` |
|---|---|---|
| min attainable sign-test p | 0.250 | 0.500 |
| 95 % CI halfwidth | 2.48 x sd | 8.98 x sd |
| interaction claim | same standing as A1/A2 | **directional only**, marked `n = 2` wherever it appears |
| mechanism claims (G3) | unaffected — per-cell, ~1,500 cells per run | unaffected |

**No seeds are added after seeing results**, under either branch — the same
rule that made A1's `B − D` permanently descriptive. If BRANCH-B fires, A3
stays at two seeds even if the interaction looks promising; wanting a third
seed *because* the first two were encouraging is exactly the data-dependent
inflation the pre-declaration exists to forbid.

Which machine produced each run is disclosed in every results table, not
only in this document.

## 9. Cross-platform probe — the branch selector

Run **once**, on the second machine, before any A3 run starts anywhere.

```
python scripts/train.py --config configs/experiment/a3/_platform_probe.yaml --eval-every 1
python scripts/a3_platform_probe.py     --reference results/a2_mechanism_check/mech_rce_clean     --candidate results/a3_platform_probe/probe
```

`configs/experiment/a3/_platform_probe.yaml` is the A2 clean mechanism-check
config with **only** `run_id` and `output_dir` changed — verified
field-by-field on the resolved model. Neither can affect a number (seeding
derives from `cfg.seed` and `seed_entropy` alone); they are changed only so
the second machine does not overwrite the reference it is compared against.
About 15 minutes.

The probe reports three levels separately, because they fail for different
reasons:

| level | what it tests | if it differs |
|---|---|---|
| 1. source seeds | `SeedSequence -> PCG64 -> integers` | **INVALID** — pure integer arithmetic, so this is a config problem, not hardware. Fix and re-run. |
| 2. policy checksum | `torch.manual_seed` -> network init | PyTorch differs across the architectures |
| 3. source values | the full MuJoCo + PyTorch float path | trajectories differ; round 0 is decisive, being the only round where both runs are still under the same policy |

**BRANCH-A** iff levels 1, 2 and round-0 level 3 all match to <= 1e-9.
**BRANCH-B** otherwise. Written to
`results/a3_platform_probe/a3_platform_probe.json`, which records the
selected branch, the observed magnitudes, and both machines' platform and
library versions.

The probe does not gate A3 and cannot fail it — "not identical" is an
expected outcome across x86-64 and ARM64, and is the case the by-seed split
was designed for. Under BRANCH-A, record the result: it also frees future
campaigns from the by-seed restriction.

## 10. Compute

12 runs (BRANCH-A) or 8 runs (BRANCH-B) x 250 rounds. `M=5, R_m=30` collects
150 trajectories per round against A2's 90, so source cost is 1.67x:
**75,000,000 source env steps per run** against A2's 45,000,000. From A2's
measured 8.03 h/run, expect **~13 h/run**.

| branch | seeds | Windows | Mac mini | critical path |
|---|---|---|---|---|
| **A** (bit-identical) | 3 | 6 runs, ~78 h | 6 runs, ~78 h | **~3.25 days** |
| **B** (not identical) | 2 | 4 runs, ~52 h | 4 runs, ~52 h | **~2.2 days** |

Both branches balance the machines evenly; that is why BRANCH-A can afford
three seeds in only one extra day. The rejected 8/4 arrangement — three
seeds under the by-seed rule — would have cost 4.3 days, and buys nothing
BRANCH-A does not buy more cheaply.

Plus the disclosed 20-round calibration phase per RCE run (~1 h each at
M=5), which unlike A2 does real work rather than recomputing a constant.

## 11. What A3 cannot establish, under either branch

Theorem 2's numerical bound. A3 makes the theorem's MAD condition
*exercisable* and measures what the margin does; it does not verify the
bound, which depends on a sub-Gaussian parameter the deployment cannot
observe. Also out of scope, unchanged from A2: `f > 1`, topology
generality, a second environment, dose response in `B`, over-reporting
liveness, and superiority over other robust aggregators. Under BRANCH-B,
additionally: any seed-level significance claim about the interaction.

---

## 12. Addendum — probe result and the BRANCH-B' amendment

**Added 2026-09-09, after the cross-platform probe, before any A3 run existed
on either machine.** Sections 1-11 above are the original pre-declaration and
are unedited. This section records what the probe measured and the one design
change it forced.

### 12.1 Probe result: BRANCH-B

Run on the Mac mini (Apple M4, 4P+6E cores, macOS 26.5, arm64) against the
committed Windows reference `results/a2_mechanism_check/mech_rce_clean`:

| level | result |
|---|---|
| 0. same experiment | PASS |
| 1. source seeds | **identical, all 4 rounds** |
| 2. policy checksum | **DIFFERS at round 0 and every round** |
| 3. source values | **DIFFERS**, `round0_max_abs_diff = 0.8856` |

Level 1 passing is what makes the branch decision meaningful: the integer
seed path agrees, so the configs agree and the differences are a property of
the hardware, not the setup. **BRANCH-B is selected.**

### 12.2 A correction to section 9's interpretation

Section 9 and the probe script both asserted that a round-0 level-3
difference is "pure numerics rather than accumulated divergence", because
both runs are still under the same policy at round 0. **That reasoning
silently assumed level 2 passes, and it does not.** PyTorch initialises
different weights on arm64 than on AMD64, so the two runs are under
*different policies from round 0*, and the round-0 value difference
confounds the initial weights with the float path. This probe cannot
separate them.

The branch decision is unaffected — either cause invalidates a cross-machine
paired contrast, and level 1 already rules out a config error — but the
claim as written was wrong and is not left standing. `scripts/a3_platform_probe.py`
now reports `round0_attributable_to_numerics_alone` alongside the difference,
and states the condition explicitly. Section 9's original text is preserved
above rather than rewritten.

The stronger conclusion this licenses: cross-machine CRN is not merely
degraded by float drift, it is **unavailable at the first step**, because
`torch.manual_seed` does not produce portable weights across these
architectures. A shared `seed_entropy` buys identical source *seeds* and
nothing downstream of them.

### 12.3 The measured speed asymmetry

Like-for-like on source collection (both 90 trajectories/round, and verified
flat across a full 250-round run — median s/round by 50-round block varies
only 85-92 s in the four completed A2 runs, so a 4-round basis extrapolates
soundly):

| | Windows | Mac mini | ratio |
|---|---|---|---|
| source s/round at M=3 | ~87 | **30.2** | **2.88x** |
| implied M=5 per-run | ~11.5 h | **~3.8 h** | |

### 12.4 BRANCH-B' — the amendment

Section 4's BRANCH-B assumed two comparably fast machines and therefore
split 1 seed each. At a 2.9x asymmetry that split is dominated:

| plan | Windows | Mac | critical path | seeds |
|---|---|---|---|---|
| BRANCH-B as declared: 1 seed each | 45.8 h | 15.1 h | 1.91 d | 2 |
| Mac 2 seeds + Windows 1 seed | 45.8 h | 30.3 h | 1.91 d | 3 |
| **BRANCH-B': all 12 runs on the Mac** | **0** | **45.4 h** | **1.89 d** | **3** |

**BRANCH-B' is adopted: A3 runs entirely on the Mac mini, all four
conditions, all three seeds, 12 runs, ~1.9 days.**

It is strictly stronger than BRANCH-B on every axis, at the same wall clock:

* **3 seeds, not 2.** `n = 3` parity with A1 and A2 is restored. The minimum
  attainable sign-test p returns to 0.250 from 0.500, and the 95 % CI
  halfwidth to 2.48 x sd from 8.98 x sd.
* **The machine confound disappears entirely.** Section 4 flagged a possible
  machine x condition interaction as a known-but-unproven limitation of
  BRANCH-B. With every run on one machine there is no machine factor at all,
  so that limitation is removed rather than merely bounded.
* **Section 8's `n = 2, directional` marker is withdrawn**, because it was
  conditioned on BRANCH-B and BRANCH-B is not what runs. A3-G5 is evaluated
  at `n = 3` under the section-6 rule exactly as written, unchanged.

**Why this is not goalpost-moving.** No A3 run exists on any machine. The
change is driven by a measured property of the hardware — the same class of
input section 4 already delegated the split to — and it moves the design
toward more evidence and fewer confounds, not fewer bars. No threshold in
sections 6-8 is altered, added, or relaxed. The A3-G5 decision rule, the
G3 margin bands, and the G7 mechanism bands are exactly as committed in
`c256f84` and `98da8ee`.

**What this costs.** The Mac must be available for ~2 continuous days, and
A3 loses the resilience of being spread over two machines: if the Mac is
reclaimed mid-campaign, A3 stalls rather than half-completing. The runs are
checkpointed and resumable, so a pause costs wall clock and not data.

**The Windows machine is freed** once A2 completes. What it runs next is a
separate decision and is not pre-declared here; nothing in A3 depends on it.

### 12.5 One incidental cross-platform confirmation

Over all 24 aggregation cells of the probe, both machines produced bitwise
identical RCE mechanism quantities —
`(spread, retained_n, applied_margin, degenerate) = (0.001, 1, 0.0015, True)`
— and identical `guarantee_calibration.json` with
`epsilon_offline = 0.001`, despite every underlying float differing.

Section 4 of `docs/a2_rce_gates.md` proves algebraically that these are
structural constants at `M=3, f=1`, independent of the input values. They
had never been checked on a second architecture. They survive one. This
sharpens rather than changes the A2 finding: at M=3 the margin is so inert
that it is invariant even to a change of CPU architecture.
