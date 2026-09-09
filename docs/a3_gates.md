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
same dual, same topology, same `d`, same `R_m`, same horizon, same three
seeds. `M` goes 3 -> 5 and nothing else moves.

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
architecture, so A3 collects its own `A'` and `B'`. 4 conditions x 3 seeds
= **12 runs**.

The decomposition is unchanged from A2 and is not renegotiable:

```
attack effect             =  B' − A'
clean RCE effect          =  E' − A'
attack effect under RCE   =  C' − E'
attack x RCE interaction  = (C' − E') − (B' − A')     <- primary
RCE main effect (raw)     =  C' − B'                  <- reported, never primary
```

## 4. Two machines — the assignment, and the rule that makes it valid

| machine | seeds | runs |
|---|---|---|
| Windows (12 logical CPU) | **0 and 1** | 8 |
| Mac mini | **2** | 4 |

**The rule: every contrast must be computed within one machine.** Each
machine runs *all four conditions* of the seeds it owns, so `B' − A'`,
`E' − A'`, `C' − E'` and the interaction are always differences between two
runs produced on the same hardware. A machine effect that shifts all four
cells of a seed equally cancels exactly in every one of those differences.

What this does **not** protect against is a machine x condition
*interaction*. That would require the hardware to affect the conditions
differently, which is implausible but not proven, so it is stated here as a
known limitation rather than assumed away.

**Cross-machine bit-identity is NOT assumed.** Source *seeds* are pure
integer arithmetic (`SeedSequence -> PCG64 -> integers`) and are identical
on any architecture; source *trajectories* run through MuJoCo and PyTorch
floats and may differ between x86-64 and ARM64. The A2-G1-ii style round-0
CRN check is therefore applied **within a machine only** (each condition
against its own-machine clean/attack partner), never across.

If the cross-platform probe (section 9) shows bit-identity, this
restriction can be lifted for future campaigns. It is not lifted for A3.

## 5. Seed pairing and common random numbers

Fresh source entropy per seed, by the G10 convention
`int(sha256(b"safelie/a3/source-entropy/seed=<k>").digest()[:16])`:

| seed | machine | attacked source (B' and C') | `seed_entropy` |
|---|---|---|---|
| 0 | Windows | `batch_1` | 25873748826208450093657097358164192435 |
| 1 | Windows | `batch_3` | 224463726283861287585842441982236519383 |
| 2 | Mac mini | `batch_5` | 11901184105024366541245543018062542394 |

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
  in `C'` exceeds that in `E'` at the same seed, in all three seeds.
  Predicted ratio 1.34; gate is the direction only.

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

`n = 3`, paired seed-level differences, exactly as A2. Primary is `I'`;
secondary are `E' − A'`, `C' − E'`, `C' − B'`. Paired t-tests are emitted
**labelled sensitivity-only** per decision D6; no significance claim is
made at three seeds regardless of p-value. No seeds are added after seeing
results. Seed 2 was produced on different hardware; that is disclosed in
every table, not just in this document.

## 9. Cross-platform probe (run once, on the Mac, before A3 starts there)

Run `configs/experiment/a2/_mechanism_check_rce_clean.yaml` on the Mac and
compare round-0 pre-attack source values against the committed Windows copy
in `results/a2_mechanism_check/mech_rce_clean`.

* **Identical to <= 1e-9** — cross-machine pairing is viable. Record it;
  future campaigns may split by condition instead of by seed. A3's
  by-seed split still stands as declared.
* **Not identical** — the by-seed split is load-bearing, exactly as
  designed. Record the observed magnitude.

Either way the probe is recorded in `results/a3_platform_probe.json`. It
does not gate A3.

## 10. Compute

12 runs x 250 rounds. `M=5, R_m=30` collects 150 trajectories per round
against A2's 90, so source cost is 1.67x: **75,000,000 source env steps per
run** against A2's 45,000,000. From A2's measured 8.03 h/run, expect
**~13 h/run**.

| machine | runs | estimate |
|---|---|---|
| Windows | 8 (seeds 0, 1) | ~104 h, ~4.3 days |
| Mac mini | 4 (seed 2) | ~52 h, ~2.2 days |

Plus the disclosed 20-round calibration phase per RCE run (~1 h each at
M=5), which at M=5 does real work rather than recomputing a constant.
