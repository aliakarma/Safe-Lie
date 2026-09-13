# DR — attack-dose / severity sweep over `B/d`

**Status: PRE-DECLARED. Written 2026-09-13, before any run of the two new
conditions exists.** Nothing below was chosen after seeing a dose-sweep
number, because no dose-sweep number exists yet. Every A1/G9/G10/U value
quoted is already committed and is reproduced here only so that the
comparison is fixed in advance rather than assembled afterwards.

**This experiment is NEW and EXPLORATORY relative to A1, A2, A3 and U.**
It is not a delayed part of any of them. Those four pre-declarations each
excluded dose response explicitly and by name:

| document | text |
|---|---|
| `docs/a1_attack_gates.md` §0 | "…and no attack-strength dose–response: `B/d = 0.5` only." |
| `docs/a1_attack_gates.md` §14 | out of scope: "other attack strengths (`B/d ∈ {0.25, 1.0}`)" |
| `docs/a2_rce_gates.md` §10 | "Cannot, and will not be claimed: … dose response in `B`" |
| `docs/a3_gates.md` §11 | "Also out of scope, unchanged from A2: … dose response in `B`" |
| `docs/submission_evidence.md` limitation 6 | "One attack point… no dose-response over `B/d`." |

The dose **grid** `B/d ∈ {0, 0.25, 0.5, 1.0}` is long-specified — by
`paper/main_iclr.tex:340`, by `AttackConfig.budget_ratio`'s `[SPEC]`
comment (`src/safelie/utils/config.py:136`), and by `docs/assumptions.md`
("the paper is explicit … and this repository implements it literally").
**A specified grid is not a pre-declared experiment.** No document before
this one fixed the hypothesis, the contrasts, the gates, the statistical
procedure or the verdict rule for a sweep over it. That is what this file
does, and it does it before the first new run starts.

Condition labels: **DR₀.₂₅** and **DR₁.₀₀**. They join the existing labels
A (clean, constrained), B (attacked at `B/d = 0.5`), C (attacked + RCE),
D (zero-mean perturbation), E (clean + RCE), U (unconstrained).

---

## 1. Question

**Does the downstream increase in true network-average cost grow
systematically with the magnitude of persistent directional cost
under-reporting?**

A1 established that persistent directional under-reporting at one
magnitude (`B/d = 0.5`) raises true network-average cost by +3.99 cost
units, sign-consistently across three seeds. A single magnitude cannot
distinguish two readings of that result:

* the effect is a **property of the corruption channel**, scaling with
  how much cost mass is injected; or
* the effect is an **artifact of that one magnitude** — a threshold
  crossed, a regime entered, a coincidence of scale.

Nothing in A1, A2, A3 or U separates them, because all four hold `B/d`
fixed at 0.5 (or 0). This experiment varies it and nothing else.

**The question is about systematic increase, not about functional form.**
Whether the relationship is linear, concave, saturating or stepwise is
**not** asked here and will not be answered here (§9.2).

## 2. Treatment

`attack.budget_ratio`, and nothing else.

```
B = attack.budget_ratio * d          (safelie.attacks.__init__.apply_attack:33)
J_hat_{C,m}^attack = J_hat_{C,m} − B      on the one corrupted source, every round
```

The attack **operator** is unchanged from A1:
`safelie.attacks.static.static_attack` with `direction="negative"`,
`support="persistent"`, `adaptivity="static"`, `consistency="consistent"`,
`f=1`. No new attack mechanism is introduced, no file in `src/safelie/` is
modified, and the injection point — after the source registry produces the
legitimate value, before mean aggregation and the dual update — is the
same single hook A1 used.

| `B/d` | `B = B/d · d` | `B/M` (predicted mean-aggregate shift) |
|---|---|---|
| 0.00 | 0.0 | 0.00000 |
| 0.25 | 6.25 | −2.08333 |
| 0.50 | 12.5 | −4.16667 |
| 1.00 | 25.0 | −8.33333 |

At `B/d = 1.0` the per-round corruption equals the entire per-agent budget
`d`. That is the largest level `main_iclr.tex:340` specifies and this
sweep does not go beyond it.

## 3. Condition matrix — 12 cells, 6 new runs

| `B/d` | condition | seeds | run dirs | status |
|---|---|---|---|---|
| 0.00 | A (clean) | 0,1,2 | `results/runs_constraint_batch_g9/g9_batch_clean`, `…g10/seed1`, `…g10/seed2` | **committed** |
| 0.25 | **DR₀.₂₅** | 0,1,2 | `results/runs_dose/B025_seed{0,1,2}` | **new** |
| 0.50 | B (A1) | 0,1,2 | `results/runs_a1/B_seed{0,1,2}` | **committed** |
| 1.00 | **DR₁.₀₀** | 0,1,2 | `results/runs_dose/B100_seed{0,1,2}` | **new** |

**No cell is dropped, no dose is dropped, no seed is dropped, and no
additional dose or seed will be added after seeing a result.** All twelve
are reported in the final table whatever they show.

### 3.1 Configuration, exactly

Six files, `configs/experiment/dose/b{025,100}_seed{0,1,2}.yaml`, each
being `configs/experiment/a1/b_seed{k}.yaml` with **three** fields changed:

| field | A1 condition B | DR |
|---|---|---|
| `run_id` | `B_seed{k}` | `B025_seed{k}` / `B100_seed{k}` |
| `output_dir` | `results/runs_a1` | `results/runs_dose` |
| `attack.budget_ratio` | `0.5` | **`0.25`** / **`1.0`** |

Only `budget_ratio` is a scientific field. The other two are identity and
filing. `scripts/dose_verify_frozen.py` proves this mechanically — it
loads each DR config and its seed's A1 condition-B config through the
current Pydantic schema, flattens both resolved models, and fails if any
key outside `{run_id, output_dir, attack.budget_ratio}` differs. It
additionally requires `seed`, `source_collection.seed_entropy`,
`attack.corrupted_source_ids`, `attack.name`, `attack.direction`,
`attack.support`, `attack.adaptivity`, `attack.consistency`, `attack.f`,
`defense.*` and `dual.*` to be **equal**, not merely permitted to differ.

Everything the G10 validation froze is inherited verbatim through A1:
`M=3`, `R_m=30`, pinned-policy parallel trajectory-batch collection, PPO,
GAE, the dual update, the ring consensus matrix, `d=25`, `η_λ=0.035`,
`λ_max=25`, `manyagent_ant` N=6 at `velocity_threshold=0.75`,
`cost_mode=per_agent_velocity`, mean aggregation with `f=0` on the
**defense** side (no RCE — this is the attack-without-defense line), 250
rounds at `rollout_length=2000`, the withheld `R_ref=120` reference at
rounds {25,75,125,175,225}, and the worker/chunk knobs (`workers: 12`,
`chunks_per_worker: 4`).

### 3.2 Which source is attacked — inherited, not re-chosen

A1 §3's balanced mapping, unchanged, at **every** dose:

| seed | corrupted source |
|---|---|
| 0 | `batch_1` |
| 1 | `batch_2` |
| 2 | `batch_3` |

The same source carries the corruption at 0.25, 0.5 and 1.0 for a given
seed, so source identity cannot confound the dose contrast any more than
it confounded A1's.

## 4. Seeds and randomness

`seed ∈ {0, 1, 2}` — the same three A1, A2 and U used, under the same
code-enforced policy (decision D6,
`safelie.analysis.stats.MIN_SEEDS_FOR_INFERENCE = 5`).

**Decision, taken before any dose run exists: this is a three-seed
experiment. Seeds 3 and 4 will not be added.** A1 fixed that rule for
itself on a pre-run power analysis and recorded that a tantalising
estimate is not grounds to extend an experiment but grounds to pre-declare
a new one. The same rule binds here. If a dose contrast comes out
underpowered, it is reported as underpowered.

### 4.1 Common random numbers — across all four doses

`source_collection.seed_entropy` is copied verbatim from A1 condition B,
which copied it from the paired clean run:

| seed | `seed_entropy` | shared by |
|---|---|---|
| 0 | `286314957402113664887331205920951063913` | A, B, DR₀.₂₅, DR₁.₀₀ |
| 1 | `52021175099534868945312500562751741008` | A, B, DR₀.₂₅, DR₁.₀₀ |
| 2 | `335268198726974212397240672597355197200` | A, B, DR₀.₂₅, DR₁.₀₀ |

All four doses at a given seed therefore draw the **identical** 23,100
`(env_seed, torch_seed)` pairs in the identical order, and `seed: k` gives
the identical policy initialisation and env stream. At **round 0**, before
the policies can diverge, the underlying source values are bit-identical
across all four doses and the only difference is the injected corruption.
That makes §7.2's cross-dose identity an exact check rather than a
two-sample comparison, and it makes every dose-vs-clean contrast paired at
the source-sampling level.

What stays independent is what must: the three replica streams remain
mutually disjoint **within** every run (23,100 unique env seeds, 0
duplicates, audited per run).

### 4.2 What the attack RNG does and does not touch

`static_attack` is deterministic — it draws no random numbers at all.
Unlike condition D's `benign_control`, the dose arms never touch
`SeedBundle.rng("attack")`, so the CRN pairing against clean and against
A1 is exact rather than approximate.

## 5. Comparability of the committed reference arms — checked, not assumed

The zero-dose arm ran at commits `73b1ae09` (seed 0) and `69aba30d` (seeds
1, 2); the `B/d = 0.5` arm at `a0075bc8` and `445bb9af`; the new runs will
run at `00861e31` or later. Three differences exist and all three were
checked rather than asserted.

**(a) Platform: identical.** All six committed reference runs record
`Windows-10-10.0.26200-SP0`, Python `3.11.9`, `git.dirty: false`. The new
runs execute on that same machine (12 logical CPUs, AMD Zen 3). No
cross-architecture contrast is being formed, so `docs/a3_gates.md` §15's
homogeneity requirement is satisfied by construction rather than by
scheduling.

**(b) Code: two additive deltas, both inert on this path.**

| delta | content | why inert here |
|---|---|---|
| clean → A1 | `select_corrupted_sources` gains an `explicit` argument; `AttackConfig.corrupted_source_ids` added | `explicit is None` returns the historical rule unchanged, which is what every clean config resolves to. A1 §1.1; pinned by `tests/unit/test_corrupted_source_selection.py` |
| A1 → HEAD | `pid_dual_update` + `DualConfig.controller/k_p/k_i/k_d`; RCE calibration-clone fix; `_provenance()` in run metadata | `controller` defaults to `"lagrangian"`, whose branch calls `dual_update` verbatim; the added work is `W @ zeros` and two zero arrays and **draws no RNG**. The calibration fix is gated by `defense.name == "rce"` (`loop.py:107`) and this line runs `defense: mean`. `_provenance()` writes metadata only. |

Reading is not evidence. **Gate DR-S0 (§7) discharges this empirically**
before the campaign starts, by re-running `a1/b_seed0.yaml` at HEAD for
two rounds and requiring bitwise equality against the committed `B_seed0`
rounds 0–1. If that fails, nothing launches.

**(c) Schema drift in `config_snapshot`, recorded not corrected.** The
snapshot is written by the software and is never hand-edited.
Consequently:

| run vintage | snapshot peculiarity |
|---|---|
| clean (G9/G10) | **lacks** `attack.corrupted_source_ids` — the field post-dates them. Resolves to `None` under the current schema, which is what `attack.name: none` means anyway. |
| A1 condition B | has `corrupted_source_ids`; **lacks** `dual.controller`, `dual.k_p/k_i/k_d` — those post-date it. |
| new DR runs | will record `dual.controller: "lagrangian"` and three `null` gains, and a `provenance` block A1's runs lack. |

**Therefore every frozen-config check in this experiment compares configs
loaded through the current schema, never snapshot-to-snapshot.** Loading
A1's and DR's YAML through today's `ExperimentConfig` resolves both to
`controller='lagrangian'` — the identical code path. This is the same
finding the U campaign recorded and it is restated here so that no future
reader mistakes a schema artifact for a treatment difference.

**No committed result is modified by this experiment.** A1's, G9/G10's and
A2/A3/U's artifacts are read, never rewritten.

## 6. Outcome measures — fixed before the data exists

**Primary:** whole-run **network-average expected true cost**
`(1/N) Σ_i J_C^i`, averaged over all 250 rounds. This is A1's primary
outcome (`docs/a1_attack_gates.md` §6.3), chosen there because the
consensus dual demonstrably enforces the network average and not the
per-agent budgets (Proposition 1). It is inherited, not re-selected.

**Primary contrast:** paired by seed,

```
ΔJ_true(dose) = J_true(dose) − J_true(clean)        for dose ∈ {0.25, 0.5, 1.0}
```

**Secondary, reported with per-seed values and CIs, never promoted:**
last-50 network-average true cost; the **mechanism-level gap**
`true_cost_return − mechanism_reported_cost_return`, whole-run and
last-50; per-agent true cost, all six, never pooled away; mechanism
aggregate; `λ̄`, `λ` max, `λ` activation and saturation fraction; residual;
source spread; task return, learning gain, first-20 and last-50; violation
rate and peak violation; the withheld `R_ref` calibration.

**Two windows, fixed now: whole-run (primary) and last-50 (secondary). No
third window will be introduced after the data exists.**

Note on the gap, restated so it is not rediscovered as a finding: the
quantity reported is the **mechanism** gap against
`mechanism_reported_cost_return`, not `oracle.jsonl:detection_gap`, which
`experiment.py` defines against the own critic and whose own comment says
it "measures critic estimation error, not the attack". The clean baseline
mechanism gap is ≈ +0.21 whole-run; the own-critic gap is ≈ +7.2 in clean
runs and would swamp any dose signal.
`scripts/reanalyze_mechanism_gap.py` established this on the committed
runs and the dose analyser uses the same field.

Violation rate is **descriptive only**, inherited from A1 §10: the paper's
constraint is an expected-cost constraint and no violation-rate bar is
imposed at any dose.

## 7. Structural gates — the stop conditions, checked per run

**Every one must pass exactly.** A failure halts the queue; the
implementation is fixed and the campaign restarts. These are correctness
checks, not statistics. `B` below means `budget_ratio · 25` for the run in
question, read from that run's own `config_snapshot` rather than from a
constant in a script.

| id | check |
|---|---|
| **DR-S0** | **Pre-campaign, once.** `a1/b_seed0.yaml` re-run at HEAD for 2 rounds is **bitwise identical** to committed `B_seed0` rounds 0–1 on every logged source value, aggregate, residual and `λ`. Discharges §5(b). Nothing launches if this fails. |
| **DR-S1** | every A1 structural gate (A1-S1…S10) passes for each new run, with `B` taken per dose |
| **DR-S2** | the attack is applied at the intended point: `rounds.jsonl` records **pre-attack** source values in `constraints.<agent>.reports[].value` and the **post-attack** aggregate in `mechanism_reported_cost_return` |
| **DR-S3** | the measured source shift is the dose's pre-declared `−B` (§7.1) |
| **DR-S4** | PPO env steps exactly `250 × 2000 = 500,000`, derived from `rounds.jsonl` and the config snapshot, **not** from `run_metadata.env_steps` (see §12); source env steps exactly 45,000,000 |
| **DR-S5** | source streams independent within the run: 23,100 unique env and torch seeds, 0 duplicates, 0 replica-pairwise overlap, 0 replica-vs-reference overlap |
| **DR-S6** | policy pinning: `worker_checksums_all_match` for all 48 chunks every round; 250 distinct θ checksums |
| **DR-S7** | oracle isolation: no learner-visible field equals the round's oracle `true_cost_return` |
| **DR-S8** | the attack does not touch the environment cost, and the withheld `R_ref` reference is **un**shifted — no `−B` offset at any dose |
| **DR-S9** | CRN integrity: each DR run shares all 23,100 source seed pairs, in order, with its seed's clean run and its seed's A1 B run |
| **DR-S10** | `scripts/dose_verify_frozen.py` exits 0 — nothing but `budget_ratio` differs from A1 condition B |
| **DR-S11** | **no accidental RCE**: `defense.name == "mean"`, `defense.f == 0`, no `rce_calibration.json` written, no calibration phase in the log |
| **DR-S12** | **correct dose**: the run's `config_snapshot.attack.budget_ratio` equals the value this document assigns to its `run_id`, and `attack.{name,direction,support,adaptivity,consistency,f}` are unchanged from A1 |

### 7.1 The in-run mechanism identity (DR-S2/S3) — exact, not approximate

Within any attacked run, on every round `k` and every owner `i`:

```
mean_m( reports[m].value ) − mechanism_reported_cost_return = +B/M
```

required to hold to **≤ 1e-9** on **100 %** of the 250 × 6 = 1500
(round, owner) cells, with

```
B/M = 2.08333…   at B/d = 0.25
B/M = 8.33333…   at B/d = 1.00
```

### 7.2 The cross-dose round-0 identity (DR-S9) — exact

§4.1 makes the round-0 draws bit-identical across all four doses at a
given seed. Two **different** logged quantities are checked at round 0
against **different** expectations, and conflating them is the one
mistake this subsection exists to prevent:

**(a) `reports[].value` is logged PRE-attack, in every condition.** Under
CRN it is therefore bit-**identical** across all four doses and the clean
run — *including the corrupted source*. This is the CRN integrity check,
not the attack check. To ≤ 1e-9, for every owner and every source:

```
reports[m].value(dose) − reports[m].value(clean)   =  0
reports[m].value(dose) − reports[m].value(A1 B)    =  0
```

**(b) `mechanism_reported_cost_return` is written POST-hook** and carries
the entire injected shift. To ≤ 1e-9, for every owner:

```
mechanism_reported(dose) − mechanism_reported(clean)  =  −B/M
                                                      =  −2.08333  at B/d = 0.25
                                                      =  −8.33333  at B/d = 1.00
mechanism_reported(dose) − mechanism_reported(A1 B)   =  −(B − 12.5)/M
                                                      =  +2.08333  at B/d = 0.25
                                                      =  −4.16667  at B/d = 1.00
```

Together these verify that the underlying draws are shared and that the
dose is what the config says it is, at the injection point, against two
independent references.

**Provenance of this subsection.** An earlier draft of it asserted the
`−B` shift on `reports[].value`, copying the notation of
`docs/a1_attack_gates.md` §7.1. That is wrong about the logged field, and
`scripts/dose_verify_run.py` was written to match it. The error was caught
**before this document was frozen and before any dose run existed**, by
running the new verifier against A1's three committed `B_seed{k}` runs,
whose gate outcome is already known: the bad check failed on runs that
must pass. Both the document and the verifier were corrected to the form
above, which is what A1's own committed checker
(`scripts/a1_mechanism_validation.cross_run_round0`, and its comment
"`reports[].value` is PRE-attack in both runs, so with CRN it must be
identical — this is the CRN integrity check, not the attack check")
already implemented.

**No A1 result is affected**, and `docs/a1_attack_gates.md` is not edited:
A1's *code* was always correct and only its prose is loose. See §12.3.

## 8. Scientific gates

`d = 25`, `M = 3`, `N = 6`. Every gate is evaluated on **paired per-seed
differences**. All bars below are fixed now.

### DR-G1 — the primary question: does the effect grow with dose?

* **G1-i** *(gating)*: `ΔJ_true(dose) > 0` in **all 3 seeds** at **each**
  of `dose ∈ {0.25, 0.5, 1.0}`, whole-run.
  *Bar:* sign consistency across all seeds is the strongest claim n=3
  licenses under decision D6. `dose = 0.5` is already known to satisfy
  this (+3.84 / +4.32 / +3.80); it is included so the gate is stated over
  the whole grid rather than only over the new cells.

* **G1-ii** *(gating — the monotonicity criterion)*: the per-seed sequence
  is **strictly increasing across the whole grid**,

  ```
  J_true(0, s) < J_true(0.25, s) < J_true(0.5, s) < J_true(1.0, s)
  ```

  for every seed `s ∈ {0,1,2}` — that is, **all 9 adjacent-dose increments
  positive**, whole-run.
  *Bar:* this is the criterion for "systematic", and it is deliberately a
  statement about ordering rather than about functional form. It is
  **not** a linearity test and passing it licenses no claim about shape
  (§9.2).

* **G1-iii** *(gating)*: the paired mean of
  `ΔJ_true(1.0) − ΔJ_true(0.25)` exceeds **+1.0 cost units**, with its
  95 % paired CI reported.
  *Bar:* +1.0 is A1-G1-ii's bar verbatim — 4 % of `d`, and 7× the
  committed clean between-seed sd of **0.143**. Reusing it rather than
  inventing a new one is deliberate.

**DR-G1 passes iff G1-i, G1-ii and G1-iii all hold.**

### DR-G2 — the mechanism-level gap by dose

* **G2-i** *(gating)*:
  `Δmech_gap(dose) = mech_gap(dose) − mech_gap(clean)` positive in all 3
  seeds at each dose, whole-run.
* **G2-ii** *(gating)*: `Δmech_gap` strictly increasing across the grid in
  all 3 seeds — the same 9-increment criterion as G1-ii.
* **G2-iii** *(reported, non-gating)*: absolute `mech_gap` for every dose
  and seed, so the baseline stays visible. Committed clean values are
  +0.45 / −0.01 / +0.19 whole-run; at `B/d = 0.5`, +4.56 / +4.50 / +4.21.

### DR-G3 — the mechanistic expectation (REPORTED, NOT GATED, NOT FITTED)

The closed loop suggests an arithmetic expectation: the dual drives the
**reported** aggregate toward `d`, and the corruption depresses that
aggregate by `B/M`, so true cost should settle near `d + B/M`, giving

```
ΔJ_true(dose)  ≈  (B/d) · d / M  =  8.3333 · (B/d)
```

| `B/d` | expectation | committed observation |
|---|---|---|
| 0.25 | +2.083 | — |
| 0.50 | +4.167 | **+3.986** (ratio 0.956) |
| 1.00 | +8.333 | — |

**This is a falsifiable mechanistic expectation, not an assumed result and
not a model that will be fitted.** No slope is estimated, no regression is
run, and no `R²` is reported. What is reported is the per-dose, per-seed
ratio `ΔJ_true(dose) / (8.3333 · B/d)` as a descriptive diagnostic. The
account is recorded as **consistent** if every such ratio lies in
`[0.7, 1.3]` and **flagged** otherwise — and a flag is a finding to
describe, not a failure: attenuation at high dose by the valid-source
architecture is a legitimate outcome (§10, Outcome C) and `B` will not be
re-tuned after observing it.

The unconstrained control bounds the whole question and is quoted here so
the bound is fixed in advance: `J_true(U) = 81.60 / 81.51 / 84.95` against
`J_true(A) = 25.21 / 24.96 / 24.98`, i.e. `ΔJ_true(U−A) ≈ +57.6`. The
largest expectation in the table above, +8.33, is 14 % of that, so **no
ceiling effect is expected inside this grid.** If one is observed, that is
itself a reportable finding about where the mechanism saturates.

### DR-G4 — the corruption reached the dual, and only through the dual

* **G4-i** *(gating)*: §7.1's in-run identity holds on 100 % of cells at
  every dose, and §7.2's cross-dose round-0 identity holds to ≤ 1e-9.
* **G4-ii** *(gating)*: the corruption must **lower** the multiplier:
  `λ̄(dose) < λ̄(clean)` in all 3 seeds at every dose. An under-reporting
  attack that raised `λ` would mean the causal story is wrong even if true
  cost rose. Committed: `λ̄(A)` = 1.51 / 1.60 / 2.18, `λ̄(B)` = 1.03 / 1.30
  / 1.69.
* **G4-iii** *(reported, non-gating)*: whether `λ̄` is itself monotone
  **decreasing** in dose, plus `λ` activation and saturation fractions,
  cumulative injected bias `K · B`, and measured-vs-predicted aggregate
  shift. Not gated because `λ̄` is a function of the policy as well as the
  corruption, and a non-monotone `λ̄` under a monotone cost response is a
  describable outcome rather than a broken run.

### DR-G5 — per-agent increments

* **G5-i** *(reported, non-gating)*: `J_C^i(dose) − J_C^i(clean)` for all
  six agents, every dose, every seed, with CIs and with the count of
  agents over `d`. Committed: 2.3 of 6 agents over `d` in clean last-50,
  5.7 of 6 at `B/d = 0.5`.
* **G5-ii** *(gating)*: the sign of the per-agent increment agrees with
  the network-average increment for **at least 4 of 6** agents in **all 3
  seeds** at **each** dose. *Bar:* A1-G3-ii verbatim, and deliberately not
  6 of 6 — the six owners' `J_C` values correlate 0.85–0.92 within a
  rollout, so this is ≈1 effective test per seed, not 6.

### DR-G6 — learning health (the run must still be a run)

Inherited from A1-G6 / G10-C, unchanged and unrelaxed: no NaN/Inf; PPO KL
median ≤ **0.0069** and p95 ≤ **0.01548**; `λ` saturation fraction
< **0.05**; entropy declines and stays finite and positive; θ checksum
distinct in 100 % of rounds.

Task-return gain is **reported but not gated**. A corruption magnitude
that degrades learning is a legitimate outcome (§10, Outcome D), not a
broken run, and gating it would make that outcome unreachable. Committed
reference: clean learning gain +153.2 mean; at `B/d = 0.5`, +144.3 (paired
B−A = −8.85).

### DR-G7 — task performance and stealth

Predeclared **before** analysis, inherited from A1 §12 verbatim.
Equivalence margin `Δ_equiv = 15.3` task-return units = 10 % of the
committed clean learning gain of +153.15.

* **Primary:** TOST on the paired `dose − clean` difference in last-50
  task return against ±15.3, **per dose**. Equivalence is claimed **only**
  if TOST establishes it. `p > 0.05` on a difference test is **never**
  reported as stealth.
* **Secondary, reported:** the paper's literal criterion
  (`main_iclr.tex:298`), "return within one standard deviation of the
  no-attack baseline" — one clean between-seed sd = **1.36** return units.
  n=3 has essentially no power to establish equivalence at it; that is
  reported, not worked around.
* A1's committed result at `B/d = 0.5`: TOST `p = 0.0047`, equivalent at
  `Δ_equiv`. Whether that survives at `B/d = 1.0` is an open question this
  experiment reports and does not prejudge.

## 9. Statistics — fixed before the data exists

### 9.1 The procedure

The design is paired by seed at every dose (§4.1), so all analyses are
paired. Decision D6 (`MIN_SEEDS_FOR_INFERENCE = 5`) is **not relaxed**:

* **Primary evidence is per-seed sign and ordering consistency** across
  the three paired differences, as gated in §8.
* The **paired mean difference and its 95 % CI** are reported for every
  contrast as estimates: `ΔJ_true(0.25)`, `ΔJ_true(0.5)`, `ΔJ_true(1.0)`,
  and the three adjacent-dose increments.
* A **paired t-test is reported only as a clearly-labelled sensitivity
  analysis**, never as a primary claim, alongside the exact sign test,
  whose minimum attainable two-sided p at n=3 is **0.25** — stated here so
  it is not mistaken for a null result.
* **No p-value from this experiment at n = 3 establishes significance, and
  none will be presented as doing so.** No conventional p-value outside
  this sensitivity role is part of the plan, because none was part of the
  plan A1/A2/A3 established.
* **No multiplicity correction is applied at n=3**, for the same reason A1
  did not apply one: Holm across a family presupposes inferential tests,
  and there are none here. If the seed count ever reaches 5, the family is
  `{ΔJ_true(0.25), ΔJ_true(0.5), ΔJ_true(1.0)}` corrected by Holm
  step-down with `ΔJ_true(1.0)` designated primary **in advance**. There is
  no other family and no test outside it will be promoted after the fact.

### 9.2 What will NOT be estimated

**No functional form will be fitted or claimed.** Specifically:

* no linear regression of `ΔJ_true` on `B/d`, no slope, no intercept, no
  `R²`, no correlation coefficient;
* no interpolation or extrapolation to doses outside
  `{0, 0.25, 0.5, 1.0}`;
* no claim of linearity, proportionality, concavity, convexity or
  saturation.

Four dose levels at three seeds cannot distinguish these, and the design
was not powered to. The pre-declared claim is confined to **ordering**:
whether the effect increases systematically with magnitude. A dose-response
**plot** will be produced (per-seed points plus paired means with CIs, with
the §8 `8.3333 · B/d` expectation drawn as a reference line, clearly
labelled as an expectation and not a fit), because showing four points is
not the same as modelling them.

### 9.3 Power, from the committed A1 paired variance

| quantity | committed value |
|---|---|
| clean between-seed sd, whole-run `J_true` | **0.143** |
| paired sd of `B−A`, whole-run `J_true` | **0.288** |
| paired sd of `B−A`, last-50 `J_true` | **0.880** |
| smallest adjacent-dose increment expected (§8 DR-G3) | **≈ +2.08** |

The smallest increment the monotonicity criterion must resolve is ≈ 7× the
committed whole-run paired sd and ≈ 2.4× the last-50 paired sd.
**Whole-run is therefore the gated window and last-50 is reported only.**
That assignment is made here, before the data, and not after seeing which
window looks better.

Caveat stated in advance: the A1 paired sd is the best available prior, not
a prediction. It was measured between a clean arm and an attacked arm; the
increments between two **attacked** arms (0.25 → 0.5, 0.5 → 1.0) could be
noisier. If they are, the gate is harder to pass, not easier, and a failure
to resolve them is reported as INCONCLUSIVE (§10), never rescued by adding
seeds.

## 10. Pre-defined interpretation and verdict rule

### 10.1 Outcomes, fixed before the data exists

* **Outcome A — systematic dose response.** DR-G1 passes: every dose
  raises true cost in every seed, and the ordering is strictly increasing
  across the grid in every seed. The A1 effect is a property of the
  corruption magnitude, not an artifact of one setting.
* **Outcome B — effect present but not ordered.** DR-G1-i passes, G1-ii
  fails: corruption raises cost at every dose but the increase does not
  track magnitude across the grid. Reported as such; no trend claimed.
* **Outcome C — attenuation or saturation.** The effect is present but
  falls materially short of the §8 DR-G3 expectation at the high dose.
  This is a **legitimate scientific result about the valid-source
  architecture**, and `B` will not be re-tuned after observing it.
* **Outcome D — learning degrades at high dose.** Even if true cost rises,
  `B/d = 1.0` is then reported as **disruptive rather than stealthy**, and
  DR-G7's TOST result is reported per dose rather than pooled.

### 10.2 Verdict rule

* **PASS** — DR-S0…S12 pass for both new doses in all three seeds, **and**
  DR-G1, DR-G2, DR-G4 and DR-G5-ii pass.
* **INCONCLUSIVE** — all structural gates pass and DR-G4 passes (so the
  corruption provably reached the dual at the intended magnitude), but the
  DR-G1-ii ordering breaks in one or more seeds **and** the 95 % paired CI
  for `ΔJ_true(1.0) − ΔJ_true(0.25)` contains 0. That is the signature of
  "too noisy to establish a trend", and it will be reported with that
  wording rather than as a negative result.
* **FAIL** — any structural gate fails; **or** DR-G4-i fails (the injected
  magnitude is not what the config declares); **or** DR-G4-ii fails (the
  corruption did not lower the multiplier); **or** `ΔJ_true(1.0) ≤ 0` in
  two or more seeds.

A FAIL on the last clause is a **scientific** result about the mechanism,
not a broken experiment. It will be reported as one, and no dose will be
re-chosen in response to it.

**No success criterion in this document will be redefined after the
results are seen.** No dose, seed, metric or evaluation window will be
selected, dropped or added post hoc.

## 11. What this experiment may and may not claim

**Permitted, if DR-G1 passes:** "the increase in true network-average cost
produced by persistent directional under-reporting grows systematically
with the magnitude of the under-reporting, across `B/d ∈ {0, 0.25, 0.5,
1.0}`, at three seeds, in this environment, without defense."

**Forbidden, in every outcome:**

* any claim about **adversarial intent**. The mechanism is persistent
  directional corruption / under-reporting. A1 §9 governs and is
  inherited: "adversarial specificity" does not appear in these
  conclusions, and neither does "malicious".
* any claim about **functional form** — linear, proportional, or otherwise
  (§9.2).
* **generality**: `M = 3`, `f = 1`, `N = 6`, ring topology,
  `manyagent_ant`, `d = 25`, 250 rounds, mean aggregation, no RCE. Nothing
  here speaks to other environments, topologies, `M`, `f`, aggregators, or
  to RCE's behaviour under dose — A2/A3 hold `B/d = 0.5` and this
  experiment does not extend them.
* any **statistical significance** claim at three seeds (§9.1).
* any claim about doses **outside** the tested grid.

## 12. Known infrastructure defects carried into this campaign

Documented here in advance so that neither is discovered mid-run and
mistaken for a new problem. Neither is a scientific-method issue and
neither changes any number.

1. **`run_experiment_with_oracle` does not accumulate `env_steps` or
   `timing` across a resume.** It rewrites `run_metadata.json` with the
   resumed segment's counters only — the defect that clobbered `D_seed0`'s
   metadata during A1
   (`results/runs_a1/D_seed0/METADATA_INCIDENT.json`). **Still unfixed at
   HEAD.** `src/safelie/experiment.py` is part of the frozen training
   implementation and will **not** be modified during this campaign;
   changing it mid-experiment would make early and late runs products of
   different code. Mitigations, both inherited from the A1 queue:
   `scripts/dose_run_queue.py` never re-enters a run already at 250 rounds
   (it gates only), and DR-S4 derives step counts from `rounds.jsonl` and
   the config snapshot rather than from `run_metadata.env_steps`.
2. **`tests/unit/test_configs_load.py` globs `configs/experiment/*.yaml`
   without recursing**, so configs in `a1/`, `a2/`, `a3/`,
   `unconstrained/` — and now `dose/` — are not schema-checked by that
   test. Rather than widen a shared test mid-campaign,
   `scripts/dose_verify_frozen.py` loads and validates all six DR configs
   through the current schema as gate DR-S10, which is strictly stronger
   for this campaign's purposes.

### 12.3 A documentation defect in A1's pre-declaration, reported not fixed

`docs/a1_attack_gates.md` §7.1 writes its cross-run round-0 identity as

```
J_hat_m^attack − J_hat_m^clean = −B    (attacked source, ≤1e-9)
```

Read as a statement about the logged field `reports[].value`, that is
false: `reports[].value` is written **pre**-attack, so at round 0 under
CRN it is bit-identical between the clean and attacked runs — measured,
`0.00e+00` on all three sources for seed 0 — and the `−B/M` shift appears
in `mechanism_reported_cost_return` instead. The notation is evidently
meant as the post-hook value, which is not what gets logged.

**A1's result is unaffected and nothing in A1 is being corrected.** A1's
gate was evaluated by `scripts/a1_mechanism_validation.cross_run_round0`,
which checks the two quantities correctly and says so in its own comment.
The defect is confined to the document's prose.

`docs/a1_attack_gates.md` is a **frozen pre-declaration and is not
edited** — this campaign records the discrepancy here instead, which is
the same convention A3 §12 used when a probe superseded part of its own
pre-declaration. The corrected identity this campaign gates on is §7.2
above.

## 13. Execution

**Sequential.** The source collector already saturates all 12 logical CPUs
(`workers: 12, chunks_per_worker: 4`); concurrent runs would not finish
sooner and would make the per-run stop rule unusable. The repository has
never established that its parallel execution is bitwise-invariant across
*runs*, only across chunk partitions within a run, so runs are not
parallelised.

**Order, and why:**

```
1. B100_seed0     2. B025_seed0
3. B100_seed1     4. B025_seed1
5. B100_seed2     6. B025_seed2
```

`B/d = 1.0` runs first at each seed because it is the condition most likely
to behave qualitatively differently — `λ` flooring at zero, learning
degradation — and a design problem there should surface after ~8 h rather
than after ~23 h. Pairing it immediately with that seed's 0.25 run means
that after every **even**-numbered run the grid is complete for one more
seed: at 2 runs there is a full four-dose curve at seed 0, at 4 runs two
seeds, at 6 runs three. No intermediate stopping point leaves a dose
present at some seeds and absent at others.

**Compute.** 6 new runs. Anchored on the 9 most recent comparable runs on
this machine (A1's six at 6.54–9.76 h, U's three at 7.52–8.02 h; mean
**7.67 h**), each 45 M source env steps + 500 k PPO + 500 k oracle:

> **≈ 46 h sequential wall clock (1.9 days), range 39–59 h**, plus 270 M
> source environment steps.

The training budget is not reduced, the horizon is not shortened, and no
code is optimised for speed. Gate DR-S0's 2-round regression adds ~4
minutes.

## 14. Out of scope

This experiment establishes nothing about: RCE or any defense under dose;
Theorem 2's numerical bound; topology generality; `f > 1`; `M = 5`; other
environments; over-reporting (`direction: positive`); adaptive, selective
or Byzantine corruption; baseline algorithms including PID-Lagrangian;
doses outside `{0, 0.25, 0.5, 1.0}`; and the functional form of the
dose-response relationship.

The manuscript is not edited on the basis of this experiment until its
report has been reviewed.
