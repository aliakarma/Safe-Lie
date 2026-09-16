# P1 — Concentrated attack / consensus-spreading: frozen protocol

**Status: FROZEN.** Written and committed before the first P1 production
round. Nothing below may be edited in the light of a P1 outcome. Amendments
are appended as dated sections at the end, never applied by rewriting.

**Campaign ID:** P1
**Output tree:** `results/runs_p1/`
**Date frozen:** 2026-09-16

---

## 1. Why this campaign exists

The external review's central methodological criticism is that the existing
attack does not instantiate the perturbation Proposition `cor:spread`
(main_iclr.tex, "Spreading") is stated for.

That criticism is correct, and this document records it as a **measurement,
not an opinion**. Under the G9+ parallel trajectory-batch source
architecture, one replica (`batch_m`) collects its own trajectories and
reports a value to **all six constraint owners**. A1's
`attack.corrupted_source_ids: [batch_1]` therefore corrupts one source in
*every* owner's report bundle, and under mean aggregation with `f=0` every
owner's aggregate shifts by the same `-B/M`. Recomputed from the committed
A1 logs (`results/runs_a1/B_seed0/rounds.jsonl`, all 250 rounds, all 6
owners):

```
delta_k = -4.16666666...  *  1      (uniform)
max |delta_k,i - (-B/M)| = 9.77e-15
```

Proposition `cor:spread` is stated for a **concentrated**
`delta_k = delta * e_j`. A1/A2/A3 therefore never instantiated its premise;
they instantiated the *uniform limit* that the proposition says connected
consensus drives the bias toward. The empirical role of consensus versus
`W = I` is consequently unidentified in the current manuscript.

P1 injects the concentrated perturbation and asks what the multipliers do.

## 2. Scientific question

> Does a localized corruption of one owner's reporting channel remain
> localized under non-communicating dual updates (`W = I`), and become
> distributed across multiplier coordinates when the same corrupted
> information is propagated through the existing connected ring consensus?

This campaign does **not** test universal topology invariance, does not
sweep `beta`, does not compare aggregators, does not test
`f`-misspecification, and does not test adaptive adversaries. Exactly two
topologies are run.

## 3. The attack definition (exact)

Identical in every taxonomy axis and in magnitude to the A1/A2/A3 primary
attack. One field is added.

| Property | Value |
|---|---|
| name | `primary` |
| direction | `negative` (under-reporting) |
| support | `persistent` (every round 0..249) |
| adaptivity | `static` |
| consistency | `consistent` |
| `f` | 1 |
| `budget_ratio` | 0.5, so `B = 0.5 * d = 12.5` |
| corrupted source | seed 0 → `batch_1`, seed 1 → `batch_2`, seed 2 → `batch_3` (A1's balanced mapping, **unchanged**) |
| **corrupted owner** | **`agent_5`, all three seeds (NEW)** |
| injection point | unchanged: the single hook in `safelie.training.loop`, on the return-scale residual, after source collection and before aggregation |

`corrupted_owner_ids` restricts the compromised reporting channel to the
named owner's report bundle. The adversary holds one **channel**
(`batch_m -> agent_5`) rather than the source outright. This is a *weaker*
adversary inside the *same* threat model — the same static, persistent,
directional, consistent `f=1` corruption at the same injection point — not
a new threat model and not an adaptive one.

**Predicted per-owner perturbation, mean aggregator, `M=3`:**

```
delta_k = [0, 0, 0, 0, 0, -B/M] = [0, 0, 0, 0, 0, -4.1666666...]
```

exactly, with the five untargeted entries equal to `0.0` bitwise, on every
round. Verified before launch (Section 9) and after every run (Section 7).

**Interpretive note fixed in advance.** Holding the per-coordinate
magnitude equal to A1's necessarily changes the total injected mass:
`1^T delta_k` is `-4.167` here against `-25.0` in A1. Concentration and
total mass cannot both be held fixed while changing localization; per
Theorem `thm:mass` the total mass is the topology-invariant quantity, so
the per-coordinate magnitude is the one matched to A1. No P1 claim may
compare P1's effect *size* against A1's.

## 4. The attacked owner, and why it is fixed

**Pre-declared selection rule:** the owner whose **clean-run mean
multiplier is largest**, i.e. furthest from the projection floor at
`lambda = 0`.

**Rationale.** The dual update projects onto `[0, lambda_max]`. A negative
(under-reporting) perturbation can only move a multiplier that is not
already pinned at zero; on an owner whose constraint is slack, both the
attacked and the counterfactual multiplier sit at the floor and the
perturbation is invisible for reasons that have nothing to do with the
topology. Computed on the three **already-committed clean runs** before any
P1 round was run, the per-agent mean `lambda` is

```
seed 0: (1.40, 1.08, 1.11, 1.45, 1.92, 2.09)
seed 1: (1.46, 1.10, 1.14, 1.56, 2.08, 2.25)
seed 2: (2.04, 1.71, 1.75, 2.14, 2.63, 2.79)
```

so the rule selects `agent_5` in all three seeds.

**This rule is computed from committed clean artifacts and frozen here; it
is not selected on any P1 outcome.** The owner is **not varied** across
seeds, so this campaign says nothing about the identity of the attacked
owner, and no P1 claim may generalize over owners.

## 5. Conditions, seeds, matrix

| Condition | Topology | `sigma_2(W)` | Attack | Seeds | Runs |
|---|---|---|---|---|---|
| **R** | `ring` (the existing W) | 0.666667 | concentrated, `agent_5` | 0, 1, 2 | 3 |
| **I** | `identity` (`W = I`) | 1.000000 | concentrated, `agent_5` | 0, 1, 2 | 3 |

**Total: 6 production runs.** No other topology is run in this campaign.

Configs: `configs/experiment/p1/{r,i}_seed{0,1,2}.yaml`. Each Condition-R
config differs from `configs/experiment/a1/b_seed{n}.yaml` in exactly
`run_id`, `output_dir`, and `attack.corrupted_owner_ids`; each Condition-I
config additionally changes `topology.name`. Verified mechanically before
launch.

Everything else is the frozen G10/A1 architecture: `manyagent_ant` N=6,
`velocity_threshold=0.75`, `cost_mode=per_agent_velocity`, `d=25`, `M=3`,
`R_m=30`, `constraint_estimator=mc_window`, mean aggregation `f=0`
(undefended), `eta_lambda=0.035`, `lambda_max=25`, 250 rounds at
`rollout_length=2000`, `R_ref=120` at rounds {25,75,125,175,225}, and each
seed's A1/clean `seed_entropy` (so round-0 source draws are bit-identical
across P1-R, P1-I, A1-B and the seed's clean run).

## 6. Clean references — what exists, what does not, and the consequence

This section records a **scientific limitation of the 6-run matrix that was
identified before launch**, per the instruction not to add runs silently.

**Available at no cost.** Three clean **ring** runs already exist and are
CRN-paired to these seeds:
`results/runs_constraint_batch_g9/g9_batch_clean` (seed 0),
`results/runs_constraint_batch_g10/seed{1,2}`.

**Missing.** There is **no clean `W = I` run**, and none is created by this
campaign.

**Why that matters.** Two routes to a clean reference for Condition I were
examined before freezing, and both are inadequate:

1. *Within-run counterfactual.* The manuscript defines the dual bias as
   `e_k = lambda_k - lambda_k^o` with `lambda^o` the trajectory "along the
   same primal sequence, under uncorrupted reports". That object is exactly
   computable from a run's own logs (the pre-attack source values are
   logged per owner, and replaying `clip(W lambda + eta r)` on committed
   clean logs reproduces the recorded multipliers to `0.0e+00`). But the
   manuscript also assumes both trajectories stay **interior** to
   `[0, lambda_max]`, and under `W = I` that assumption fails
   comprehensively: with no mixing, each coordinate independently integrates
   a residual with a nonzero mean and runs to a rail. Simulated on the three
   committed clean runs, the projected counterfactual bias is **identically
   zero in 72–100 % of the last 50 rounds for 5 of the 6 possible attacked
   owners**, and for the sixth it equals the full range `lambda_max`. The
   statistic is dominated by saturation, not by the mechanism.
2. *Raw within-run dispersion of `lambda`.* Confounded at the root:
   Proposition `prop:enforced` says `W = I` enforces the **per-agent**
   condition while consensus pins only the network average, so `W = I`
   produces intrinsically dispersed multipliers **with no adversary present
   at all**. Comparing raw dispersion between R and I without clean
   references measures the topology, not the attack.

**Consequence, pre-declared.** The 6-run matrix can support a claim about
Condition R against its existing clean ring pair, and can characterize the
**injected** perturbation and the realized multiplier structure in both
conditions. It **cannot** support an attack-attributable displacement claim
for Condition I. A clean `W = I` arm (3 further runs, ~29 h) is required for
the full 2x2 and is **recommended but not launched here**, because adding it
is a scope decision for the principal investigator, not for the
implementation. Until it exists, every Condition-I statement in Section 12's
vocabulary is restricted accordingly.

## 7. Primary measurements

Computed per round, per seed, per condition by `scripts/p1_analyze.py` from
`rounds.jsonl` and `oracle.jsonl`, and written to
`results/runs_p1/p1_report.json`.

1. **Injected perturbation `delta_k`** — per owner, measured as
   `aggregate(corrupted reports) - aggregate(uncorrupted reports)` under the
   same aggregator. Logged live as `injected_delta`.
2. **Realized multiplier vector `lambda_k`** — per agent, per round.
3. Mean, max, min multiplier displacement (Condition R, against its clean
   ring pair; Condition I, reported as levels with the Section 6 caveat).
4. **Dispersion across agents** — sd and IQR of `lambda_k` per round.
5. **Concentration / localization** — `max_i |v_i| / sum_i |v_i|`, on
   `delta_k` and on the displacement. `1.0` = one coordinate, `1/N = 0.1667`
   = perfectly uniform. NaN when the total mass is numerically zero (a
   vanished perturbation has no concentration, and reporting either extreme
   would invent a signature).
6. **Theoretical prediction under `W = I`**: `e_K = eta * delta * K * e_j`,
   concentration 1.0.
7. **Theoretical prediction under connected consensus**:
   `e_K = eta*delta*[(K/N) 1 + r_K]`, `||r_K||_2 <= (1-sigma_2)^{-1} = 3.0`
   for the ring, concentration → `1/N`.
   Both are computed by driving the *unprojected* linear recursion
   `e_{k+1} = W e_k + eta * delta_k` with the **actually injected**
   `delta_k`, and are reported **alongside** the projected quantity, never
   substituted for it.
8. **Aggregate multiplier displacement** `1^T e_K` (Theorem `thm:mass`).
9. **True network-average cost** from the withheld oracle (`oracle.jsonl`).
10. **Mechanism-level cost-report gap** — the mechanism gap, i.e. oracle
    true cost minus `mechanism_reported_cost_return`. (Memory note: this is
    ~4x the own-critic gap; the mechanism gap is the one reported.)
11. **Attack magnitude actually injected** — `sum_k 1^T delta_k`, from the
    per-round measured values, cross-checked against the ledger.

**Primary figure.** The six-agent `lambda`-displacement vector over time,
one panel per condition, **one line per agent**. Per-agent structure is
never averaged away.

## 8. The monitor

**Name.** *Reporting-residual / median-deviation monitor.* It must **not**
be called an adversarial-intent detector anywhere.

**Definition (from main_iclr.tex, the sentence after Prop. `cor:spread`):**
"an operator flagging any agent whose multiplier deviates from the fleet
median". Implemented in `safelie/eval/monitor.py` as

```
d_k^i = | lambda_k^i - median_j(lambda_k^j) |      per-agent statistic
D_k   = max_i d_k^i                                fleet statistic
flag at round k  iff  D_k > tau
```

The median is taken **across agents within a round**, never across rounds.

**RECORDED AMBIGUITY — the threshold.** The manuscript specifies the
statistic but **names no `tau`**. This is recorded here rather than papered
over. It is resolved without inventing a detector, by two pre-declared
devices:

1. **A threshold sweep.** Detection frequency is reported over the fixed
   grid `tau in {0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0}`, declared now. A
   conclusion is only stated if it survives the whole grid.
2. **A clean-calibrated `tau`, where and only where a valid clean reference
   under the SAME topology exists.** `tau` is the 95th percentile of `D_k`
   on the clean run, i.e. a 5 % false-positive rate by construction. This is
   available for **Condition R only** (Section 6). It is **not** computed for
   Condition I, because calibrating on a ring run and applying it to a
   `W = I` run measures the topology's intrinsic dispersion, not the
   adversary.

**Pre-declared caveat.** The monitor is a dispersion statistic. Under
`W = I` the multipliers are intrinsically dispersed with no adversary
present (Prop. `prop:enforced`), so a raised statistic under `W = I` is not
by itself evidence of corruption.

## 9. Pre-launch gates (all must pass before round 0 of the first run)

| Gate | Check | Status |
|---|---|---|
| P1-SMOKE | On the real production config: `delta_k ~ [0,..,-B/M,..,0]`, **not** `[-B/M,..,-B/M]`; off-target entries exactly `0.0` | **PASS** — `[0,0,0,0,0,-4.1666666667]`, concentration 1.000000 |
| P1-a | Only the named owner receives corruption | PASS (smoke + unit) |
| P1-b | Other owners bitwise unshifted (`aggregate == clean_point_estimate`) | PASS |
| P1-c | `W = I` is exactly `I`, doubly stochastic, `sigma_2 = 1`, non-mixing | PASS |
| P1-c | ring is connected, doubly stochastic, `sigma_2 = 0.6667 < 1`, mixes to `1/N` | PASS |
| P1-d | Config validation rejects unknown/empty/duplicate owners | PASS |
| P1-e | No RNG leakage: attack stream untouched under either scope | PASS |
| P1-REG | A1 path behaviourally inert: 2 rounds of `a1/b_seed0.yaml` under P1 code vs HEAD code, all pre-existing fields bitwise identical | **PASS** — 706 leaves, 0 mismatches, 36 additive P1-only fields |
| P1-SUITE | Full existing test suite (a3 tests excluded: they mutate committed artifacts) | PASS — 332 passed |
| P1-CFG | Each R config differs from its A1-B config in exactly `run_id`, `output_dir`, `corrupted_owner_ids` | PASS |
| P1-VIA | `W = I` viability: `lambda` is not pinned at a rail for the whole run | see Section 11 |

## 10. Reproducibility regression — result and a finding

`scripts/p1_a1_regression.py` compares a fresh run of
`configs/experiment/a1/b_seed0.yaml` against the committed A1 artifact.

**Finding, pre-existing and unrelated to P1.** Round 0 reproduces the
committed artifact exactly. Round 1 does **not**: `n_terminated` is 2 vs 0
and `mc_episode_lengths` is `[1000,7,7]` vs `[1000,1000]`, i.e. the **PPO
environment rollout** has drifted since the artifact's commit
(`a0075bc`). **The same divergence occurs on unmodified HEAD**, so it is not
caused by P1. On both rounds the attack → aggregate → dual path
(`point_estimate`, `constraint_residual`, `lambda_after`) reproduces
**bitwise**. P1's isolation is therefore established against HEAD-run-today
(`P1-REG` above), which is the stronger comparison; the drift of A1's PPO
rollout against its months-old artifact is logged here as a separate
repository issue and is **not** addressed by this campaign. **A1 production
is not rerun.**

## 11. Runtime

Measured on the run machine (this Windows laptop, 12 logical CPUs) from
`results/runs_a1/B_seed0/run_metadata.json`: `wall_clock_s = 35141`, i.e.
**9.76 h/run**, `140.6 s/round`, of which `123.6 s/round` is source
collection. This is **above** the 7.8 h/run figure assumed when the campaign
was scoped.

```
6 runs x 9.76 h = 58.6 h  (~2.4 days), sequential
```

The owner-localized attack does **not** change runtime: the corruption is
one scalar subtraction and the counterfactual aggregate is arithmetic over
`M=3` floats; all 45 M env steps of source collection per run are unchanged.
Training steps are **not** reduced.

## 12. Interpretation constraints (binding)

- **This campaign does not prove Proposition `cor:spread`.** The real
  learned policy diverges between conditions, and the projection onto
  `[0, lambda_max]` binds; the proposition's fixed-primal, interior
  assumptions do not hold in a live run.
- The strongest permissible form, **if and only if the data support it**:
  > Under a localized owner-level perturbation, `W = I` preserves
  > concentration of the injected dual perturbation, whereas connected
  > consensus redistributes it across multiplier coordinates, consistent
  > with the structure predicted by Proposition `cor:spread`.
- **Condition I is restricted by Section 6**: with no clean `W = I` run, no
  attack-attributable displacement may be claimed for it.
- **`n = 3`.** No inferential language. No "statistically significant", no
  population-level generalization, no universal topology claim. Per-seed
  trajectories, per-seed dispersion, means, and descriptive uncertainty
  summaries only. No p-values. The repository's five-seed floor for
  inference applies.
- One environment, one operating point, one attacked owner, two topologies.
- **Under `W = I` the coordinates of the dual update decouple exactly**, so
  a within-run counterfactual displacement is concentrated by algebraic
  identity, not by measurement. Any concentration figure derived from the
  counterfactual must say so. The empirical content of Condition I is in the
  **realized** multipliers, the monitor statistic, and the cost outcomes.
- If the results do not show spreading, **the negative result is reported as
  such** and the mechanism definition is not modified after the fact.

## 13. Success / diagnostic criteria

These are **diagnostic**, not pass/fail for publication.

- **D1 (injection).** `delta_k` concentration `= 1.000` on 100 % of rounds
  of all 6 runs, off-target entries exactly `0.0`. A failure here invalidates
  the campaign.
- **D2 (ring spreading).** Concentration of the Condition-R displacement
  against its clean ring pair declines from its round-0 value toward `1/N`.
- **D3 (identity localization).** Condition-I realized multipliers show the
  attacked owner separating from the fleet in a way Condition R does not.
- **D4 (monitor).** Detection frequency reported across the whole `tau`
  grid, both conditions, plus the clean-calibrated `tau` for R only.
- **D5 (mass).** `sum_k 1^T delta_k` agrees with `K * (-B/M)` to `1e-9`.

## 14. Stop conditions

Halt the queue, do not patch around, and report:

- any run whose per-round `delta_k` is not exactly `[0,..,-B/M,..,0]`;
- any non-finite value in `rounds.jsonl` or `oracle.jsonl`;
- duplicate source seeds within a run;
- a `theta_k` checksum failure (G9h);
- fewer than 250 rounds on a run marked complete;
- any attempt to write into a completed run directory.

## 15. Provenance

Recorded in each run's `run_metadata.json` by the existing orchestrator:
git SHA and dirty state, Python / PyTorch / MuJoCo / Gymnasium versions,
platform and CPU, the full resolved config snapshot, seed, topology,
attacked owner and source, timings, env-step counts, and the source-seed
audit. The frozen commit SHA for this protocol is recorded in Section 16.

## 16. Freeze record

- Protocol committed before the first P1 production round.
- Commit SHA: recorded in `results/runs_p1/p1_freeze.json` at launch.
- Files in the freeze: `docs/p1_concentrated_attack_gates.md`,
  `configs/experiment/p1/*.yaml`, `src/safelie/utils/config.py`,
  `src/safelie/training/loop.py`, `src/safelie/eval/monitor.py`,
  `scripts/p1_*.py`, `tests/unit/test_p1_concentrated_attack.py`.

---

## 17. Amendment, 2026-09-16 — saturation makes the concentration ratio
## non-monotone, and the analysis must show it

**Appended during R_seed0, at round 50 of 250. No pre-declared criterion,
condition, seed, metric definition or success bar is changed by this
amendment. It adds a diagnostic stratification and nothing else.**

**What was observed.** In the Condition-R run in flight, the observed
displacement's concentration ratio falls from 1.000 toward `1/N` as
predicted while the multipliers are interior, and then *rises* again once
the projection engages:

```
k    coords at floor (att/clean)   observed conc    linear conc
35            0 / 0                   0.3690          0.2072
40            0 / 0                   0.5456          0.2022
45            4 / 4                   0.6777          0.1984
```

**Why.** When a coordinate is pinned at `lambda = 0` in BOTH the attacked
and the reference run, its displacement is identically zero. The
concentration ratio `max_i|v_i| / sum_i|v_i|` then divides a smaller
support into the same numerator and rises **mechanically**. The rise is an
artifact of the projection, not a re-concentration of the perturbation: the
unprojected linear trace over the identical injected `delta_k` keeps
descending smoothly toward `1/N` throughout.

This is the same projection pathology Section 6 identified before launch
for `W = I`. Section 6 established it for the identity topology by
simulation; it is now observed in the **ring** arm as well, on live data.

**Consequence for reading the results.** A concentration number computed
over a window in which coordinates are pinned is not a measurement of
localization. In particular the last-50-rounds window, where the campaign's
summary statistics are taken, is expected to be the MOST saturated part of
the run (from the committed clean runs, `lambda` collapses toward 0-2 around
round 50 and 7-13 % of all cells sit at the floor).

**What is added.** `scripts/p1_analyze.py` now reports, alongside every
existing quantity and replacing none of them:

* `n_coords_at_floor` per round, for the attacked and reference runs;
* `interior_rounds`: the rounds in which NO coordinate is pinned in either
  run — the subset on which the concentration ratio is interpretable;
* concentration summarized **twice**, over all rounds and over
  `interior_rounds` only, each labelled;
* `saturation_fraction` next to every concentration figure, so a
  concentration number can never be quoted without the context that
  determines whether it means anything.

**What is NOT changed.** Diagnostic D2 still reads as frozen. No claim is
promoted or demoted. If the interior-window and all-rounds numbers disagree,
**both are reported**, and the disagreement is itself reported as the
projection's effect rather than resolved in favour of the more convenient
one.
