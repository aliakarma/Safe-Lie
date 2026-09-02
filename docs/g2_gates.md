# G2-peer: source-layer repair, pre-declared acceptance gates

**Status: DECLARED BEFORE ANY G2 RUN EXISTS.** Written and committed while
`results/runs_constraint_mc_g2/` does not yet exist; the commit that
introduces this file precedes every G2 run artifact. Nothing below may be
edited after a G2 number has been read. If a gate turns out to have been
badly chosen, the correct response is to record that it failed and say so,
not to move it.

Every numeric bar here is justified against a measurement that **already
exists on disk at declaration time**: the three completed G1 runs in
`results/runs_constraint_mc_g1/` (`g1_gates.json`, and
`source_diversity.json` per seed, produced by
`scripts/audit_source_independence.py`). Each justification below names
the G1 number it is derived from.

## What G2-peer is for

G1's verdict was FAIL (`results/runs_constraint_mc_g1/g1_gates.json`):
G1 (via G1d), G2 (via G2c), G5 (via G5b) and G7 (via G7d) all failed. The
own-critic estimator itself was vindicated -- G1a/b/c passed in all three
seeds, whole-run own-critic bias fell to -1.03 / +0.36 / +1.16 against a
pre-repair-estimator baseline of -6.9 to -8.8. But the **mechanism**
bias -- what the dual actually consumes -- stayed at -5.85 / -5.42 / -2.19,
outside the 0.20*d = 5.0 bar in 2 of 3 seeds. `g1_gates.md` pre-declared
exactly this failure mode and its cause: 4 of the 7 sources
(`peer_critic_1..4`) were left unchanged by G1 -- a peer's PPO cost-value
network (`AgentBundle.cost_value`, trained via GAE(lambda) against `ret_c`
for the PEER's own cost stream), evaluated at the OWNER's initial
observation.

`source_diversity.json` from the three completed G1 runs makes the
mechanism concrete. Averaged over all 6 owners x 4 peer offsets (24
(owner, source) pairs per seed):

| | seed0 | seed1 | seed2 |
| --- | ---: | ---: | ---: |
| `own_critic` bias | -1.03 | +0.36 | +1.16 |
| `own_critic` corr(pred, true) | 0.514 | 0.554 | 0.625 |
| `peer_critic` bias | -7.90 | -7.87 | -3.27 |
| `peer_critic` MAE | 15.99 | 15.12 | 14.47 |
| `peer_critic` corr(pred, true) | -0.098 | -0.074 | +0.008 |
| `monitor` bias | -4.17 | -3.41 | -1.71 |
| `monitor` corr(pred, true) | 0.652 | 0.661 | 0.690 |
| statistical effective M (nominal 7) | 1.84 | 1.93 | 1.78 |
| mechanism bias | -5.85 | -5.42 | -2.19 |
| mechanism MAE | 10.55 | 9.65 | 9.55 |
| mechanism RMSE | 13.32 | 12.21 | 11.94 |
| clean FSR (`P(mech<=d \| true>d)`) | 0.381 | 0.398 | 0.457 |

`peer_critic`'s correlation with true cost is approximately **zero**
(-0.10 to +0.01) -- worse than useless as an estimator of `J_C^i`, not
merely biased, because it was never trained toward any owner-specific
target at all: it is one agent's learned mapping from *its own*
observation distribution to *its own* GAE-biased cost target, force-fed a
different agent's observation. `own_critic` (already MC-repaired in G1)
and `monitor` (already MC-target-trained in G1) both clear corr >= 0.5.
This is the direct, on-disk evidence that G1d's cause is exactly what
`g1_gates.md` predicted, and it is what G2-peer repairs.

G2-peer asks exactly one question: **does replacing the peer-critic
source's underlying function -- from "a peer's PPO cost critic evaluated
out of distribution" to "a peer's own constraint-report head, trained on
the peer's own masked MC cost-to-go targets, evaluated at the owner's
observation" -- produce a mechanism estimate that actually tracks `J_C`,
without destabilising anything G1 established?**

A G2-peer pass licenses only the claim that the clean dual constraint
signal is no longer systematically corrupted by source-layer defects, so
that the attack study's eventual measured divergence is attributable to
the corrupted safety-feedback channel and not to a clean-baseline
artifact. It licenses nothing about attack stealth, adversarial
specificity, RCE, Theorem 2, topology invariance, genuine source
independence, or a second environment.

## The single scientific change

A new per-agent object, `safelie.training.loop.ExperimentRun.
constraint_report_heads: dict[AgentID, DiversifiedReplica]`, one
independently-initialized regression head per physical agent, refit once
per round on that agent's own `_cost_to_go_targets` (the same masked
discounted-MC cost-to-go targets `ensemble_replica`/`monitor` already use
under `constraint_estimator == "mc_window"`, unchanged by this repair).

* `peer_critic_<k>` sources now report
  `constraint_report_heads[peer_id].predict(owner_obs0)`, where `peer_id`
  is the existing, already-correct P0 #7 owner-relative mapping
  `(owner_index + k) mod N`. This is a straight swap of WHICH function is
  queried at the owner's observation (the query-input contract from
  `docs/assumptions.md`'s "Peer critic observability" decision is
  unchanged); it is no longer `AgentBundle.cost_value`, a network trained
  against `ret_c` for PPO's own advantage estimation.
* `own_critic` is **unchanged**: it already reports the raw discounted
  Monte-Carlo sum over the owner's own sampled cost
  (`mc_cost_return`), with zero function-approximation bias by
  construction. G1a/b/c already passed; there is nothing to repair.
* `ensemble_replica`/`monitor` are **unchanged**: they already refit a
  `DiversifiedReplica` on masked MC targets every round, per owner, per
  call. G1's own-critic-adjacent calibration numbers above (corr 0.65-0.69)
  already show this design works; peer_critic's new head reuses the exact
  same class for consistency, not because monitor's calibration needed
  fixing.
* `safelie.training.ppo`, `safelie.training.gae`, GAE's `gamma`/`lambda`,
  the cost critic's own regression target (`ret_c`), the threat model,
  RCE, attack magnitude, `d=25`, and the statistical methodology are all
  untouched. `DiversifiedReplica` gains `refit()`/`predict()` methods
  (split out of the existing `refit_and_predict()`, which now just calls
  them in sequence and is otherwise behaviourally identical) so a head can
  be fit once and queried by multiple owners in the same round -- required
  because a physical peer agent is queried by up to 4 different owners per
  round and must not be re-fit (and hence silently biased toward whichever
  owner asked most recently) on each query.

Nothing else changes. `results/runs/`, `results/runs_g0/` and
`results/runs_constraint_mc_g1/` are never written to and never merged
with the G2 directory.

## Run definition

Three seeds (0, 1, 2) of `configs/experiment/pilot_A_clean.yaml`,
unmodified except for `--seed` and `--output-dir
results/runs_constraint_mc_g2`. 250 rounds x 2000 steps; ManyAgent Ant;
N=6; d=25; M=7; ring topology; `attack: none`; `defense: mean`, f=0;
oracle evaluation every round. Identical to the G1 run definition in every
respect except the peer-source implementation described above.

## Gate G2a -- Source-target correctness (structural)

A unit-test gate, not a numeric threshold:
`pytest tests/unit/test_constraint_report_head_wiring.py` must pass in
full, covering, for N=6 and every owner/offset combination:

* no owner ever receives itself as a peer (already covered by
  `tests/unit/test_peer_critic_wiring.py`, re-asserted here against the
  new code path);
* the peer_critic value returned for `(owner, k)` equals
  `constraint_report_heads[peer_id].predict(owner_obs0)` computed
  independently in the test, for `peer_id` given by the existing closed
  form `(owner_index + k) mod N` -- i.e. the source is not silently wired
  to the wrong agent's head;
  * each agent's head is refit, once per round, on **that agent's own**
  `(obs, mc_cost_to_go)` pairs masked by `n_mc_targets` -- never another
  agent's rollout -- verified by making agents' cost streams
  distinguishable and checking the head's prediction tracks its own
  agent's data, not a peer's;
* a peer_critic report for owner A, computed with owner B's observation
  substituted in, differs from the correctly-wired report whenever A and B
  have distinguishable observations (catches "peer queried using another
  owner's state");
* `own_critic` and the attack/aggregation wiring from
  `tests/unit/test_dual_estimator_wiring.py` remain unaffected (regression
  guard: rerun of the existing G1 wiring tests, unmodified, must still
  pass).

**Gate G2a passes iff every test in the file passes**, plus the full
existing suite (`pytest tests/`) has no new failures relative to the
pre-G2 commit.

## Gate G2b -- Individual source calibration

Computed by `scripts/audit_source_independence.py` against each completed
G2 run, restricted to `source_type == "peer_critic"`, pooled over all 24
(owner, source) pairs, compared against **that same seed's** G1
`peer_critic` numbers above (the same relative-improvement convention
`g1_gates.md`'s G1a used against G0).

* **G2b-bias (relative).** `|mean bias_peer_critic_G2| <= 0.5 *
  |mean bias_peer_critic_G1|`, same seed, all three seeds.
  Thresholds: seed0 <= 3.949, seed1 <= 3.933, seed2 <= 1.636.
* **G2b-mae (relative).** `mean MAE_peer_critic_G2 <= 0.75 *
  mean MAE_peer_critic_G1`, same seed, all three seeds. A looser bar than
  bias's 0.5x because MAE also reflects the cross-agent out-of-distribution
  extrapolation variance this repair does not target (the query
  observation is still a different agent's state; only the regression
  TARGET is repaired) -- an MAE bar as strict as the bias bar would
  conflate "the target is fixed" with "the extrapolation is now precise",
  which this experiment does not claim. Thresholds: seed0 <= 11.990,
  seed1 <= 11.339, seed2 <= 10.853.

**Gate G2b passes iff G2b-bias and G2b-mae both hold in all three seeds.**
This gate is about `peer_critic` specifically, the sources this repair
touches; `own_critic` and `monitor` are reported for context but gate
nothing here since neither was broken by the G1d finding.

## Gate G2c -- Mechanism calibration

`mechanism_reported_cost_return` against `oracle.jsonl: true_cost_return`,
same rounds, same oracle episodes -- absolute bias, MAE and RMSE, per the
task's explicit instruction not to gate on correlation alone.

* **G2c-bias (absolute, G1's own bar, now ungated by G1d's exemption).**
  `|mean bias_mechanism| <= 0.20 * d = 5.0`, all three seeds.
* **G2c-mae (relative).** `mechanism MAE_G2 <= 0.75 * mechanism MAE_G1`,
  same seed. Thresholds: seed0 <= 7.915, seed1 <= 7.240, seed2 <= 7.163.
* **G2c-rmse (relative).** `mechanism RMSE_G2 <= 0.75 * mechanism
  RMSE_G1`, same seed. Thresholds: seed0 <= 9.989, seed1 <= 9.159,
  seed2 <= 8.954.

**Gate G2c passes iff G2c-bias, G2c-mae and G2c-rmse all hold in all three
seeds.** Correlation (`corr(mechanism, true)`) is still computed and
reported per section 8 of the task, but gates nothing, for the reason
`g1_gates.md`'s own G2c already documented: the oracle's round-to-round
sampling variance can dominate a correctly-calibrated estimator's
correlation with it.

## Gate G2d -- Clean false-safe rate

`FSR = P(mechanism_reported_cost_return <= d | true_cost_return > d)`,
computed per (round, agent) cell where `true_cost_return > d`, whole run,
compared against the same seed's G1 FSR above.

* **G2d-relative.** `FSR_G2 <= 0.5 * FSR_G1`, same seed. Thresholds:
  seed0 <= 0.191, seed1 <= 0.199, seed2 <= 0.229.
* **G2d-absolute.** `FSR_G2 <= 0.20`, all three seeds. 0.20 is chosen
  because it sits below half of G1's mean FSR (0.412) and represents
  "false-safe reads that occasional two-rollout sampling noise can
  produce" rather than "false-safe reads the mechanism produces on
  every-other unsafe round" -- the qualitative distinction section 19 of
  the task cares about (attack signature present by default, or not).

**Gate G2d passes iff G2d-relative and G2d-absolute both hold in all three
seeds. This is the single most important gate in this document** -- per
the task's own framing, a FAIL here blocks the attack study regardless of
what every other gate says.

## Gate G2e -- Constraint response

G1's Gate G3, reused verbatim (`docs/g1_gates.md` Gate G3), since the dual
update's arithmetic is untouched by this repair and should not need to be
re-derived, only re-verified against the new mechanism values:

* **G2e-a (identity).** `constraint_residual == mechanism_reported_cost_return
  - d` to within 1e-9, every round, agent, seed.
* **G2e-b (sign).** Restricted to interior (non-projected) cells, the sign
  of `lambda_{k+1} - (W lambda_k)_i` equals the sign of `residual_i` in
  `>= 99%` of such cells, all three seeds.
* **G2e-c (responsiveness).** `corr(residual_k, lambda_{k+1} - lambda_k) >
  0`, averaged over agents, all three seeds.

**Gate G2e passes iff G2e-a, G2e-b and G2e-c all hold.**

## Gate G2f -- Learning

G0/G1's Gate G4, reused verbatim, on `oracle.jsonl: episodic_task_return`,
first-20 vs last-20 rounds.

* **G2f-a.** `mean(last 20) - mean(first 20) > 0`, all three seeds.
* **G2f-b.** Averaged across seeds, that improvement is at least 1.0x the
  pooled standard deviation of the first-20 window.

**Gate G2f passes iff G2f-a and G2f-b both hold.**

## Gate G2g -- True-cost response

`head = mean(true_cost[first 20% of rounds])`,
`tail = mean(true_cost[last 20% of rounds])`, per seed, averaged over
agents.

* **G2g.** `tail <= 1.15 * head`, all three seeds -- true cost must not
  end the run more than 15% above where it started. This is deliberately
  NOT a requirement that true cost decrease (G1's G5b showed 2 of 3 seeds
  trending up over a run this short, and `docs/g1_gates.md` already
  declared why an initial-policy-relative velocity-threshold calibration
  at this pilot's scale does not guarantee monotonic convergence within
  250 rounds): it is a bound against *systematic divergence away* from
  the budget, which is what "the clean controller has negative-feedback
  behaviour" concretely rules out. 15% is chosen because it is tighter
  than G1's own worst-case excursion (seed0: 28.77 -> 35.63, +23.9%) but
  loose enough to tolerate G1's best-case seed's noise (seed1: 31.02 ->
  34.24, +10.4%) without redefining what "systematic" means after seeing
  G2 numbers.

Reported alongside, gating nothing: raw head/tail values, the lag between
a `lambda` increase and the following true-cost response (section 14 of
the task), and whether true cost ends the run below `d=25` (feasibility,
never gated per `g0_gates.md`/`g1_gates.md`'s standing declaration, since
5e5 steps is 1/20 of the paper's scale).

## Gate G2h -- Stability

G0/G1's Gate G6, reused verbatim:

* **G2h-a (numerics).** No NaN/Inf in any logged field, any round, agent,
  seed.
* **G2h-b (no lambda saturation).** `max lambda < 0.9 * lambda_max =
  22.5`, zero cells at `lambda >= 22.5`, all three seeds.
* **G2h-c (no runaway PPO).** Tail (last 20%) mean `approx_kl <= 0.05`,
  whole-run max `approx_kl <= 0.5`, all three seeds.
* **G2h-d (no entropy collapse).** Tail mean policy entropy `> 0.5`, all
  three seeds.
* **G2h-e (dual volatility tripwire).** Mean `|lambda_{k+1} - lambda_k|
  <= 0.45` (G1's own bar, 3x G0's), all three seeds. Rationale unchanged
  from `g1_gates.md`: a correctly-tracking mechanism is expected to have
  higher variance than the pre-G1 over-smoothed estimator, and that
  increase is the intended effect, not instability -- G2h-b is the clause
  that actually detects a controller destroyed by variance.

**Gate G2h passes iff G2h-a through G2h-e all hold.**

## Gate G2i -- Reproducibility

* **G2i-a.** All three runs reach round 249 with `run_metadata.json:
  status == "complete"`, `rounds.jsonl` and `oracle.jsonl` each holding
  exactly 250 sequential records, and a checkpoint present.
* **G2i-b.** `run_metadata.json` records `git.dirty == false` and the
  same `git.sha` for all three seeds, and
  `config_snapshot.constraint_estimator == "mc_window"` for all three
  (the config-level scientific setting is unchanged from G1; the repair
  lives entirely in code, so the commit SHA -- not a new config field --
  is what identifies which peer-source implementation ran).
* **G2i-c.** `scripts/analyze_matrix.py --runs-dir
  results/runs_constraint_mc_g2` reports no integrity problems.
* **G2i-d.** `scripts/audit_source_independence.py` runs to completion
  (exit code aside -- `assumption_1ii_supported` is expected and allowed
  to remain False, see "Reported in full, gating nothing" below) for all
  three seeds and produces a `source_diversity.json` in each run
  directory.

**Gate G2i passes iff G2i-a through G2i-d all hold.**

---

## Reported in full, gating nothing

* **Effective source diversity.** `statistical_effective_m_overall` per
  seed, alongside nominal `effective_M = 7`, exactly as G1 reported it.
  G1 measured 1.84 / 1.93 / 1.78. This repair changes WHAT `peer_critic`
  estimates, not how correlated its errors are with the other six
  sources' errors -- all seven remain critic-or-regression networks
  trained on rollouts from six agents facing near-identical dynamics in a
  highly symmetric environment, so a rise in `statistical_effective_m` is
  possible but not implied by fixing the target mismatch, and this
  document does not predict one. Whatever is measured is reported as-is,
  per section 4 of the task: fixing the target does not establish
  independence, and `assumption_1ii_supported` is expected to remain
  `False`.
* **Per-source bias/MAE/RMSE/correlation**, all seven sources, all six
  owners, both from `scripts/audit_source_independence.py`'s
  `source_diversity.json` and summarized by source type.
* **Cross-source error correlation matrix** and its eigenspectrum, per
  owner, per seed.
* **`corr(mechanism, true)`**, reported per section 8 of the task as a
  secondary diagnostic only, per the reasoning in Gate G2c above.
* **FSR by training phase, by seed, worst/best agent** -- the full
  breakdown behind Gate G2d's whole-run number, per section 13 of the
  task.
* **Lag between `lambda` changes and true-cost response** (section 14).
* **Feasibility** -- whether true cost ends below `d=25`.

## Verdict rule

* **PASS** -- G2a through G2i all pass.
* **CONDITIONAL PASS** -- exactly one gate fails, excluding G2a and G2d
  (both of which are automatic FAIL on their own if they fail: G2a
  because a structural wiring failure invalidates every other number in
  this document, G2d per the task's own section 19 stop condition), the
  failure has an identified cause, and that cause does not invalidate the
  others.
* **FAIL** -- two or more gates fail, or G2a or G2d fails on its own.

On FAIL or CONDITIONAL PASS with G2d unresolved: per the task's section
19, STOP. Do not launch the attack study. Identify the failed gate, find
the root cause, and report what would need to change -- do not relax a
threshold, discard a seed, or redefine FSR to pass.

Passing G2-peer does **not** license the attack study on its own; it
licenses the claim that the clean dual constraint estimator's *complete
source layer* -- not just the own-critic estimator G1 repaired -- tracks
`J_C` closely enough that the clean baseline no longer routinely produces
the false-safe pattern the eventual attack study needs to be able to
attribute to corruption rather than to a defective mechanism.
