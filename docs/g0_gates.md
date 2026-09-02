# G0 clean-baseline validation: pre-declared acceptance gates

**Status: DECLARED BEFORE ANY G0 RUN EXISTS.** Written and committed
while `results/runs_g0/` did not yet exist; the commit that introduces
this file precedes every G0 run artifact. Nothing below may be edited
after a G0 number has been read. If a gate turns out to have been badly
chosen, the correct response is to record that it failed and say so, not
to move it.

## Provenance, stated honestly

The P0 repair report proposed gates of exactly the shapes below. That
report was a conversational deliverable and was never written into this
repository, so its numeric thresholds cannot be recovered verbatim and
are **re-declared here** rather than quoted. The shapes (which window,
which quantity, which direction) are as originally specified; the
numeric bars are fixed here, before any G0 data exists, and each is
justified below against a measurement that already existed at
declaration time (the pre-repair pilot matrix and the post-repair
synthetic smoke run, both recorded in `docs/assumptions.md`). Treat the
thresholds as pre-registered as of this commit, not as independently
pre-registered before the P0 work.

## What G0 is for

G0 asks one question: **does the repaired implementation produce a
functioning constrained MARL learner whose clean baseline is
scientifically meaningful?** It is not a test of the paper's attack
hypothesis, of RCE, of Theorem 2, or of stealth. A G0 pass licenses only
the claim that later experiments are *interpretable*.

## Run definition

Three seeds (0, 1, 2) of `configs/experiment/pilot_A_clean.yaml`,
unmodified except for `--seed` and `--output-dir results/runs_g0`.
250 rounds x 2000 steps; N=6; d=25; M=7; ring topology; `attack: none`;
`defense: mean`, f=0. Oracle evaluation every round.

## Evaluation quantities (and the ones that must NOT be substituted)

Gate arithmetic uses ONLY these. All the oracle quantities are proper
discounted Monte-Carlo sums over the oracle's own withheld rollout:

| Quantity | Field | Log |
| --- | --- | --- |
| Task return | `episodic_task_return` | `oracle.jsonl` |
| True cost return | `true_cost_return` | `oracle.jsonl` |
| Reported cost return | `episodic_reported_cost_return` | `oracle.jsonl` |
| Mechanism cost estimate | `mechanism_reported_cost_return` | `rounds.jsonl` (the value that actually drove the dual update) |

`rounds.jsonl`'s `task_return` (`ret_r[0]`) and `reported_cost_return`
(`ret_c[0]`) are GAE(lambda) bootstrap TARGETS on the pre-update policy.
They are diagnostics. They are **not** admissible in any gate below, and
must not be mixed into any evaluation figure.

---

## Gate L -- Learning

Window: first 20 rounds (k = 0..19) against last 20 rounds
(k = 230..249) of `episodic_task_return`.

* **L1 (direction).** `mean(last 20) - mean(first 20) > 0` for **all
  three** seeds.
* **L2 (magnitude).** Averaged across seeds, that improvement is at
  least **1.0x the pooled standard deviation of the first-20 window**.

Rationale for the 1.0-sd bar: it is the smallest effect that cannot be
explained by the round-to-round noise the run itself exhibits, and it is
scale-free, so it does not require knowing MaMuJoCo's return scale in
advance. A numerically higher final return that does not clear its own
noise floor is not "learning" and does not pass.

**Gate L passes iff L1 and L2 both hold.**

## Gate C -- Constraint

Uses `true_cost_return`, averaged over the 6 agents.

* **C1 (the constraint is exercised, whole-run).** Over the whole run,
  true cost return must exceed the budget d=25 in at least one round for
  at least one agent, in all three seeds. A run in which the constraint
  never binds tests nothing (PROJECT_REPORT.md section R6.1) and fails
  regardless of how good the returns look.
* **C2 (trend).** `mean(last 20% of rounds) < mean(first 20% of rounds)`
  -- true cost moves TOWARD the budget -- with the **same sign in all
  three seeds**.

Deliberately NOT required: that true cost end below d=25. At d=25 with
`velocity_threshold=0.75` the initial-policy measurement is ~1.66x
budget, and 5e5 steps is 1/20 of the paper's scale; demanding
feasibility within 250 rounds would fail the gate for a reason unrelated
to whether the learner works. Whether d is actually met is **reported as
a headline number** and feeds the next-experiment decision, but it is
not a G0 gate. This exemption is declared in advance precisely so it
cannot be granted later.

**Gate C passes iff C1 and C2 both hold.**

## Gate K -- Cost-critic calibration

Bias is defined per agent per round as `estimate - true_cost_return`,
with the oracle's `true_cost_return` as truth.

Window: last 20% of rounds (k = 200..249).

* **K1 (own critic).** `|mean own-critic bias|` <= **0.20 x d = 5.0**,
  in all three seeds.
* **K2 (mechanism estimate).** `|mean mechanism bias|` <= **0.20 x d =
  5.0**, in all three seeds. This is the causally important one: the
  mechanism estimate is what the dual update compares against d.
* **K3 (improvement).** `|mean bias|` in the last 20% window is strictly
  smaller than in the first 20% window, for both quantities, in all
  three seeds.

Rationale for 0.20 x d: the pre-repair clean run measured own-critic
bias of -10.8 to -15.1 (0.43x to 0.60x of d) sustained for 250 rounds,
which is the failure this repair targeted. 0.20 x d sits well inside
that failure and well outside the post-repair synthetic smoke run's
converged |bias| < 0.1. It is a bar the repair should clear if it worked
and should miss if it did not.

**Gate K passes iff K1, K2 and K3 all hold.**

## Gate D -- Dual behaviour

Uses `lambda_after` from `rounds.jsonl`.

* **D1 (activation).** lambda > 0 in at least **25% of (round, agent)
  cells** over the whole run, in all three seeds.
* **D2 (sustained).** `mean lambda over the last 20% of rounds > 0` in
  all three seeds.

Rationale for 25%: the pre-repair clean run at `velocity_threshold=1.0`
measured 14% of cells with lambda > 0 and was documented as leaving
"~86% of the run structurally unable to distinguish condition A from
B/C/D". 25% is the minimum that meaningfully improves on the documented
failure. Note that D is an *activation* gate, not a *stability* gate:
lambda saturating at `lambda_max` would satisfy D1/D2 but is reported
separately and counts against the verdict qualitatively.

**Gate D passes iff D1 and D2 both hold.**

## Gate R -- Reproducibility

* **R1.** All three runs reach round 249 with
  `run_metadata.json: status == "complete"`, `rounds.jsonl` and
  `oracle.jsonl` each holding exactly 250 sequential records, and a
  checkpoint present.
* **R2.** `scripts/analyze_matrix.py --runs-dir results/runs_g0`
  reports no integrity problems.
* **R3.** The Gate-C2 true-cost trend direction is the same across all
  three seeds. (Stated in Gate C as well; repeated here because
  cross-seed consistency is itself the reproducibility claim.)

**Gate R passes iff R1, R2 and R3 all hold.**

---

## Explicitly NOT a gate

**Source independence / effective M.**
`scripts/audit_source_independence.py` is run for every G0 seed and its
participation-ratio "statistical effective M" is reported in full. It
does **not** gate G0. `docs/assumptions.md` already records that
measured effective M is ~1.10 against a nominal 7, both before and after
the P0 repair, and that this is a property of the source ensemble's and
the environment's actual diversity rather than a bug a further patch
fixes. It is a threat to the empirical validity of Theorem 2 / RCE,
which are not what G0 tests. It is reported, not laundered, and it is
not permitted to mask a learner that works.

## Verdict rule

* **PASS** -- L, C, K, D, R all pass.
* **CONDITIONAL PASS** -- exactly one gate fails, the failure has an
  identified cause, and that cause does not invalidate the other four.
* **FAIL** -- two or more gates fail, or any single failure whose cause
  implicates the learner itself.

On FAIL: stop. Do not run the attack study, do not relax the threshold,
do not discard a seed, do not raise the budget, do not redefine the
return, do not change the metric or the statistical criterion. Identify
the failed gate, find the root cause, make the minimum justified repair,
rerun the relevant validation, and report what changed.
