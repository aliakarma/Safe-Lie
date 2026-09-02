# Evaluation

## Metrics (`safelie.eval.metrics`)

| Metric | Definition | Report reference |
|---|---|---|
| Task return | Discounted sum of shared reward | `J_R`, Eq. 1 |
| Reported cost | The learner's own GAE-corrected cost-return estimate | what a normal evaluation pipeline sees |
| True cost | Discounted sum of true per-step cost, from the withheld oracle | `J_true_C`, §5.1 |
| Violation rate | Fraction of evaluated episodes with `J_true_C > d` | `[GAP]` G9, resolved as the fraction over evaluation episodes |
| Peak violation | Max per-step true cost observed | `[GAP]` G18, declared in the paper and never reported there — implemented here |
| Detection gap | `J_true_C - J_reported_C` | §5.1; named `detection_gap` in code, never `Delta`, to avoid the paper's own notation collision (W12) between the detection gap and the cumulative corruption budget |

## The withheld oracle

`safelie.eval.oracle.OracleEvaluator` accumulates true cost over an
episode using a privileged handle only it can obtain
(`env._oracle_handle_privileged()`). `safelie.eval.harness.evaluate_true_cost`
wraps this into a full episode: a **fresh environment instance**, a fresh
seed, and the **current** (frozen, no-gradient) policy weights. This is
run by the orchestrator (`safelie.experiment`), never by the learner — see
[architecture.md](architecture.md).

## The pre-registered success criterion

`main_iclr.tex` §5.1 states the attack succeeds iff all three hold: (1)
return within one standard deviation of the no-attack baseline, (2)
reported cost ≤ d, (3) true cost > d. `safelie.eval.protocol.attack_succeeded`
implements this as a direct assertion:

```python
def attack_succeeded(res, baseline, d):
    return (abs(res.return_mean - baseline.return_mean) <= max(baseline.return_std, 1e-9)
            and res.reported_cost_mean <= d
            and res.true_cost_mean > d)
```

This requires ≥2 seeds (a standard deviation of one seed is undefined)
and is best used at ≥5 seeds. **The compact pilot's 3-seed matrix should
not use this function** — see the next section.

## The compact pilot's 3-seed reporting (decision D6)

`PROJECT_REPORT.md` §R2.3: at 3 seeds, running Welch's t-test or reporting
mean ± SD manufactures false precision. `safelie.eval.protocol.pilot_seed_summary`
and `consistent_across_seeds` instead report **per-seed sign and
ordering** — did every seed show true cost rising under attack, in the
same direction? `safelie.analysis.stats.welch_t_test` and
`holm_correction` enforce this at the code level: they raise
`ValueError` if called with fewer than 5 samples per arm
(`MIN_SEEDS_FOR_INFERENCE`), rather than silently producing an
underpowered p-value.

## Regenerating summary tables

```bash
python scripts/evaluate.py \
  --run clean=results/runs/local_demo_clean \
  --run attack=results/runs/local_demo_attack \
  --run rce=results/runs/local_demo_rce \
  --budget 5.0
```

`safelie.analysis.tables.build_summary_table` reads only what is in the
JSONL logs a run actually produced — it never reads or compares against
`main_iclr.tex`'s projected Table 3/4 values. Any resemblance in *format*
between this output and the paper's tables is not a claim about
*content*.

## The margin: three quantities, never conflated (decision D9)

`safelie.eval.margin` keeps these separate, per `PROJECT_REPORT.md` §R10.4:

| Quantity | What it is | Where |
|---|---|---|
| `applied_margin` (`beta * sigma`) | Empirical conservatism, measured every round | `safelie.defenses.rce.RceResult.applied_margin` |
| `epsilon_offline` | An offline-calibrated *reference*, from a clean run's observed disagreement — not the true (uncomputable) theoretical bound `epsilon(M,f,alpha)` | `calibrate_epsilon_offline` |
| `guarantee_in_force` | Whether the empirical margin meets or exceeds the offline reference, logged per round | `compute_guarantee_in_force` |

**No runtime safety guarantee is claimed anywhere in this codebase.**
`guarantee_in_force` is the best available runtime *estimate* of Theorem
2's precondition, not a proof that it holds — the paper's own analysis
(§13.3, weakness W3) explains why the true precondition cannot be checked
without ground truth the deployment lacks by construction.

## Averaging window: use the whole run, not a trailing slice

**This is not a stylistic preference. A trailing window nearly produced a
wrong scientific conclusion in this repository, and the mistake is easy to
repeat.**

The dual variable oscillates. `lambda` rises until the policy complies,
the true cost falls below the budget, `lambda` relaxes, the policy drifts
back, and the cycle repeats — the documented behaviour of a primal-dual
Lagrangian method without a PID controller. Measured here the period is
roughly 100 rounds, so a 250-round pilot contains only about two cycles.

A trailing-100-round average therefore does not measure a steady state. It
samples a near-random *phase* of that oscillation, and which phase a run
ends on varies by seed. Concretely, summarizing the same completed runs
over different windows gave:

| `lambda`, condition vs clean | whole run | last 200 | last 150 | last 100 |
|---|---|---|---|---|
| attacked (B) | −0.13 | −0.17 | −0.19 | −0.36 |
| benign control (D) | **+0.12** | **+0.15** | **+0.09** | **−0.35** |

On the trailing-100 window both the attacked and the benign arm appeared
to suppress `lambda` by ~2.6 standard deviations of the clean condition's
seed spread — a striking result, and a spurious one. It reverses sign for
the benign arm on every longer window. Read from the whole run the
ordering is monotone and interpretable instead: `lambda` D > A > B, with
true cost in exactly the inverse order, which is what the corruption
mechanism predicts.

Two consequences:

1. `scripts/analyze_matrix.py` defaults to the whole run (`--window 0`)
   and always prints a **window-sensitivity table**. A contrast whose sign
   changes across windows is an artifact of the oscillation, not an
   effect, and must not be reported as one.
2. Seeds, not rounds, are the independent samples. Rounds within a run are
   strongly autocorrelated at the oscillation's timescale, so a run's ~250
   rounds are worth roughly two independent observations — which is why
   `safelie.analysis.stats.MIN_SEEDS_FOR_INFERENCE` is enforced against
   the seed count and never against the round count.

## Which "reported cost" the detection gap is measured against

`Delta = J_true_C - J_reported_C` is the paper's headline metric, and the
answer to "which reported cost?" decides whether the metric can see the
attack at all.

`safelie.training.loop` logs `reported_cost_return` as the agent's **own
cost-critic estimate** (`ret_c[0]` from GAE over its own cost stream), and
`safelie.experiment` computes the logged `detection_gap` against that. But
the attack does not touch the agent's own critic. Corruption is applied to
the *source reports* (`safelie.attacks.apply_attack`), and lands in
`aggregate.point_estimate` — which is the quantity the dual update
consumes and therefore the quantity the safety mechanism actually
believes.

Measured on the same completed runs, one attacked seed against the
3-seed clean baseline:

| detection gap measured against | attacked (B) | benign control (D) |
|---|---|---|
| the agent's own cost critic (as logged) | +0.17 sd | −1.88 sd |
| **the aggregate (what the dual consumes)** | **+3.16 sd** | −1.75 sd |

The attack's effect on the paper's headline metric is roughly 18x larger
under the second definition. The first is not meaningless — it measures
cost-critic estimation error — but it is not the corruption channel, and
reporting it as "the detection gap" understates the attack to the point of
looking like a null result.

`scripts/analyze_matrix.py` therefore reports
`detection_gap_vs_aggregate` as the primary metric, recomputed from
`aggregate.point_estimate`, which is already present in every run's
`rounds.jsonl` — so this required no re-running.

**Reconstructing the RCE estimate.** For the RCE conditions the dual
consumes `pessimistic_estimate` (`= point_estimate + applied_margin`,
Algorithm 1 line 7), and neither that nor `applied_margin` is logged
directly. They are nonetheless recoverable exactly: `applied_margin =
beta * spread`, `spread` **is** logged, and the logged value is the
post-flooring one actually used for the margin
(`safelie.defenses.rce`). So

    effective_estimate = point_estimate + beta * spread

with `beta` read from the condition's own config, and an RCE run
identified by `guarantee_in_force` being non-null (which
`safelie.training.loop` sets only for RCE).
`detection_gap_vs_aggregate` is therefore **exact for every condition**,
computed from existing logs with no re-running.

That reconstruction also makes RCE's mechanism directly visible. On one
attacked seed:

| | aggregate fed to the dual | agent's own cost critic |
|---|---|---|
| B — attacked, mean aggregation | 19.85 | 24.56 |
| C — attacked, RCE | 21.32 | 21.49 |

The attack pulls the undefended aggregate 4.7 below the agent's own
estimate; under RCE the effective estimate lands within 0.17 of it. The
defense is recovering the corrupted aggregate, not merely adding
conservatism on top of it.
