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
