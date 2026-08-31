# Troubleshooting

## `NotImplementedError` when running a `pilot_*.yaml` config

Expected. Those configs use `env.name: manyagent_ant`, which requires the
Safe MAMuJoCo adapter that is intentionally not implemented in this
repository build. See [reproducibility.md](reproducibility.md) and
`safelie/envs/mamujoco.py`'s docstring.

## `ValueError: Defense 'rce' requires effective_M > 2f`

Not a bug — this is `safelie.utils.config.ExperimentConfig`'s
config-validation rule working as intended (smoke test S5, decision D3).
It means your `sources` list's *distinct independence classes* (not raw
report count) don't exceed `2 * defense.f`. Either add genuinely
independent sources (different `independence_class` values) or lower
`f`. Do not "fix" this by giving unrelated sources the same class or by
inflating the report count — that recreates exactly weakness W4 the
config validator exists to prevent.

## The attack doesn't seem to change anything

Check, in order:

1. **Is `defense.f` (or `attack.f`) actually corrupting a source that
   affects the aggregate?** `select_corrupted_sources` picks the first
   `f` non-`own_critic` sources in config order — confirm the source you
   expect is in `self.corrupted_ids` (add a print, or check
   `rounds.jsonl`'s `corrupted_source_ids` field).
2. **Has the constraint begun to bind?** This is the most common cause,
   and it is not a bug — it's `PROJECT_REPORT.md` §R6.1's precondition.
   If the aggregate cost estimate is far below the budget, `lambda`
   stays clipped at 0 regardless of the attack (the projection eats the
   attack's effect entirely). Check `rounds.jsonl`'s `lambda_after`
   field across rounds — if it never moves off 0.0, either lower the
   budget, raise `total_steps`, or both. The `local_demo_*.yaml` configs
   use `budget=5.0` instead of the paper's `d=25` for exactly this
   reason — see [assumptions.md](assumptions.md).
3. **Is the aggregator diluting the attack as Theorem 1 predicts?** With
   `mean` aggregation over M sources, one corrupted source's effect on
   the aggregate is `delta / M` — small by design when M is large. This
   is not a bug; it is the paper's own mass-conservation result.

## `pytest.warns(RuntimeWarning)` test fails / RCE seems to silently drop the margin

If you've changed `sigma_min` or `min_retained` in a config, check
`RceResult.degenerate` in the round log — a degenerate retained set
(`|T| < min_retained`) floors the spread at `sigma_min` and should log a
`RuntimeWarning`. If you see a `spread` of exactly `0.0` **without** a
warning, that is the single most dangerous bug this repository could
contain (per `PROJECT_REPORT.md` §13.2) — please file an issue
immediately; it should be structurally impossible given
`safelie.defenses.rce.rce_aggregate`'s implementation, but if you've
modified it, re-run `tests/property/test_aggregators.py`.

## Oracle isolation test failures

`tests/isolation/test_oracle_isolation.py` failing means either (a) you
added a call to `env._oracle_handle_privileged()` or an import of
`safelie.eval.oracle` from `safelie.training` or `safelie.algos`, or (b)
`DualCostStep` gained a `true_cost`-shaped field. Both are correctness
regressions, not lint nitpicks — see
[architecture.md](architecture.md)'s isolation-boundary section. Revert
the change; if you have a legitimate reason the learner needs a new
metric, add it to `safelie.eval` and have the orchestrator
(`safelie.experiment`) merge it in, the way `detection_gap` is handled.

## `mypy` / `ruff` failures after editing

Both are clean on this repository as shipped
(`ruff check src/ scripts/ tests/`, `mypy src/safelie`). If your change
introduces new warnings, fix them rather than suppressing — the two
places this repository does add `# type: ignore`-style workarounds
(there are none as shipped) would need a comment explaining why.

## Windows-specific: `python -c "multi\nline"` fails with `IndentationError`

This is a Git-Bash / PowerShell quoting issue with inline multi-line
`python -c` snippets, not a bug in this codebase. Write the snippet to a
`.py` file and run `python file.py` instead.

## Slow test runs

The full suite (`pytest tests/`) takes roughly 8-16 seconds on a modern
laptop. If it's taking much longer, check whether `tests/integration`
(which actually trains a tiny policy) is the bottleneck — run
`pytest tests/unit tests/property tests/theory tests/isolation -q` (the
CRITICAL subset `safelie.preflight` also uses) to isolate it.
