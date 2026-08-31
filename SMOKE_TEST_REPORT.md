# Smoke Test Report

Every result below was actually executed during this repository's build.
Nothing here is inferred or assumed. Commands are copy-pasteable.

## Environment

| | |
|---|---|
| OS | Windows 11 Pro (10.0.26200) |
| Python | 3.11.9 |
| Install | `pip install -e .` (editable), confirmed importable as the `safelie` package (not via `PYTHONPATH` hacks) |
| Key dependency versions | numpy 2.2.6, scipy 1.13.0, torch 2.12.1+cpu, pydantic 2.13.4, networkx 3.6.1, pandas 2.3.3, statsmodels 0.14.1, pytest 9.0.3 |
| GPU | None used or required — everything below ran on CPU |

## Test suite

| Suite | Command | Result |
|---|---|---|
| Full suite | `pytest tests/ -v` | **116 passed, 0 failed, 0 skipped, 7.13s** |
| Unit | `pytest tests/unit -v` | **59 passed** — config validation (S2/S5), consensus (S4), sources (S12), auditor, stats, shipped-config loading, environment-factory dispatch |
| Property | `pytest tests/property -v` | **21 passed** — aggregators (S6-S8, S21-S25), attacks (S16-S20), dual update (S9, S23) |
| Theory | `pytest tests/theory -v` | **26 passed** — mass conservation, spreading, closed-loop diagnostic |
| Smoke | `pytest tests/smoke -v` | **4 passed** — end-to-end run (S15), determinism (S3), checkpoint-restore (S14) |
| Isolation | `pytest tests/isolation -v` | **5 passed** — oracle isolation (S10), fails closed |
| Integration | `pytest tests/integration -v` | **1 passed** — analysis table regeneration from a completed run |

`tests/unit` includes `test_configs_load.py` (10 tests, all shipped
configs validate) and `test_env_factory.py` (5 tests, added after the
environment-dispatch bug described in "Fixes applied" below — a
regression test for exactly that bug).

## GREEN SIGNAL gate

```
python scripts/smoke_test.py
```

**Result: GREEN** (115 tests across `unit`, `property`, `theory`, `smoke`,
`isolation` — deliberately excludes `tests/integration`, which is not
part of the CRITICAL gate — passed in 6.96s). All 7 automated conditions
(G-1 through G-7) report PASS. G-8 (no unresolved specification gap) is
a manual review item by design — `docs/decisions/README.md` and
`docs/assumptions.md` were both current as of this build.

## Code quality

| Tool | Command | Result |
|---|---|---|
| Lint | `ruff check src/ scripts/ tests/` | **Clean, 0 issues** (37 issues found and fixed during the build — 30 auto-fixed, 7 fixed by hand, including one closure-over-loop-variable pattern in `safelie.training.loop` that was safe but worth cleaning up) |
| Type check | `mypy src/safelie --ignore-missing-imports` | **Clean, 0 issues, 55 source files checked** (13 issues found and fixed during the build, including strengthening `DualCostEnvWrapper`/`OracleCapableEnv` so the isolation boundary is visible in the type signatures, not only the docstrings) |
| Notebook lint | `python scripts/lint_notebooks.py` | **Passed** — no `def`/`class` in any notebook cell (decision D13) |

## Notebooks

All three notebooks were executed end to end (every code cell run in
sequence, in order, with the actual output captured):

| Notebook | Result |
|---|---|
| `notebooks/theory_validation.ipynb` | Ran successfully. Theorem 1 (Test A): exact agreement across `complete`, `ring`, `star`, `erdos_renyi`, `identity`, `shared_constraint` — max deviation from closed form 2.5e-13, cross-topology spread 4.0e-13. Test B (spreading): ring residual norm 0.130 within bound 3.0; identity/ring stealth ratio measured at ~308×. Test C (closed-loop diagnostic): `rho = 0.0010` at heterogeneous feedback gains (mean 0.025, std 0.016) across `complete`/`ring`/`star` — invariance appears robust at this scale, reported as a measurement per decision D11. |
| `notebooks/local_smoke_test.ipynb` | Ran successfully. Tiny end-to-end run produced `rounds.jsonl` and `oracle.jsonl`; the full GREEN SIGNAL gate subprocess call exited 0. |
| `notebooks/colab_full_experiment.ipynb` | Ran successfully as far as it honestly can without a Colab GPU or the MuJoCo adapter: runtime verification correctly reports no GPU present; dependency/version cells report actual installed versions; preflight (cell 5) passes; the throughput probe and training cells (6, 8) correctly report that `safelie.envs.mamujoco` is not implemented, rather than silently doing nothing. |

## Local demo runs (synthetic environment — NOT Safe MAMuJoCo, NOT the paper's numbers)

```
python scripts/train.py --config configs/experiment/local_demo_clean.yaml
python scripts/train.py --config configs/experiment/local_demo_attack.yaml
python scripts/train.py --config configs/experiment/local_demo_rce.yaml
python scripts/train.py --config configs/experiment/local_demo_benign.yaml
```

All four completed successfully, ~45 seconds each (1 seed, 75 rounds, 6
agents), producing complete `rounds.jsonl` + `oracle.jsonl` artifact
pairs.

**A genuine bug was found and fixed during this process.** The first
attempt (at the paper's literal `d=25` budget) produced *bitwise-identical*
logs across all four conditions. Investigation (a traced run showing the
attack correctly injecting `-12.5` into the targeted source's residual,
but the aggregate estimate remaining so far below the budget that the
dual update's `[0, lambda_max]` projection clipped the result to `0.0`
in every condition) confirmed this was not an attack-wiring bug — it was
exactly the "constraint not binding" confound `PROJECT_REPORT.md` §R6.1
warns about, caused by the budget being calibrated to Safe MAMuJoCo's
cost scale, not this synthetic environment's. The budget was recalibrated
to `d=5` (found by direct measurement — see `docs/assumptions.md`) so the
constraint begins to bind by round ~30 of a 75-round run, and the runs
were repeated.

**Measured summary** (`scripts/evaluate.py`, mean of the last 10 rounds,
budget d=5.0):

| Run | True cost (6 agents) |
|---|---|
| Clean | 21.39, 20.63, 22.07, 21.78, 24.07, 21.39 |
| Attacked (f=1, undefended) | 21.43, 20.66, 22.11, 21.83, 24.09, 21.42 |
| Attacked, RCE-defended | 21.30, 20.57, 22.02, 21.70, 24.09, 21.33 |
| Benign control | 21.39, 20.63, 22.08, 21.78, 24.08, 21.39 |

**Observed pattern**: attacked true cost exceeded clean for all 6 agents
(6/6); RCE's true cost was below the attacked baseline's for 5 of 6
agents (tied on the 6th); the benign control's true cost matched the
clean baseline far more closely than the attack's did, in all 6 agents.
This is directionally consistent with the paper's mechanism. **The
magnitude is small** (differences of ~0.05-0.15 on a base of ~20-24), an
expected consequence of `mean` aggregation dividing a single corrupted
source's effect by M=7 (Theorem 1's own mass-conservation logic), of the
toy environment's scale, and of only 75 rounds / 1 seed. **This is not,
and is not presented as, evidence for or against the paper's hypothesis**
— see `docs/reproducibility.md`.

## Governance CLI

```
python scripts/audit_sources.py --preset m5_two_agent --assumed-f 2
```
→ `RESULT = FAIL` (exit code 1) — correctly rejects the paper's own N=2,
M=5 configuration, because its 3 ensemble replicas share one
independence class (`effective_M=3 < 2*2+1=5`).

```
python scripts/audit_sources.py --preset m7_primary --assumed-f 1
```
→ `RESULT = PASS` (exit code 0) — the primary M=7 configuration, all 7
sources in distinct independence classes, `effective_M=7 >= 2*1+1=3`.

This matches `PROJECT_REPORT.md` Phase 10's own stated exit criterion
exactly.

## What was NOT run, and why

| Not run | Why |
|---|---|
| Any `configs/experiment/pilot_*.yaml` (Stage-2 compact pilot) | Requires `safelie.envs.mamujoco`, not implemented — raises `NotImplementedError` immediately on attempted execution. See fix #5 below: this correct behavior required a real bug fix, not just documentation. |
| Any Stage-3 experiment (MACPO, Dec-PDO, PID-Lagrangian, additional aggregators, topology sweep, reward-poisoning control, etc.) | Not implemented; explicitly out of scope for the compact study per the report's own §R2.1 |
| Welch's t-test / Holm correction on any real result | Requires ≥5 seeds; only 1 seed was run for the local demos, and `safelie.analysis.stats.welch_t_test` correctly raises `ValueError` if called with fewer |

## Fixes applied during this build (honesty log)

1. **Checkpoint/restore missing state** (`safelie.sources.estimators.DiversifiedReplica`
   network and optimizer weights, plus global torch/numpy RNG state)
   caused a restored run to diverge from an uninterrupted one starting
   exactly at the round after restore. Fixed by capturing all of it in
   `ExperimentRun.checkpoint`/`restore`; `tests/smoke/test_determinism.py::test_checkpoint_restore_continues_bitwise_identically`
   now passes and would have caught the original bug.
2. **Isolation test over-strict on docstrings**: an early version of the
   grep-level oracle-isolation test used plain string search and failed
   on `safelie.training.loop`'s own docstring, which *discusses* (but does
   not call) the privileged oracle path. Fixed by switching to an
   AST-based check for actual attribute access, not text search.
3. **The budget-scale bug described above** (local demo configs, `d=25`
   → `d=5`), found by noticing four conditions produced bitwise-identical
   logs and tracing it to the dual update's projection eating the
   attack's effect.
4. **13 mypy findings and 36 ruff findings**, listed and fixed as part of
   the code-quality pass documented above — none were logic bugs; all
   were missing type annotations, unused loop variables, or import-style
   issues, except the closure-over-loop-variable pattern (verified safe
   in context, since `SourceRegistry.collect` invokes its callback
   synchronously, but rewritten with `functools.partial` for clarity and
   to remove the warning honestly rather than suppress it).
5. **`ExperimentRun.__init__` silently ignored `cfg.env.name`**: it
   unconditionally constructed `SyntheticConstrainedMarlEnv` regardless
   of the config, so a `manyagent_ant` pilot config would not fail —
   it would silently train on the *wrong environment* for however long
   `total_steps` specified (500,000 steps in the pilot configs; this was
   caught because the command exceeded a 120s interactive timeout and
   was moved to a background process, not because it errored). Found by
   actually attempting `python scripts/train.py --config
   configs/experiment/pilot_A_clean.yaml` rather than assuming the
   `NotImplementedError` documented elsewhere would fire. Fixed by adding
   a single dispatch point, `safelie.envs.factory.build_env`, used by both
   `ExperimentRun.__init__` and the orchestrator's evaluation-env
   factory, and used consistently by `scripts/infer.py`. Verified after
   the fix: the same command now raises `NotImplementedError` immediately
   (traceback captured, `safelie.envs.factory.build_env` at the top of the
   stack) instead of running. Regression test:
   `tests/unit/test_env_factory.py`.
