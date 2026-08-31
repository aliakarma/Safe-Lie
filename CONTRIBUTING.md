# Contributing to SafeLie

## Before you start

Read [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) and
[docs/paper_implementation_mapping.md](docs/paper_implementation_mapping.md)
first. This repository implements a research threat model whose own
source paper has not been empirically validated (see the paper's own
`[PROJECTED]` markings). Contributions that quietly treat projected
numbers as measured results, or that weaken the isolation boundary
between the learner and the oracle evaluator (`safelie.eval.oracle`), will
not be accepted.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest tests/ -v
```

See [docs/setup.md](docs/setup.md) for a from-clean-environment walkthrough.

## Before opening a PR

1. `pytest tests/` — every test must pass. If you touch the aggregator
   zoo (`safelie.defenses`) or the attack module (`safelie.attacks`), add
   property tests, not just example-based ones — those two modules are
   where the report identifies the highest-consequence silent-bug risk
   (a degenerate margin turning RCE into plain trimmed mean while the
   logs still say "RCE").
2. `python scripts/smoke_test.py` — the GREEN SIGNAL gate must report
   GREEN.
3. If you add or change a config-validation rule, add a test in
   `tests/unit/test_config.py` that exercises it.
4. `ruff check src/ scripts/ tests/` and `mypy src/safelie` should be clean
   (or the new warnings explained in the PR description).

## Code style

- Type hints on public functions; docstrings that cite the specific paper
  section or report section a piece of code implements, not generic
  descriptions of what the code obviously does.
- No comments that restate the code. A comment should explain a
  non-obvious constraint, an assumption, or a citation to
  `main_iclr.tex` / `PROJECT_REPORT.md`.
- Prefer raising over silently clamping or falling back, for anything
  touching the defense's degenerate cases (`M <= 2f`, `|T| < 3`) — this
  mirrors decision D3 in `docs/assumptions.md`.

## Extending to a real environment (Safe MAMuJoCo / Safety-Gymnasium)

If you're picking up the Stage-2 Colab work, start at
`src/safelie/envs/mamujoco.py` — its docstring lists exactly what is
needed and why it was left unimplemented here. Run the full local smoke
suite against your new adapter before spending any GPU time
(`PROJECT_REPORT.md §R5`, the GREEN SIGNAL gate).

## Reporting issues

Open a GitHub issue. For anything touching the oracle isolation boundary
or a potential fabrication of results in downstream analysis, please
flag it as high-priority in the issue title.
