# Setup

## Requirements

- Python >= 3.10 (developed and tested on 3.11.9)
- **No GPU required, including for the Stage-2 pilot.** Nothing in
  `safelie` moves a tensor to CUDA: the networks are small MLPs stepped
  one observation at a time inside a Python loop, so the workload is
  CPU-bound and a GPU runtime does not accelerate it. Prefer a high-CPU
  machine. Measured throughput on the real environment is ~130
  env-steps/s, which puts a 5e5-step pilot run at roughly two hours.
- MuJoCo is an **optional** dependency, needed only for the `pilot_*`
  configs. See "Real environments" below.
- No external services, API keys, or secrets of any kind. There is no
  `.env` file for this repository — every configuration is a YAML file
  under `configs/`.

## From a clean environment

```bash
git clone <this repository>
cd safelie   # or wherever you cloned it — the "Github/" root of the deliverable
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

This installs the package in editable mode plus the dev toolchain
(pytest, ruff, mypy). For an exact, pinned install instead of a resolved
range, use `pip install -r requirements.txt` followed by
`pip install -e . --no-deps`.

## Verify the install

```bash
python -c "import safelie; print(safelie.__version__)"
pytest tests/ -v
```

Expect 116 tests to pass in roughly 7-16 seconds on a modern laptop CPU
(see [SMOKE_TEST_REPORT.md](../SMOKE_TEST_REPORT.md) for the actual
numbers from this build). If anything fails here, do not proceed to
training — see [troubleshooting.md](troubleshooting.md).

## Code quality tools (optional but recommended before contributing)

```bash
ruff check src/ scripts/ tests/
mypy src/safelie
```

Both are clean on this repository as shipped.

## Docker

```bash
docker build -t safelie .
docker run --rm safelie
```

The image installs the package and runs `scripts/smoke_test.py` as its
entrypoint. It does **not** install MuJoCo — see
[reproducibility.md](reproducibility.md) for why that stage is
intentionally out of scope for this image.

## Real environments (the `pilot_*` configs)

Two backends implement Safe MAMuJoCo. They **cannot coexist**:
`safety-gymnasium` pins `gymnasium==0.28.1`, `gymnasium-robotics==1.2.2`
and `mujoco==2.3.3`. Install one per environment.

```bash
pip install "safelie[mujoco]"        # portable: mujoco + gymnasium-robotics
pip install safety-gymnasium==1.0.0  # the reference implementation, Linux only
```

`safety-gymnasium` does not install on Windows (its pinned `pygame` has no
wheel and builds from source via msys2). The portable backend installs
everywhere, including Windows and Colab.

Which one you install changes what the numbers mean, not just whether
they compute — the reference backend brings Safe MAMuJoCo's own cost
function, the portable one has none and this repository supplies it. Read
`safelie/envs/mamujoco.py`'s docstring before running a pilot, and run

```bash
python scripts/calibrate_cost.py --config configs/experiment/pilot_A_clean.yaml
```

to confirm the cost constraint actually binds before spending hours on a
run that cannot test anything (`PROJECT_REPORT.md` §R6.1).

## What you do NOT need

- MuJoCo — not required for the test suite, the local demos, or the
  smoke run, all of which use the synthetic environment. Needed only for
  the `pilot_*` configs; see "Real environments" below.
- `mujoco-py` or `dm_control` — superseded by the `mujoco` package; not
  used here.
- Weights & Biases, MLflow, or any experiment tracker account — logging
  is plain JSONL to a local directory (`safelie.utils.logging`).
- A GitHub token, cloud credentials, or any other secret.
