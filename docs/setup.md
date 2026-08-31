# Setup

## Requirements

- Python >= 3.10 (developed and tested on 3.11.9)
- No GPU required for anything in this repository as it stands (the
  synthetic environment and every test run on CPU). A GPU is required
  only for the Stage-2 Colab pilot once the Safe MAMuJoCo adapter
  (`safelie/envs/mamujoco.py`) is completed.
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

## What you do NOT need

- MuJoCo, `mujoco-py`, `dm_control`, or `safety-gymnasium` — not
  installed, not required for anything this repository can currently run.
  Needed only to complete the Stage-2 adapter (`safelie/envs/mamujoco.py`).
- Weights & Biases, MLflow, or any experiment tracker account — logging
  is plain JSONL to a local directory (`safelie.utils.logging`).
- A GitHub token, cloud credentials, or any other secret.
