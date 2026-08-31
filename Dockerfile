# CPU-only research image for the Stage-1 pipeline (theory validation,
# smoke tests, the synthetic-environment demos). This image intentionally
# does NOT install MuJoCo / Safe MAMuJoCo -- that stack belongs on the
# Colab GPU runtime described in docs/reproducibility.md and is not part
# of what this repository can execute locally (see
# safelie/envs/mamujoco.py).
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

ENTRYPOINT ["python", "scripts/smoke_test.py"]
