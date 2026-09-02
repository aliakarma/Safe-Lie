# CPU-only research image for the dependency-free pipeline (theory
# validation, smoke tests, the synthetic-environment demos). This image
# deliberately does NOT install a Safe MAMuJoCo backend: the two available
# backends pin mutually incompatible stacks, so which one belongs here is
# a choice the user has to make rather than one this image should bake in
# (see safelie/envs/mamujoco.py's docstring).
#
# To run the pilot configs, extend this image with ONE of:
#   RUN pip install -e ".[mujoco]"          # portable; no native cost signal
#   RUN pip install safety-gymnasium==1.0.0 # the reference implementation
#
# Note that the pilot is CPU-bound -- nothing in safelie uses CUDA -- so a
# GPU base image buys nothing here.
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
