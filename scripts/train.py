#!/usr/bin/env python
"""Run one experiment config end to end.

Usage:
    python scripts/train.py --config configs/experiment/smoke.yaml
    python scripts/train.py --config configs/experiment/local_demo_attack.yaml --eval-every 5

The Stage-2 pilot specs (configs/experiment/pilot_*.yaml) need a Safe
MAMuJoCo backend installed -- `pip install "safelie[mujoco]"`. Run
`scripts/calibrate_cost.py` against the config first: a non-binding cost
constraint produces a run that looks like a clean null result but tests
nothing (PROJECT_REPORT.md §R6.1).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safelie.experiment import run_experiment_with_oracle  # noqa: E402
from safelie.utils.config import load_experiment_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config")
    parser.add_argument("--eval-every", type=int, default=1, help="Run the oracle evaluation every N rounds")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the config's seed. The run_id is suffixed '_seed<N>' so each "
        "seed writes to its own directory -- without this, a second seed would "
        "resume the first one's checkpoint instead of starting a fresh run.",
    )
    parser.add_argument("--run-id", default=None, help="Override the run_id (and output directory name)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the config's output_dir. Use this to keep a validation stage's "
        "runs in a directory that CANNOT be confused with an earlier campaign's -- "
        "e.g. --output-dir results/runs_g0 for the G0 clean-baseline validation, whose "
        "runs must never be aggregated together with the pre-P0-repair matrix still on "
        "disk in results/runs (docs/assumptions.md, 'P0 implementation repair').",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="torch intra-op threads (default 1). Verified bitwise-identical to the "
        "torch default on this pipeline -- the networks are too small for intra-op "
        "parallelism to pay -- so pinning it keeps results independent of the host's "
        "core count and lets several runs share a machine without contending.",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    cfg = load_experiment_config(args.config)
    updates: dict = {}
    if args.seed is not None:
        updates["seed"] = args.seed
        updates["run_id"] = f"{cfg.run_id}_seed{args.seed}"
    if args.run_id is not None:
        updates["run_id"] = args.run_id
    if args.output_dir is not None:
        updates["output_dir"] = args.output_dir
    if updates:
        cfg = cfg.model_copy(update=updates)
    print(f"Output directory: {Path(cfg.output_dir) / cfg.run_id}")
    print(f"Loaded and validated config: run_id={cfg.run_id}, seed={cfg.seed}, env={cfg.env.name}, "
          f"attack={cfg.attack.name}, defense={cfg.defense.name}, "
          f"sources M={cfg.sources.M} (effective_M={cfg.sources.effective_M})")

    if cfg.env.name != "synthetic_constrained_marl":
        from safelie.envs.mamujoco import available_backend

        print(
            f"NOTE: env.name='{cfg.env.name}' runs on the Safe MAMuJoCo adapter "
            f"(backend={available_backend()}). Read safelie/envs/mamujoco.py's "
            f"docstring before interpreting these numbers, and confirm the cost "
            f"constraint binds with scripts/calibrate_cost.py."
        )

    t0 = time.time()
    out_dir = run_experiment_with_oracle(cfg, eval_every=args.eval_every)
    elapsed = time.time() - t0
    print(f"Run complete in {elapsed:.1f}s. Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
