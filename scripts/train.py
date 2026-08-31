#!/usr/bin/env python
"""Run one experiment config end to end.

Usage:
    python scripts/train.py --config configs/experiment/smoke.yaml
    python scripts/train.py --config configs/experiment/local_demo_attack.yaml --eval-every 5

For `env.name: manyagent_ant` configs (the Stage-2 pilot specs under
configs/experiment/pilot_*.yaml), this will raise NotImplementedError:
those require the Safe MAMuJoCo adapter, which is not implemented in
this repository (see safelie.envs.mamujoco).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safelie.experiment import run_experiment_with_oracle  # noqa: E402
from safelie.utils.config import load_experiment_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config")
    parser.add_argument("--eval-every", type=int, default=1, help="Run the oracle evaluation every N rounds")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    print(f"Loaded and validated config: run_id={cfg.run_id}, env={cfg.env.name}, "
          f"attack={cfg.attack.name}, defense={cfg.defense.name}, "
          f"sources M={cfg.sources.M} (effective_M={cfg.sources.effective_M})")

    if cfg.env.name != "synthetic_constrained_marl":
        print(
            f"NOTE: env.name='{cfg.env.name}' requires an environment adapter "
            f"not implemented in this repository. See safelie/envs/mamujoco.py."
        )

    t0 = time.time()
    out_dir = run_experiment_with_oracle(cfg, eval_every=args.eval_every)
    elapsed = time.time() - t0
    print(f"Run complete in {elapsed:.1f}s. Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
