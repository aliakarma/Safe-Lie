#!/usr/bin/env python
"""Load a checkpoint and roll out the trained policy for inspection.

Usage:
    python scripts/infer.py --config configs/experiment/smoke.yaml \\
                             --checkpoint results/runs/smoke_clean/checkpoint.pt \\
                             --episodes 5

Reports task return and reported cost per episode from the learner's own
policies, and separately (via the withheld evaluator) the true cost --
labelled explicitly, since this is the one script where both numbers are
legitimately computed side by side for a human to read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safelie.envs.factory import build_env  # noqa: E402
from safelie.eval.harness import evaluate_true_cost  # noqa: E402
from safelie.training.loop import ExperimentRun  # noqa: E402
from safelie.utils.config import load_experiment_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)

    # ExperimentRun(cfg) raises NotImplementedError immediately for a
    # non-synthetic env.name (via safelie.envs.factory.build_env), before
    # any checkpoint is touched.
    run = ExperimentRun(cfg)
    run.restore(Path(args.checkpoint))

    def factory():
        return build_env(cfg.env, rollout_length=cfg.rollout_length)

    print(f"Loaded checkpoint {args.checkpoint}; running {args.episodes} evaluation episode(s).")
    for ep in range(args.episodes):
        result = evaluate_true_cost(
            env_factory=factory, agents=run.agents, gamma=cfg.ppo.gamma,
            budget=cfg.env.budget, rollout_length=cfg.rollout_length, seed=args.seed + ep,
        )
        for aid in run.env.agent_ids:
            print(
                f"  episode {ep} {aid}: true_cost_return={result.true_cost_return[aid]:.2f} "
                f"peak_true_cost={result.peak_true_cost[aid]:.2f} "
                f"violated={result.violated[aid]}"
            )


if __name__ == "__main__":
    main()
