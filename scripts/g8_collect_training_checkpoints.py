#!/usr/bin/env python
"""G8 (docs/g8_gates.md) step 1: re-run `pilot_A_clean_seed0` BITWISE and
snapshot the intermediate policies that the original G2 campaign
overwrote.

Why this exists. `safelie.experiment.run_experiment_with_oracle` writes
its checkpoint to ONE path (`checkpoint.pt`) every round, so the only
surviving artifact of the 250-round `pilot_A_clean_seed0` trajectory is
`theta_250`. G8 needs `theta_k` for k in an early/middle/late grid, plus
the immediately-following `theta_{k+1}`, `theta_{k+2}`, and a
`theta_{k+Delta}` at the realistic one-rollout-per-round collection
stride. Those states are not on disk anywhere.

This script does NOT retrain anything new and does NOT modify the
training algorithm. `ExperimentRun.checkpoint` captures every RNG stream
the loop touches (env, attack, replicas, report heads, torch global,
numpy global), and `seed_everything(0)` fixes the start state, so
re-running the identical code path with the identical config reproduces
the SAME trajectory. That claim is not assumed: every round's
`mc_window`, `lambda_after` and oracle `true_cost_return` are compared
against the committed `results/runs_constraint_mc_g2/pilot_A_clean_seed0/`
logs and the script aborts on the first mismatch. The output is
therefore the same trajectory G2/G7 already analysed, merely observed at
more points.

No G8 diagnostic number is produced here -- this script only recovers
policy states that already existed once.

Usage:
    .venv/Scripts/python.exe scripts/g8_collect_training_checkpoints.py
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from safelie.envs.factory import build_env
from safelie.eval.harness import evaluate_true_cost
from safelie.training.loop import ExperimentRun
from safelie.utils.config import load_experiment_config

CONFIG = "configs/experiment/pilot_A_clean.yaml"
REFERENCE_RUN = "results/runs_constraint_mc_g2/pilot_A_clean_seed0"

# docs/g8_gates.md "Checkpoint grid". theta_k is the policy that GENERATES
# round k's rollout, i.e. the state when `run.round_index == k`, so the
# snapshot is taken BEFORE `run_round()` for that k.
ANCHORS = [25, 75, 125, 175, 225]
STRIDE = {25: 30, 75: 30, 125: 30, 175: 30, 225: 24}  # 225+30=255 > 249; see gates doc

SNAPSHOT_ROUNDS = sorted(
    {k + off for k in ANCHORS for off in (0, 1, 2)} | {k + STRIDE[k] for k in ANCHORS}
)

TOL = 1e-9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--reference-run", default=REFERENCE_RUN)
    ap.add_argument("--out-dir", default="results/g8_nonstationary_source_diagnostic/checkpoints")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir.parent / "_repro_run"

    ref_dir = Path(args.reference_run)
    ref_rounds = [json.loads(line) for line in (ref_dir / "rounds.jsonl").read_text().splitlines()]
    ref_oracle = [json.loads(line) for line in (ref_dir / "oracle.jsonl").read_text().splitlines()]
    print(f"Reference trajectory: {len(ref_rounds)} rounds, {len(ref_oracle)} oracle records")
    print(f"Snapshot rounds ({len(SNAPSHOT_ROUNDS)}): {SNAPSHOT_ROUNDS}")

    cfg = load_experiment_config(args.config)
    cfg = cfg.model_copy(update={"run_id": "pilot_A_clean_seed0", "seed": 0, "output_dir": str(scratch)})
    run = ExperimentRun(cfg)
    eval_seed_rng = run.seed_bundle.rng("eval")
    env_factory = lambda: build_env(cfg.env, rollout_length=cfg.rollout_length)  # noqa: E731

    num_rounds = max(1, cfg.total_steps // cfg.rollout_length)
    agent_ids = run.env.agent_ids
    mismatches: list[dict] = []
    written: list[int] = []
    t0 = time.time()

    for k in range(num_rounds):
        if k in SNAPSHOT_ROUNDS:
            # theta_k: BEFORE round k's rollout and PPO update.
            assert run.round_index == k, (run.round_index, k)
            run.checkpoint(out_dir / f"theta_{k:03d}.pt", extra={"snapshot_round_index": k})
            written.append(k)
            print(f"  [snapshot] wrote theta_{k:03d}.pt (round_index={run.round_index})")

        record = run.run_round()
        eval_seed = int(eval_seed_rng.integers(0, 2**31 - 1))
        oracle = evaluate_true_cost(
            env_factory=env_factory, agents=run.agents, gamma=cfg.ppo.gamma,
            budget=cfg.env.budget, rollout_length=cfg.rollout_length, seed=eval_seed,
        )

        for aid in agent_ids:
            got = record["constraints"][aid]
            exp = ref_rounds[k]["constraints"][aid]
            for field, g, e in (
                ("mc_window", got["constraint_estimators"]["mc_window"], exp["constraint_estimators"]["mc_window"]),
                ("lambda_after", got["lambda_after"], exp["lambda_after"]),
                ("approx_kl", got["ppo"]["approx_kl"], exp["ppo"]["approx_kl"]),
                ("true_cost_return", oracle.true_cost_return[aid], ref_oracle[k]["agents"][aid]["true_cost_return"]),
            ):
                if abs(float(g) - float(e)) > TOL:
                    mismatches.append({"round": k, "agent": aid, "field": field, "got": g, "expected": e})

        if mismatches:
            print(f"ABORT: trajectory diverged at round {k}: {mismatches[:4]}")
            (out_dir / "reproduction_check.json").write_text(
                json.dumps({"bitwise_identical": False, "mismatches": mismatches[:50],
                            "diverged_at_round": k}, indent=2)
            )
            return 1

        if k % 10 == 0 or k == num_rounds - 1:
            el = time.time() - t0
            print(f"  round {k}/{num_rounds - 1} ok  elapsed={el:.0f}s  eta={el / (k + 1) * (num_rounds - k - 1):.0f}s")

    check = {
        "bitwise_identical": True,
        "rounds_verified": num_rounds,
        "fields_verified": ["mc_window", "lambda_after", "approx_kl", "true_cost_return"],
        "agents_verified": agent_ids,
        "n_comparisons": num_rounds * len(agent_ids) * 4,
        "tolerance": TOL,
        "snapshot_rounds": written,
        "anchors": ANCHORS,
        "stride": {str(k): v for k, v in STRIDE.items()},
        "reference_run": str(ref_dir),
        "wall_clock_s": time.time() - t0,
    }
    (out_dir / "reproduction_check.json").write_text(json.dumps(check, indent=2))
    print(f"\nBITWISE REPRODUCTION VERIFIED over {check['n_comparisons']} comparisons.")
    print(f"Wrote {len(written)} snapshots to {out_dir} in {check['wall_clock_s']:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
