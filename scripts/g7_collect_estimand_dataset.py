#!/usr/bin/env python
"""G7 (docs/g7_gates.md): collect three fresh, disjoint frozen-policy
rollout pools of `G_r^i = mc_cost_return` -- the trajectory-level
discounted cost return, `safelie.training.constraint_return.
discounted_window_return` on `reported_cost` -- for every owner, every
round.

This is deliberately NOT `scripts/g3_collect_calibration_dataset.py`
rerun: G3's dataset (seed 777) is training/eval data for neural
`constraint_report_heads` and has already been read by G3-G6. G7 tests a
different estimand (independent sample means of the trajectory-level
return, not per-step regression) and per docs/g7_gates.md section 3 must
use a seed independent of every prior collection. No observation, no
regression target, no critic call is stored or computed anywhere in this
script -- only the one scalar `mc_cost_return` per (round, agent), because
the primary G7 estimator is a sample mean, not a fitted model (task
section 11).

Restores the completed G2 `pilot_A_clean_seed0` checkpoint (round 249) and
rolls out 450 fresh episodes under the FROZEN policy -- no PPO update, no
GAE, no dual update, no attack, no change to the environment or the
config. Only `env.step` and each agent's already-trained
`policy.distribution` are called, identical action-selection code to
G3's `collect_one_round` (on-policy stochastic sample, tanh-squashed).

Writes `results/g7_estimand_source_diagnostic/dataset/{pool}.npz`, one per
pool (`eval_pool_a`, `reference_pool`, `eval_pool_b`), each holding:

  mc_cost_return   (n_rounds, n_agents)  float64   G_r^i, round-major
  round_seeds      (n_rounds,)           int64     the env.reset seed used

plus `meta.json` recording every seed, the round-seed-disjointness check
(G7a), and wall-clock time.

Usage:
    .venv/Scripts/python.exe scripts/g7_collect_estimand_dataset.py
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

import numpy as np
import torch

from safelie.training.buffer import AgentRollout
from safelie.training.loop import ExperimentRun
from safelie.utils.config import load_experiment_config

CHECKPOINT = "results/runs_constraint_mc_g2/pilot_A_clean_seed0/checkpoint.pt"
CONFIG = "configs/experiment/pilot_A_clean.yaml"

# Pool name -> (generator seed, round count). All three seeds are NEW:
# disjoint from G3-G6's collection seed (777) and from every training
# seed used anywhere in G0-G6 (0, and derived offsets 1000/4000/6000/
# 21000-24000/90000s), per docs/g7_gates.md "Data generation".
POOLS: dict[str, tuple[int, int]] = {
    "eval_pool_a": (8801, 90),
    "reference_pool": (8802, 270),
    "eval_pool_b": (8803, 90),
}


def collect_one_round(run: ExperimentRun, round_seed: int) -> dict:
    """One frozen-policy rollout round. Byte-identical action-selection
    logic to `scripts/g3_collect_calibration_dataset.py::collect_one_round`
    -- calls no critic, no PPO update, no dual update, refits no head.
    Returns each agent's `mc_cost_return` only (the rest of `finalize()`'s
    dict is computed internally by `AgentRollout.finalize` but discarded
    here, since G7 needs no per-step obs/target)."""
    step = run.env.reset(seed=round_seed)
    rollouts = {aid: AgentRollout() for aid in run.env.agent_ids}
    for _t in range(run.cfg.rollout_length):
        actions_taken: dict = {}
        for aid in run.env.agent_ids:
            obs_t = torch.as_tensor(step.obs[aid], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                obs_n = run.agents[aid].normalize_obs_tensor(obs_t)
                dist = run.agents[aid].policy.distribution(obs_n)
                raw_action = dist.sample()
                action = torch.tanh(raw_action)
            actions_taken[aid] = action.squeeze(0).numpy()

        prev_obs = step.obs
        step = run.env.step(actions_taken)
        for aid in run.env.agent_ids:
            rollouts[aid].add(
                prev_obs[aid], np.zeros(1, dtype=np.float32), 0.0, step.reward,
                step.reported_cost[aid], 0.0, 0.0,
                terminated=bool(step.terminated[aid]), truncated=bool(step.truncated[aid]),
                truncation_value_bootstrap=0.0, truncation_cost_value_bootstrap=0.0,
            )

    mc_cost_return = {}
    for aid in run.env.agent_ids:
        finalized = rollouts[aid].finalize(run.cfg.ppo.gamma, run.cfg.ppo.gae_lambda, 0.0, 0.0)
        mc_cost_return[aid] = float(finalized["mc_cost_return"])
    return mc_cost_return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--out-dir", default="results/g7_estimand_source_diagnostic/dataset")
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    cfg.run_id = "g7_frozen_policy_collection"
    cfg.output_dir = str(Path(args.out_dir).parent / "_scratch_run")
    print(f"Building env + agents from {args.config} ...")
    run = ExperimentRun(cfg)
    print(f"Restoring frozen policy from {args.checkpoint} ...")
    extra = run.restore(Path(args.checkpoint))
    print(f"Restored at round_index={run.round_index} (extra keys: {list(extra.keys())})")

    agent_ids = run.env.agent_ids
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_round_seeds: dict[str, list[int]] = {}
    t0 = time.time()
    for pool_name, (gen_seed, n_rounds) in POOLS.items():
        rng = np.random.default_rng(gen_seed)
        mc_by_agent = {aid: [] for aid in agent_ids}
        round_seeds: list[int] = []
        pool_t0 = time.time()
        for r in range(n_rounds):
            round_seed = int(rng.integers(0, 2**31 - 1))
            round_seeds.append(round_seed)
            result = collect_one_round(run, round_seed)
            for aid in agent_ids:
                mc_by_agent[aid].append(result[aid])
            elapsed = time.time() - pool_t0
            total_elapsed = time.time() - t0
            print(
                f"  [{pool_name}] round {r + 1}/{n_rounds} done, seed={round_seed}, "
                f"pool_elapsed={elapsed:.1f}s, total_elapsed={total_elapsed:.1f}s"
            )

        pool_round_seeds[pool_name] = round_seeds
        mc_matrix = np.stack([np.array(mc_by_agent[aid], dtype=np.float64) for aid in agent_ids], axis=1)
        np.savez_compressed(
            out_dir / f"{pool_name}.npz",
            mc_cost_return=mc_matrix,  # (n_rounds, n_agents)
            round_seeds=np.array(round_seeds, dtype=np.int64),
            agent_ids=np.array(agent_ids),
        )
        print(f"[{pool_name}] wrote {mc_matrix.shape} -> {out_dir / f'{pool_name}.npz'}")

    # G7a (docs/g7_gates.md): assert round-seed disjointness across pools
    # IN CODE, not by inspection after the fact.
    names = list(pool_round_seeds.keys())
    disjoint_ok = True
    collisions: list[dict] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = set(pool_round_seeds[a]) & set(pool_round_seeds[b])
            if overlap:
                disjoint_ok = False
                collisions.append({"pool_a": a, "pool_b": b, "overlap": sorted(overlap)})
    if not disjoint_ok:
        print(f"G7a FAILED: round-seed collisions found: {collisions}")
    else:
        print("G7a: round-seed sets are pairwise disjoint across all three pools (checked in code).")

    meta = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "pools": {name: {"seed": seed, "n_rounds": n} for name, (seed, n) in POOLS.items()},
        "rollout_length": cfg.rollout_length,
        "gamma": cfg.ppo.gamma,
        "obs_dim": run.obs_dim,
        "agent_ids": agent_ids,
        "budget_d": cfg.env.budget,
        "restored_round_index": run.round_index,
        "g7a_round_seed_disjoint": disjoint_ok,
        "g7a_collisions": collisions,
        "wall_clock_s": time.time() - t0,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone in {meta['wall_clock_s']:.1f}s. Metadata: {out_dir / 'meta.json'}")
    return 0 if disjoint_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
