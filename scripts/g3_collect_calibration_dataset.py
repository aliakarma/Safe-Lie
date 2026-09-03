#!/usr/bin/env python
"""G3 (docs/g3_gates.md): build the ONE fixed source-calibration dataset.

Restores the completed G2 `pilot_A_clean_seed0` checkpoint (round 249) and
rolls out 60 fresh episodes under the FROZEN policy -- no PPO update, no
GAE, no dual update, no attack, no refit of `constraint_report_heads`, no
change to the environment or the config. Only `env.step` and each agent's
already-trained `policy.distribution` are called.

Writes one `.npz` per agent to `--out-dir` (default
`results/g3_source_diagnostic/dataset`):

  fit_obs            (N_fit_rows, obs_dim)   float32   pooled over rounds 0-39, masked
  fit_target         (N_fit_rows,)           float64   mc_cost_to_go, same rows
  eval_query_obs     (20, obs_dim)           float32   obs[0] of rounds 40-59
  eval_truth         (20,)                   float64   mc_cost_return of rounds 40-59

Usage:
    .venv/Scripts/python.exe scripts/g3_collect_calibration_dataset.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from safelie.training.buffer import AgentRollout
from safelie.training.loop import ExperimentRun
from safelie.utils.config import load_experiment_config

CHECKPOINT = "results/runs_constraint_mc_g2/pilot_A_clean_seed0/checkpoint.pt"
CONFIG = "configs/experiment/pilot_A_clean.yaml"
N_FIT_ROUNDS = 40
N_EVAL_ROUNDS = 20
COLLECTION_SEED = 777


def collect_one_round(run: ExperimentRun, round_seed: int) -> dict:
    """One frozen-policy rollout round. Mirrors `ExperimentRun.run_round`'s
    action-selection code exactly (obs normalization + stochastic sample +
    tanh squash), but calls no critic, no PPO update, no dual update, and
    refits no head -- this function only ever reads `run.agents[*].policy`
    and `run.env`."""
    step = run.env.reset(seed=round_seed)
    rollouts = {aid: AgentRollout() for aid in run.env.agent_ids}
    for _t in range(run.cfg.rollout_length):
        actions_taken: dict = {}
        raw_actions: dict = {}
        logprobs: dict = {}
        for aid in run.env.agent_ids:
            obs_t = torch.as_tensor(step.obs[aid], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                obs_n = run.agents[aid].normalize_obs_tensor(obs_t)
                dist = run.agents[aid].policy.distribution(obs_n)
                raw_action = dist.sample()
                logprob = dist.log_prob(raw_action).sum(-1)
                action = torch.tanh(raw_action)
            raw_actions[aid] = raw_action.squeeze(0).numpy()
            actions_taken[aid] = action.squeeze(0).numpy()
            logprobs[aid] = float(logprob.item())

        prev_obs = step.obs
        step = run.env.step(actions_taken)
        for aid in run.env.agent_ids:
            rollouts[aid].add(
                prev_obs[aid], raw_actions[aid], logprobs[aid], step.reward,
                step.reported_cost[aid], 0.0, 0.0,
                terminated=bool(step.terminated[aid]), truncated=bool(step.truncated[aid]),
                truncation_value_bootstrap=0.0, truncation_cost_value_bootstrap=0.0,
            )

    finalized = {}
    for aid in run.env.agent_ids:
        # last_value/last_cost_value only feed GAE's ret_r/ret_c, which this
        # diagnostic never reads (see module docstring) -- 0.0 is inert here.
        finalized[aid] = rollouts[aid].finalize(run.cfg.ppo.gamma, run.cfg.ppo.gae_lambda, 0.0, 0.0)
    return finalized


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--out-dir", default="results/g3_source_diagnostic/dataset")
    ap.add_argument("--n-fit-rounds", type=int, default=N_FIT_ROUNDS)
    ap.add_argument("--n-eval-rounds", type=int, default=N_EVAL_ROUNDS)
    ap.add_argument("--collection-seed", type=int, default=COLLECTION_SEED)
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    # Isolate this diagnostic's own ExperimentRun construction (which
    # unconditionally creates an output directory and a JsonlLogger) away
    # from results/runs/pilot_A_clean -- that path is reserved for a real
    # run of this exact config and must not collide with a frozen-policy
    # data-collection pass that writes no round records at all.
    cfg.run_id = "g3_frozen_policy_collection"
    cfg.output_dir = str(Path(args.out_dir).parent / "_scratch_run")
    print(f"Building env + agents from {args.config} ...")
    run = ExperimentRun(cfg)
    print(f"Restoring frozen policy from {args.checkpoint} ...")
    extra = run.restore(Path(args.checkpoint))
    print(f"Restored at round_index={run.round_index} (extra keys: {list(extra.keys())})")

    collect_rng = np.random.default_rng(args.collection_seed)
    n_total = args.n_fit_rounds + args.n_eval_rounds
    agent_ids = run.env.agent_ids
    obs_dim = run.obs_dim

    fit_obs_by_agent = {aid: [] for aid in agent_ids}
    fit_target_by_agent = {aid: [] for aid in agent_ids}
    eval_query_obs_by_agent = {aid: [] for aid in agent_ids}
    eval_truth_by_agent = {aid: [] for aid in agent_ids}

    t0 = time.time()
    for r in range(n_total):
        round_seed = int(collect_rng.integers(0, 2**31 - 1))
        finalized = collect_one_round(run, round_seed)
        is_fit = r < args.n_fit_rounds
        for aid in agent_ids:
            n_mc = int(finalized[aid]["n_mc_targets"])
            if is_fit:
                fit_obs_by_agent[aid].append(finalized[aid]["obs"][:n_mc].astype(np.float32))
                fit_target_by_agent[aid].append(finalized[aid]["mc_cost_to_go"][:n_mc])
            else:
                eval_query_obs_by_agent[aid].append(finalized[aid]["obs"][0].astype(np.float32))
                eval_truth_by_agent[aid].append(float(finalized[aid]["mc_cost_return"]))
        kind = "fit" if is_fit else "eval"
        elapsed = time.time() - t0
        print(f"  round {r + 1}/{n_total} ({kind}) done, seed={round_seed}, elapsed={elapsed:.1f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for aid in agent_ids:
        np.savez_compressed(
            out_dir / f"{aid}.npz",
            fit_obs=np.concatenate(fit_obs_by_agent[aid], axis=0),
            fit_target=np.concatenate(fit_target_by_agent[aid], axis=0),
            eval_query_obs=np.stack(eval_query_obs_by_agent[aid], axis=0),
            eval_truth=np.array(eval_truth_by_agent[aid], dtype=np.float64),
        )
        n_fit_rows = sum(len(x) for x in fit_target_by_agent[aid])
        print(f"{aid}: {n_fit_rows} fit rows, {len(eval_truth_by_agent[aid])} eval rounds -> {out_dir / f'{aid}.npz'}")

    meta = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "collection_seed": args.collection_seed,
        "n_fit_rounds": args.n_fit_rounds,
        "n_eval_rounds": args.n_eval_rounds,
        "rollout_length": cfg.rollout_length,
        "gamma": cfg.ppo.gamma,
        "obs_dim": obs_dim,
        "agent_ids": agent_ids,
        "budget_d": cfg.env.budget,
        "restored_round_index": run.round_index,
        "wall_clock_s": time.time() - t0,
    }
    import json

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone in {meta['wall_clock_s']:.1f}s. Metadata: {out_dir / 'meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
