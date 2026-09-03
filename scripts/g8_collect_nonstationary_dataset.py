#!/usr/bin/env python
"""G8 (docs/g8_gates.md) step 2: collect frozen-policy trajectory pools at
FOUR policies per training anchor, so the G7 disjoint-batch source
construction can be evaluated inside the nonstationary regime.

For one anchor `k` this restores, in turn, `theta_k`, `theta_{k+1}`,
`theta_{k+2}` and `theta_{k+Delta}` (the snapshots written by
`scripts/g8_collect_training_checkpoints.py`, which are the real states of
the reproduced `pilot_A_clean_seed0` trajectory), and under each ONE
rolls out two disjoint pools of fresh episodes:

  * a `src` pool  -- the source batches (90 rounds at `theta_k`, split into
    three disjoint 30-round blocks; 30 rounds at each drifted policy)
  * a `ref` pool  -- 120 rounds, used ONLY for that policy's reference
    mean `Jhat_ref^i(theta)`

The policy is frozen for the whole of a pool: no PPO update, no GAE, no
dual update, no attack, no RCE, no replica/report-head refit, no critic
forward pass. Only `env.step` and `policy.distribution(...).sample()` are
called, byte-identical to `scripts/g7_collect_estimand_dataset.py::
collect_one_round` (itself identical to G3's). Only two scalars per
(round, agent) are stored -- `mc_cost_return` (= `G_r^i`, the paper's
`J_C` integrand, and identically the withheld oracle's true cost return in
this pipeline) and `mc_task_return`. No observation is stored except the
one saved evaluation state set described below, and no model of any kind
is fitted anywhere.

Additionally, at `theta_k`'s `src` pool round 0 only, the full 2000-step
observation stream is saved per agent as the FIXED evaluation state set
`S_k` on which the analysis script computes the exact closed-form KL
between `theta_k` and each drifted policy (docs/g8_gates.md step 4).

Run one anchor per process; the five anchors are independent and are meant
to run in parallel:

    .venv/Scripts/python.exe scripts/g8_collect_nonstationary_dataset.py --anchor 25
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

CONFIG = "configs/experiment/pilot_A_clean.yaml"

# docs/g8_gates.md step 1. Kept in ONE place shared with the snapshot
# script's own copy; both are asserted consistent at load time below.
ANCHORS = [25, 75, 125, 175, 225]
STRIDE = {25: 30, 75: 30, 125: 30, 175: 30, 225: 24}

# docs/g8_gates.md step 2: (pool name -> rounds) per policy role.
SRC_ROUNDS_ANCHOR = 90   # M=3 x R_m=30, G7's S2-M3 budget, unchanged
SRC_ROUNDS_DRIFTED = 30  # one source batch at R_m=30
REF_ROUNDS = 120

# Seed family, disjoint from every seed used in G0-G7 (G3-G6: 777;
# G7: 8801/8802/8803; training: 0 and derived offsets).
SEED_BASE = 970000
TORCH_SEED_OFFSET = 500000


def pool_seed(anchor_idx: int, policy_idx: int, pool_idx: int) -> int:
    return SEED_BASE + 1000 * anchor_idx + 10 * policy_idx + pool_idx


def collect_one_round(run: ExperimentRun, round_seed: int, keep_obs: bool = False) -> dict:
    """One frozen-policy rollout. Identical action-selection logic to
    `scripts/g7_collect_estimand_dataset.py::collect_one_round`: on-policy
    stochastic sample from the normalized observation, tanh-squashed, no
    critic call, no head refit, no gradient anywhere."""
    step = run.env.reset(seed=round_seed)
    rollouts = {aid: AgentRollout() for aid in run.env.agent_ids}
    obs_log: dict[str, list] = {aid: [] for aid in run.env.agent_ids} if keep_obs else {}

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
        if keep_obs:
            for aid in run.env.agent_ids:
                obs_log[aid].append(np.asarray(prev_obs[aid], dtype=np.float32))
        step = run.env.step(actions_taken)
        for aid in run.env.agent_ids:
            rollouts[aid].add(
                prev_obs[aid], np.zeros(1, dtype=np.float32), 0.0, step.reward,
                step.reported_cost[aid], 0.0, 0.0,
                terminated=bool(step.terminated[aid]), truncated=bool(step.truncated[aid]),
                truncation_value_bootstrap=0.0, truncation_cost_value_bootstrap=0.0,
            )

    out: dict = {"mc_cost_return": {}, "mc_task_return": {}}
    for aid in run.env.agent_ids:
        finalized = rollouts[aid].finalize(run.cfg.ppo.gamma, run.cfg.ppo.gae_lambda, 0.0, 0.0)
        out["mc_cost_return"][aid] = float(finalized["mc_cost_return"])
        out["mc_task_return"][aid] = float(finalized["mc_task_return"])
    if keep_obs:
        out["obs"] = {aid: np.stack(v) for aid, v in obs_log.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor", type=int, required=True, choices=ANCHORS)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--ckpt-dir", default="results/g8_nonstationary_source_diagnostic/checkpoints")
    ap.add_argument("--out-dir", default="results/g8_nonstationary_source_diagnostic/dataset")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    k = args.anchor
    anchor_idx = ANCHORS.index(k)
    delta = STRIDE[k]
    # policy_idx order is fixed and recorded, so a pool's seed is
    # reconstructible from its name alone.
    policies = [("anchor", k), ("plus1", k + 1), ("plus2", k + 2), ("plusdelta", k + delta)]

    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir) / f"anchor_{k:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_experiment_config(args.config)
    cfg.run_id = f"g8_frozen_collection_anchor{k}"
    cfg.output_dir = str(Path(args.out_dir).parent / "_scratch_run")
    print(f"[anchor {k}] building env + agents (Delta={delta}) ...")
    run = ExperimentRun(cfg)
    agent_ids = run.env.agent_ids

    pool_round_seeds: dict[str, list[int]] = {}
    pool_meta: dict[str, dict] = {}
    t0 = time.time()
    round_times: list[float] = []

    for policy_idx, (role, round_k) in enumerate(policies):
        ckpt = ckpt_dir / f"theta_{round_k:03d}.pt"
        if not ckpt.exists():
            print(f"[anchor {k}] MISSING checkpoint {ckpt}; run g8_collect_training_checkpoints.py first")
            return 2
        extra = run.restore(ckpt)
        assert int(extra["snapshot_round_index"]) == round_k, (extra, round_k)
        assert run.round_index == round_k, (run.round_index, round_k)
        print(f"[anchor {k}] restored theta_{round_k:03d} ({role})")

        n_src = SRC_ROUNDS_ANCHOR if role == "anchor" else SRC_ROUNDS_DRIFTED
        for pool_idx, (pool_kind, n_rounds) in enumerate((("src", n_src), ("ref", REF_ROUNDS))):
            pool_name = f"{role}_{pool_kind}"
            seed = pool_seed(anchor_idx, policy_idx, pool_idx)
            rng = np.random.default_rng(seed)
            # Independent action-sampling noise per pool: without this,
            # two pools restored from the SAME snapshot would replay the
            # identical torch stream and share their action randomness.
            torch.manual_seed(seed + TORCH_SEED_OFFSET)

            mc_cost = {aid: [] for aid in agent_ids}
            mc_task = {aid: [] for aid in agent_ids}
            round_seeds: list[int] = []
            pool_t0 = time.time()
            for r in range(n_rounds):
                round_seed = int(rng.integers(0, 2**31 - 1))
                round_seeds.append(round_seed)
                keep_obs = (role == "anchor" and pool_kind == "src" and r == 0)
                rt0 = time.time()
                res = collect_one_round(run, round_seed, keep_obs=keep_obs)
                round_times.append(time.time() - rt0)
                for aid in agent_ids:
                    mc_cost[aid].append(res["mc_cost_return"][aid])
                    mc_task[aid].append(res["mc_task_return"][aid])
                if keep_obs:
                    np.savez_compressed(
                        out_dir / "kl_eval_states.npz",
                        **{aid: res["obs"][aid] for aid in agent_ids},
                        agent_ids=np.array(agent_ids),
                        source_round_seed=np.int64(round_seed),
                        source_policy_round=np.int64(round_k),
                    )
                if (r + 1) % 10 == 0 or r == n_rounds - 1:
                    print(
                        f"[anchor {k}] {pool_name}: {r + 1}/{n_rounds}  "
                        f"pool={time.time() - pool_t0:.0f}s  total={time.time() - t0:.0f}s"
                    )

            pool_round_seeds[pool_name] = round_seeds
            np.savez_compressed(
                out_dir / f"{pool_name}.npz",
                mc_cost_return=np.stack([np.array(mc_cost[a], dtype=np.float64) for a in agent_ids], axis=1),
                mc_task_return=np.stack([np.array(mc_task[a], dtype=np.float64) for a in agent_ids], axis=1),
                round_seeds=np.array(round_seeds, dtype=np.int64),
                agent_ids=np.array(agent_ids),
            )
            pool_meta[pool_name] = {
                "policy_role": role, "policy_round": round_k, "pool_kind": pool_kind,
                "n_rounds": n_rounds, "generator_seed": seed,
                "torch_seed": seed + TORCH_SEED_OFFSET, "checkpoint": str(ckpt),
                "wall_clock_s": time.time() - pool_t0,
            }

    # G8g(iii): round-seed disjointness across every pool pair of this
    # anchor, checked in code rather than asserted in prose. Cross-anchor
    # disjointness is checked by the analysis script, which sees all five.
    names = list(pool_round_seeds)
    collisions = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = set(pool_round_seeds[names[i]]) & set(pool_round_seeds[names[j]])
            if overlap:
                collisions.append({"a": names[i], "b": names[j], "overlap": sorted(overlap)})
    disjoint_ok = not collisions
    print(f"[anchor {k}] G8g(iii) within-anchor round-seed disjointness: {disjoint_ok}")

    meta = {
        "anchor": k, "anchor_idx": anchor_idx, "stride_delta": delta,
        "policies": {role: r for role, r in policies},
        "pools": pool_meta,
        "round_seeds": pool_round_seeds,
        "config": args.config, "checkpoint_dir": str(ckpt_dir),
        "rollout_length": cfg.rollout_length, "gamma": cfg.ppo.gamma,
        "budget_d": cfg.env.budget, "agent_ids": agent_ids,
        "R_m": SRC_ROUNDS_DRIFTED, "M": 3, "R_ref": REF_ROUNDS,
        "within_anchor_round_seed_disjoint": disjoint_ok,
        "collisions": collisions,
        "n_rounds_total": sum(p["n_rounds"] for p in pool_meta.values()),
        "env_steps_total": sum(p["n_rounds"] for p in pool_meta.values()) * cfg.rollout_length,
        "mean_round_wall_clock_s": float(np.mean(round_times)),
        "median_round_wall_clock_s": float(np.median(round_times)),
        "wall_clock_s": time.time() - t0,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[anchor {k}] done in {meta['wall_clock_s']:.0f}s "
        f"({meta['n_rounds_total']} rounds, {meta['mean_round_wall_clock_s']:.2f}s/round)"
    )
    return 0 if disjoint_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
