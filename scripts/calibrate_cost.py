#!/usr/bin/env python
"""Check that a config's cost constraint actually binds, before spending GPU time.

Usage:
    python scripts/calibrate_cost.py --config configs/experiment/pilot_A_clean.yaml
    python scripts/calibrate_cost.py --config <cfg> --sweep 0.5 0.75 1.0 1.5 2.418

PROJECT_REPORT.md §R6.1 names a precondition failure that is easy to miss
and fatal to interpret: if the constraint never binds, lambda stays at
zero, the safety pathway the paper studies is never exercised, and every
condition (clean / attacked / defended / benign) produces the same
numbers -- for reasons that have nothing to do with the hypothesis. The
run looks like a clean null result. It is not a result at all.

This is not hypothetical for Safe MAMuJoCo. Its velocity thresholds are
calibrated against a *converged* unconstrained PPO agent, whereas the
compact pilot trains for 5e5 steps. Measured here, a randomly-initialized
ManySegmentAnt 6x1 policy exceeds Ant's 2.418 threshold on well under 1%
of steps, giving a discounted cost return near 0.75 against the paper's
budget d=25 -- a constraint that is roughly 30x from binding. The same
measurement is what set `local_demo_clean.yaml`'s budget to d=5 on the
synthetic environment.

Two numbers come out, and they are not interchangeable. The *true*
discounted cost is what the environment charges and the withheld oracle
records -- it says whether the threshold makes the task
constraint-relevant at all. The *learner's estimate* is what the cost
critic believes, and it alone drives the dual update, so it is what
decides whether lambda ever leaves zero. At initialization the estimate
runs several times below the truth (untrained cost critic; GAE's
effective horizon is ~17 steps at gamma=0.99, lambda=0.95, not ~100), and
that gap closes only as the critic trains.

Both are measured under the *initial* policy, so this bounds the problem
from one side only: training moves the rate, and a constraint slack at
initialization may bind later, or the reverse. Read it as a go/no-go
screen for obviously-vacuous settings, not as a guarantee the pilot is
well-posed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safelie.algos.networks import AgentBundle  # noqa: E402
from safelie.envs.factory import build_env  # noqa: E402
from safelie.training.buffer import AgentRollout  # noqa: E402
from safelie.utils.config import load_experiment_config  # noqa: E402
from safelie.utils.seeding import seed_everything  # noqa: E402


def measure(cfg, steps: int, threshold: float | None = None) -> dict:
    """Roll out a freshly-initialized policy and measure cost two ways.

    Mirrors `safelie.training.loop.ExperimentRun`'s construction exactly
    (same seeding, same network shapes, same tanh-squashed sampling, same
    `AgentRollout.finalize`) so both measurements are the ones the real
    run would start from.

    The two quantities differ, and the difference matters:

    *True* discounted cost is the physical fact -- what the environment
    actually charges, and what the withheld oracle records. It answers
    "does this threshold make the task constraint-relevant at all?"

    *Estimated* cost return is `ret_c[0]` from GAE over the cost stream:
    what the learner's cost critic believes, what the sources report, and
    therefore what the dual update actually compares against the budget.
    It answers "will lambda ever leave zero?" -- which is the question
    that decides whether the run tests anything.

    At initialization the second is far below the first: the cost critic
    is random, and GAE at gamma=0.99, lambda=0.95 has an effective horizon
    near 1/(1 - 0.99*0.95) ~ 17 steps rather than the ~100 of the
    undiscounted-horizon return. The gap closes as the cost critic trains.
    A run can therefore be genuinely constraint-relevant in truth while
    lambda still sits at zero for many rounds, so both numbers are
    reported and neither alone is a verdict.
    """
    seed_everything(cfg.seed)
    env_cfg = cfg.env.model_copy(update={"velocity_threshold": threshold}) if threshold else cfg.env
    env = build_env(env_cfg, rollout_length=cfg.rollout_length)
    agents = {
        aid: AgentBundle(
            int(env.obs_dim), int(env.action_dim), cfg.ppo.hidden_dim, cfg.ppo.lr, critic_lr=cfg.ppo.critic_lr
        )
        for aid in env.agent_ids
    }

    step = env.reset(seed=cfg.seed)
    rollouts = {aid: AgentRollout() for aid in env.agent_ids}
    per_step: list[list[float]] = []
    discounted = dict.fromkeys(env.agent_ids, 0.0)
    for t in range(steps):
        actions, raws, lps, vals, cvals = {}, {}, {}, {}, {}
        for aid in env.agent_ids:
            obs_t = torch.as_tensor(step.obs[aid], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                # P0 #2: act on the normalized observation, matching the
                # real training loop (safelie.training.loop);
                # .value/.cost_value normalize internally already.
                obs_n = agents[aid].normalize_obs_tensor(obs_t)
                dist = agents[aid].policy.distribution(obs_n)
                raw = dist.sample()
                lps[aid] = float(dist.log_prob(raw).sum(-1).item())
                vals[aid] = float(agents[aid].value(obs_t).item())
                cvals[aid] = float(agents[aid].cost_value(obs_t).item())
            raws[aid] = raw.squeeze(0).numpy()
            actions[aid] = torch.tanh(raw).squeeze(0).numpy()

        prev_obs = step.obs
        step = env.step(actions)
        per_step.append([step.reported_cost[aid] for aid in env.agent_ids])
        # Mirrors safelie.training.loop.ExperimentRun.run_round's
        # terminated/truncated handling exactly (P0 terminal-handling
        # fix): a truncation bootstraps from the true final observation,
        # not from 0.
        final_obs = step.info.get("final_observation")
        for aid in env.agent_ids:
            discounted[aid] += (cfg.ppo.gamma**t) * step.reported_cost[aid]
            trunc_v = trunc_cv = 0.0
            if final_obs is not None and bool(step.truncated[aid]) and not bool(step.terminated[aid]):
                fobs_t = torch.as_tensor(final_obs[aid], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    trunc_v = float(agents[aid].value(fobs_t).item())
                    trunc_cv = float(agents[aid].cost_value(fobs_t).item())
            rollouts[aid].add(
                prev_obs[aid], raws[aid], lps[aid], step.reward,
                step.reported_cost[aid], vals[aid], cvals[aid],
                terminated=bool(step.terminated[aid]), truncated=bool(step.truncated[aid]),
                truncation_value_bootstrap=trunc_v, truncation_cost_value_bootstrap=trunc_cv,
            )

    estimated = []
    for aid in env.agent_ids:
        obs_last = torch.as_tensor(step.obs[aid], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            last_v = float(agents[aid].value(obs_last).item())
            last_cv = float(agents[aid].cost_value(obs_last).item())
        final = rollouts[aid].finalize(cfg.ppo.gamma, cfg.ppo.gae_lambda, last_v, last_cv)
        estimated.append(final["cost_return_estimate"])

    rates = np.asarray(per_step).mean(axis=0)
    # The rollout is finite; report what an infinite-horizon discounted
    # return would be at this rate, since that is the quantity the budget
    # is compared against once the critics have converged.
    asymptotic = rates / (1.0 - cfg.ppo.gamma)
    return {
        "agent_ids": list(env.agent_ids),
        "rates": rates,
        "discounted": np.array([discounted[a] for a in env.agent_ids]),
        "asymptotic": asymptotic,
        "estimated": np.asarray(estimated),
        "threshold": getattr(env, "velocity_threshold", None),
        "backend": getattr(env, "backend", "synthetic"),
        "cost_mode": getattr(env, "cost_mode", "n/a"),
    }


def _verdict(asymptotic: np.ndarray, budget: float) -> tuple[str, str]:
    ratio = float(asymptotic.mean() / budget) if budget else float("inf")
    if ratio < 0.25:
        return "NON-BINDING", (
            "The constraint is far from active. lambda will sit at zero, no condition "
            "will differ from any other, and the run will not test the hypothesis. "
            "Lower the velocity threshold (or the budget) before spending compute."
        )
    if ratio > 4.0:
        return "SATURATED", (
            "The constraint is violated so heavily that lambda will rail against "
            "lambda_max and the policy will collapse toward inaction. Raise the "
            "threshold (or the budget)."
        )
    return "BINDING", (
        "The constraint is active at initialization and controllable. This is the "
        "regime the pilot needs -- though training will move the rate, so re-check "
        "against the logged reported_cost_return once the run is under way."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument(
        "--sweep",
        type=float,
        nargs="*",
        default=None,
        help="Velocity thresholds to compare instead of the config's own.",
    )
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    budget = cfg.env.budget

    print("=" * 78)
    print(f"COST CALIBRATION -- {cfg.run_id}  (env={cfg.env.name}, budget d={budget})")
    print("Initial-policy measurement only; see this script's docstring for what it")
    print("does and does not establish (PROJECT_REPORT.md §R6.1).")
    print("=" * 78)

    thresholds = args.sweep if args.sweep else [None]
    for thr in thresholds:
        m = measure(cfg, args.steps, thr)
        label = f"threshold={m['threshold']}" if m["threshold"] is not None else "config default"
        verdict, advice = _verdict(m["asymptotic"], budget)
        print(f"\n-- {label}  [backend={m['backend']}, cost_mode={m['cost_mode']}]")
        print(f"   per-agent cost rate : {np.array2string(m['rates'], precision=3)}")
        print(f"   J_C over {args.steps} steps: {np.array2string(m['discounted'], precision=2)}")
        print(f"   J_C asymptotic      : {np.array2string(m['asymptotic'], precision=1)}")
        print(f"   mean / budget       : {m['asymptotic'].mean() / budget:.2f}x")
        print(f"   TASK VERDICT        : {verdict}")
        print(f"   {advice}")

        # What the dual update actually sees. Reported second and framed
        # as a lag, not a second verdict: at initialization it is expected
        # to be well below the true cost, and it is not a reason to
        # retune the threshold.
        est = m["estimated"]
        print(f"   learner estimate    : {np.array2string(est, precision=1)}  "
              f"({est.mean() / budget:.2f}x budget)")
        if est.mean() < 0.5 * m["asymptotic"].mean():
            print(
                "   NOTE: the learner's cost-return estimate lags the true cost, as "
                "expected from an untrained cost critic under GAE. lambda stays at "
                "zero until the critic catches up -- watch reported_cost_return in "
                "rounds.jsonl rather than retuning the threshold on this number."
            )

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
