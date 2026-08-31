"""The experiment orchestrator (component C12, PROJECT_REPORT.md §4.1).

This is the ONLY module that ties the learner (`safelie.training`) and the
withheld evaluator (`safelie.eval`) together. It lives at the top level of
the package, deliberately outside both `safelie.training` and `safelie.eval`,
because it is the one place allowed to see both a learner's rollout and
the oracle's true-cost accounting for the same round.

Each round produces two separate, independently-written log records:

  - `rounds.jsonl`, written by `safelie.training.loop.ExperimentRun` (the
    learner): sources, attack, aggregate, dual state, reported cost.
  - `oracle.jsonl`, written HERE using `safelie.eval.harness`: true cost,
    peak violation, violation flag, and the detection gap (computed by
    comparing the oracle's true cost against the learner's own reported
    estimate — the orchestrator reads the learner's number, the learner
    never reads the oracle's).

This mirrors PROJECT_REPORT.md §8.2's schema rule verbatim: "The oracle
block must be written by the evaluator process, never by the learner."
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from safelie.envs.factory import build_env
from safelie.eval.harness import evaluate_true_cost
from safelie.eval.metrics import detection_gap, peak_violation, violation_rate
from safelie.training.loop import ExperimentRun
from safelie.utils.config import ExperimentConfig
from safelie.utils.logging import JsonlLogger


def _env_factory(cfg: ExperimentConfig):
    def factory():
        return build_env(cfg.env, rollout_length=cfg.rollout_length)

    return factory


def run_experiment_with_oracle(cfg: ExperimentConfig, eval_every: int = 1) -> Path:
    """Run the full pipeline: learner rounds, each optionally followed by
    an independent oracle evaluation episode. Returns the run's output
    directory (containing rounds.jsonl and oracle.jsonl)."""
    run = ExperimentRun(cfg)
    env_factory = _env_factory(cfg)
    oracle_logger = JsonlLogger(run.output_dir / "oracle.jsonl")
    eval_seed_rng = run.seed_bundle.rng("eval")

    num_rounds = max(1, cfg.total_steps // cfg.rollout_length)
    episodes_by_agent: dict[str, list] = {aid: [] for aid in run.env.agent_ids}

    for k in range(num_rounds):
        learner_record = run.run_round()

        if k % eval_every == 0:
            eval_seed = int(eval_seed_rng.integers(0, 2**31 - 1))
            oracle_result = evaluate_true_cost(
                env_factory=env_factory,
                agents=run.agents,
                gamma=cfg.ppo.gamma,
                budget=cfg.env.budget,
                rollout_length=cfg.rollout_length,
                seed=eval_seed,
            )
            oracle_record: dict[str, Any] = {"round_k": k, "agents": {}}
            for aid in run.env.agent_ids:
                episodes_by_agent[aid].append(oracle_result)
                reported = learner_record["constraints"][aid]["reported_cost_return"]
                gap = detection_gap(oracle_result.true_cost_return[aid], reported)
                oracle_record["agents"][aid] = {
                    "true_cost_return": oracle_result.true_cost_return[aid],
                    "reported_cost_return": reported,
                    "detection_gap": gap,
                    "peak_true_cost": oracle_result.peak_true_cost[aid],
                    "violated": oracle_result.violated[aid],
                    "violation_rate_so_far": violation_rate(episodes_by_agent[aid], aid),
                    "peak_violation_so_far": peak_violation(episodes_by_agent[aid], aid),
                }
            oracle_logger.write(oracle_record)

    oracle_logger.close()
    return run.output_dir
