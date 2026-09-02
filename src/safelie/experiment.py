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

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safelie.envs.factory import build_env
from safelie.eval.harness import evaluate_true_cost
from safelie.eval.metrics import detection_gap, peak_violation, violation_rate
from safelie.training.loop import ExperimentRun
from safelie.utils.config import ExperimentConfig
from safelie.utils.logging import JsonlLogger


def _git_sha() -> dict[str, Any]:
    """The commit this run's code came from, plus whether the tree was
    dirty at launch. A dirty tree is recorded, never silently tolerated:
    the SHA alone does not identify the code that ran if there are
    uncommitted edits, so both facts are needed to reproduce a run."""
    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                args, cwd=Path(__file__).resolve().parents[2],
                capture_output=True, text=True, timeout=30, check=False,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    sha = _run(["git", "rev-parse", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    return {
        "sha": sha,
        "dirty": bool(dirty) if dirty is not None else None,
        "dirty_paths": sorted(line[3:] for line in dirty.splitlines()) if dirty else [],
    }


def _write_run_metadata(out_dir: Path, cfg: ExperimentConfig, status: str, **extra: Any) -> None:
    """Everything needed to identify and rerun this run, in one file.

    Written twice: once as `status="running"` before round 0, once as
    `status="complete"` after the final checkpoint. A run interrupted
    mid-way therefore leaves `status="running"` on disk, which is what
    distinguishes a genuinely finished run from a killed one during
    aggregation -- rather than inferring completion from a line count.
    """
    record = {
        "run_id": cfg.run_id,
        "seed": cfg.seed,
        "status": status,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "git": _git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "num_rounds_planned": max(1, cfg.total_steps // cfg.rollout_length),
        "config_snapshot": json.loads(cfg.model_dump_json()),
        **extra,
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(record, indent=2, sort_keys=False), encoding="utf-8"
    )


def _env_factory(cfg: ExperimentConfig):
    def factory():
        return build_env(cfg.env, rollout_length=cfg.rollout_length)

    return factory


def run_experiment_with_oracle(
    cfg: ExperimentConfig,
    eval_every: int = 1,
    checkpoint_every: int = 1,
    auto_resume: bool = True,
) -> Path:
    """Run the full pipeline: learner rounds, each optionally followed by
    an independent oracle evaluation episode.

    Supports automatic checkpointing and resuming:
      - If auto_resume is True and a checkpoint exists in run.output_dir,
        restores the state and resumes from the last completed round.
      - Saves a checkpoint every `checkpoint_every` rounds and on completion.
    """
    run = ExperimentRun(cfg)
    ckpt_path = run.output_dir / "checkpoint.pt"

    eval_seed_rng = run.seed_bundle.rng("eval")
    episodes_by_agent: dict[str, list] = {aid: [] for aid in run.env.agent_ids}

    start_k = 0
    if auto_resume and ckpt_path.exists():
        extra = run.restore(ckpt_path)
        start_k = run.round_index
        if "eval_rng" in extra:
            eval_seed_rng.bit_generator.state = extra["eval_rng"]
        else:
            for _ in range(start_k):
                eval_seed_rng.integers(0, 2**31 - 1)
        if "episodes_by_agent" in extra:
            episodes_by_agent = extra["episodes_by_agent"]
        # Re-attach round logger in append mode
        run.round_logger = JsonlLogger(run.output_dir / "rounds.jsonl")
        print(f"Resumed {cfg.run_id} from round {start_k} via {ckpt_path}")

    env_factory = _env_factory(cfg)
    oracle_logger = JsonlLogger(run.output_dir / "oracle.jsonl")

    num_rounds = max(1, cfg.total_steps // cfg.rollout_length)
    _write_run_metadata(run.output_dir, cfg, "running", resumed_from_round=start_k)

    for k in range(start_k, num_rounds):
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
                constraint = learner_record["constraints"][aid]
                # P0 #6: two DIFFERENT, both legitimate, "reported cost"
                # baselines for Delta = J_true_C - J_reported_C, and they
                # answer different questions. `reported_cost_return` is
                # the agent's own cost-critic GAE estimate -- the attack
                # never touches it (corruption lands in the source
                # reports and the aggregate), so a gap here measures
                # critic estimation error, not the attack.
                # `mechanism_reported_cost_return` is the post-attack,
                # post-aggregation return-scale estimate that actually
                # drove this round's dual update -- the channel the paper
                # studies, and the metric analyze_matrix.py already
                # treated as primary (`detection_gap_vs_aggregate`,
                # docs/evaluation.md). Both are logged; neither is
                # silently preferred by only reporting one.
                own_critic_reported = constraint["reported_cost_return"]
                mechanism_reported = constraint["mechanism_reported_cost_return"]
                gap_vs_own_critic = detection_gap(oracle_result.true_cost_return[aid], own_critic_reported)
                gap_vs_aggregate = detection_gap(oracle_result.true_cost_return[aid], mechanism_reported)
                # G1 (docs/g1_gates.md): the head-to-head estimator
                # comparison, written HERE rather than by the learner
                # because it needs the oracle's truth and the learner is
                # never allowed to see it. The learner logged both
                # estimators from the same rollout; this block pairs each
                # against the same oracle episode, so `bias_gae` and
                # `bias_mc` differ ONLY in the estimator and share every
                # other source of error (the two rollouts' own sampling
                # noise, and the fact that the oracle measures theta_{k+1}
                # while the learner's rollout was drawn under theta_k).
                # That shared-error property is what makes the DIFFERENCE
                # of the two biases interpretable even though neither
                # bias in isolation is pure estimator error.
                estimators = constraint.get("constraint_estimators", {})
                true_i = oracle_result.true_cost_return[aid]
                estimator_block = {
                    "active": estimators.get("active"),
                    "gae_lambda": estimators.get("gae_lambda"),
                    "mc_window": estimators.get("mc_window"),
                    "mc_episodic": estimators.get("mc_episodic"),
                    "mc_window_task_return": estimators.get("mc_window_task_return"),
                    "mc_n_complete_episodes": estimators.get("mc_n_complete_episodes"),
                }
                for key in ("gae_lambda", "mc_window", "mc_episodic"):
                    v = estimators.get(key)
                    estimator_block[f"bias_{key}"] = (
                        float(v) - float(true_i) if v is not None else None
                    )
                oracle_record["agents"][aid] = {
                    "constraint_estimators": estimator_block,
                    "true_cost_return": oracle_result.true_cost_return[aid],
                    "episodic_task_return": oracle_result.episodic_task_return,
                    "episodic_reported_cost_return": oracle_result.episodic_reported_cost_return[aid],
                    "reported_cost_return": own_critic_reported,
                    "mechanism_reported_cost_return": mechanism_reported,
                    "detection_gap": gap_vs_own_critic,
                    "detection_gap_vs_aggregate": gap_vs_aggregate,
                    "peak_true_cost": oracle_result.peak_true_cost[aid],
                    "violated": oracle_result.violated[aid],
                    "violation_rate_so_far": violation_rate(episodes_by_agent[aid], aid),
                    "peak_violation_so_far": peak_violation(episodes_by_agent[aid], aid),
                }
            oracle_logger.write(oracle_record)

        if checkpoint_every > 0 and (k + 1) % checkpoint_every == 0:
            run.checkpoint(
                ckpt_path,
                extra={
                    "eval_rng": eval_seed_rng.bit_generator.state,
                    "episodes_by_agent": episodes_by_agent,
                },
            )

    if checkpoint_every > 0:
        run.checkpoint(
            ckpt_path,
            extra={
                "eval_rng": eval_seed_rng.bit_generator.state,
                "episodes_by_agent": episodes_by_agent,
            },
        )

    run.round_logger.close()
    oracle_logger.close()
    _write_run_metadata(
        run.output_dir, cfg, "complete",
        rounds_completed=run.round_index,
        checkpoint=str(ckpt_path.name) if ckpt_path.exists() else None,
        eval_every=eval_every,
    )
    return run.output_dir
