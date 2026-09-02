"""Regenerate summary tables from run logs — MEASURED values only.

Report reference: PROJECT_REPORT.md Phase 8 exit criterion —
"python -m safelie.analysis.tables --table 3 regenerates the main table
from logs with measured values, and every projected number in this
report is replaced or explicitly marked as unreproduced."

**This module never reads main_iclr.tex's projected numbers and never
compares against them.** It only summarizes whatever is in the JSONL
logs a run actually produced. Any resemblance between a table this
module prints and Table 3/4 of the paper is a resemblance of *format*,
not of *content* — see PROJECT_REPORT.md §9's repeated warning against
comparing pilot-scale numbers to the paper's projections.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from safelie.eval.metrics import RunMetrics
from safelie.utils.logging import read_jsonl


def load_run_summary(run_dir: Path, last_n_rounds: int = 5) -> dict[str, RunMetrics]:
    """Average the last `last_n_rounds` rounds of a completed run into one
    `RunMetrics` per agent. Raises if the run's logs are missing or if
    fewer than one round is available (there is nothing to summarize)."""
    rounds_path = run_dir / "rounds.jsonl"
    oracle_path = run_dir / "oracle.jsonl"
    if not rounds_path.exists() or not oracle_path.exists():
        raise FileNotFoundError(f"{run_dir} does not contain both rounds.jsonl and oracle.jsonl")

    rounds = read_jsonl(rounds_path)[-last_n_rounds:]
    oracle_records = read_jsonl(oracle_path)[-last_n_rounds:]
    if not rounds or not oracle_records:
        raise ValueError(f"{run_dir}'s logs are empty")

    # Match rounds.jsonl and oracle.jsonl by round_k rather than assuming
    # equal length / matching order: a run sampled mid-round (learner
    # writes first, orchestrator appends the oracle record after) can
    # have rounds.jsonl one record ahead, and either log can be truncated
    # independently by `last_n_rounds`.
    rounds_by_k = {rec["round_k"]: rec for rec in rounds}
    oracle_by_k = {rec["round_k"]: rec for rec in oracle_records}
    common_ks = sorted(set(rounds_by_k) & set(oracle_by_k))
    if not common_ks:
        raise ValueError(f"{run_dir}: no round_k present in both rounds.jsonl and oracle.jsonl")

    agent_ids = list(rounds[-1]["constraints"].keys())
    summaries = {}
    for aid in agent_ids:
        oracle_for_agent = [oracle_by_k[k]["agents"][aid] for k in common_ks if aid in oracle_by_k[k].get("agents", {})]
        if not oracle_for_agent:
            raise ValueError(
                f"{run_dir}'s oracle.jsonl has no records for {aid} in the last "
                f"{last_n_rounds} rounds; cannot summarize evaluation quantities."
            )
        # P0 #6: evaluation quantities come from the oracle's own episodic
        # Monte-Carlo rollout (safelie.eval.harness), never from the
        # learner's GAE(lambda) training targets in rounds.jsonl --
        # `task_return`/`reported_cost_return` there are training
        # diagnostics on a different policy snapshot and a different
        # return definition (see safelie.training.loop's docstring on
        # `round_record["constraints"]`). `reported_cost_mean` uses the
        # SAME mechanism-based quantity as `detection_gap` below
        # (`mechanism_reported_cost_return`, from rounds.jsonl -- the
        # only one of the two logs that has it, since it is what drove
        # that round's dual update, not an oracle-episode quantity), so
        # the two columns stay consistent with each other in the printed
        # table.
        returns = [o["episodic_task_return"] for o in oracle_for_agent]
        reported = [rounds_by_k[k]["constraints"][aid]["mechanism_reported_cost_return"] for k in common_ks]
        true_costs = [o["true_cost_return"] for o in oracle_for_agent]
        # The mechanism-based gap (what the aggregate/dual actually saw),
        # not the agent's-own-critic gap: see docs/evaluation.md and
        # safelie.experiment's per-round computation of both.
        gaps = [o["detection_gap_vs_aggregate"] for o in oracle_for_agent]
        violation_rate = oracle_for_agent[-1]["violation_rate_so_far"]
        peak = oracle_for_agent[-1]["peak_violation_so_far"]

        summaries[aid] = RunMetrics(
            return_mean=float(np.mean(returns)),
            reported_cost_mean=float(np.mean(reported)),
            true_cost_mean=float(np.mean(true_costs)),
            violation_rate=violation_rate,
            peak_violation=peak,
            detection_gap=float(np.mean(gaps)),
        )
    return summaries


def build_summary_table(run_dirs: dict[str, Path], budget: float, last_n_rounds: int = 5) -> str:
    """`run_dirs`: {label: path to a completed run's output directory}."""
    lines = [
        "MEASURED, from local run logs. NOT the paper's Table 3/4, and "
        f"NOT comparable to it (budget d={budget}, synthetic/toy or Safe "
        "MAMuJoCo scale depending on the environment actually run — check "
        "the config).",
        "",
        "| Run | Return (mean, last N rounds) | Reported cost | True cost | Detection gap | Violation rate | Peak violation |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, run_dir in run_dirs.items():
        try:
            summary = load_run_summary(run_dir, last_n_rounds)
        except (FileNotFoundError, ValueError) as exc:
            lines.append(f"| {label} | ERROR: {exc} | | | | | |")
            continue
        for aid, m in summary.items():
            lines.append(
                f"| {label} / {aid} | {m.return_mean:.2f} | {m.reported_cost_mean:.2f} | "
                f"{m.true_cost_mean:.2f} | {m.detection_gap:.2f} | "
                f"{m.violation_rate:.2%} | {m.peak_violation:.2f} |"
            )
    return "\n".join(lines)
