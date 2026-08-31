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

    agent_ids = list(rounds[-1]["constraints"].keys())
    summaries = {}
    for aid in agent_ids:
        returns = [r["constraints"][aid]["task_return"] for r in rounds]
        reported = [r["constraints"][aid]["reported_cost_return"] for r in rounds]
        oracle_for_agent = [rec["agents"][aid] for rec in oracle_records if aid in rec.get("agents", {})]
        true_costs = [o["true_cost_return"] for o in oracle_for_agent]
        gaps = [o["detection_gap"] for o in oracle_for_agent]
        violation_rate = oracle_for_agent[-1]["violation_rate_so_far"] if oracle_for_agent else float("nan")
        peak = oracle_for_agent[-1]["peak_violation_so_far"] if oracle_for_agent else float("nan")

        summaries[aid] = RunMetrics(
            return_mean=float(np.mean(returns)),
            reported_cost_mean=float(np.mean(reported)),
            true_cost_mean=float(np.mean(true_costs)) if true_costs else float("nan"),
            violation_rate=violation_rate,
            peak_violation=peak,
            detection_gap=float(np.mean(gaps)) if gaps else float("nan"),
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
