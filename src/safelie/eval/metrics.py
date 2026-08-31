"""Metric definitions: violation rate, peak violation, detection gap.

Report reference: main_iclr.tex §5.1; PROJECT_REPORT.md Appendix B gaps
G9 (violation rate definition) and G18 (peak violation, "declared and
never reported... implement it anyway").
"""

from __future__ import annotations

from dataclasses import dataclass

from safelie.eval.oracle import OracleEpisodeResult


def detection_gap(true_cost: float, reported_cost: float) -> float:
    """Delta = J_true_C - J_reported_C (main_iclr.tex §5.1). Named
    `detection_gap` in code, never `Delta`, to avoid the paper's own
    notation collision with the cumulative corruption budget (W12)."""
    return true_cost - reported_cost


def violation_rate(episodes: list[OracleEpisodeResult], agent_id: str) -> float:
    """`[GAP]` G9 resolution: fraction of evaluation episodes in which
    J_true_C > d for the given agent."""
    if not episodes:
        return 0.0
    violations = sum(1 for ep in episodes if ep.violated.get(agent_id, False))
    return violations / len(episodes)


def peak_violation(episodes: list[OracleEpisodeResult], agent_id: str) -> float:
    """`[GAP]` G18 resolution: max per-step true cost observed across all
    evaluation episodes, for the given agent."""
    if not episodes:
        return 0.0
    return max(ep.peak_true_cost.get(agent_id, 0.0) for ep in episodes)


@dataclass(frozen=True)
class RunMetrics:
    """One run's (one seed's) summary metrics for one agent."""

    return_mean: float
    reported_cost_mean: float
    true_cost_mean: float
    violation_rate: float
    peak_violation: float
    detection_gap: float
