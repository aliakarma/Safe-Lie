"""Smoke test S15: a complete clean training run, start to finish, on CPU.

Report reference: PROJECT_REPORT.md §R4.1 — "One complete clean run --
2 agents or 6 agents, tiny networks, ~2000 steps -- executes locally on
CPU start to finish and writes a complete artifact directory."
"""

from __future__ import annotations

from pathlib import Path

from safelie.experiment import run_experiment_with_oracle
from safelie.utils.logging import read_jsonl


def test_tiny_end_to_end_run_produces_a_complete_artifact_directory(tiny_config_factory):
    cfg = tiny_config_factory(run_id="s15_clean")
    out_dir = run_experiment_with_oracle(cfg)

    assert out_dir == Path(cfg.output_dir) / cfg.run_id
    rounds_path = out_dir / "rounds.jsonl"
    oracle_path = out_dir / "oracle.jsonl"
    assert rounds_path.exists()
    assert oracle_path.exists()

    rounds = read_jsonl(rounds_path)
    oracle_records = read_jsonl(oracle_path)
    expected_rounds = cfg.total_steps // cfg.rollout_length
    assert len(rounds) == expected_rounds
    assert len(oracle_records) == expected_rounds

    for record in rounds:
        for c in record["constraints"].values():
            assert len(c["reports"]) == 3  # own_critic, peer_critic_1, monitor_1
            assert "lambda_after" in c
    for record in oracle_records:
        for a in record["agents"].values():
            assert a["true_cost_return"] >= 0.0

    # no NaNs anywhere in the numeric fields
    import math

    for record in rounds:
        for c in record["constraints"].values():
            assert math.isfinite(c["task_return"])
            assert math.isfinite(c["reported_cost_return"])
            assert math.isfinite(c["lambda_after"])
