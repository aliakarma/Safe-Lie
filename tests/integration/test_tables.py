"""Integration test: run a tiny experiment end-to-end, then regenerate a
summary table from its logs (Phase 8 exit criterion)."""

from __future__ import annotations

import math

from safelie.analysis.tables import build_summary_table, load_run_summary
from safelie.experiment import run_experiment_with_oracle


def test_summary_table_regenerates_from_a_completed_run(tiny_config_factory):
    cfg = tiny_config_factory(run_id="analysis_test")
    out_dir = run_experiment_with_oracle(cfg)

    summary = load_run_summary(out_dir, last_n_rounds=2)
    assert set(summary) == {"agent_0", "agent_1", "agent_2"}
    for m in summary.values():
        assert math.isfinite(m.return_mean)
        assert math.isfinite(m.reported_cost_mean)
        assert math.isfinite(m.true_cost_mean)
        assert math.isfinite(m.detection_gap)
        assert 0.0 <= m.violation_rate <= 1.0

    table = build_summary_table({"clean": out_dir}, budget=cfg.env.budget)
    assert "clean / agent_0" in table
    assert "MEASURED" in table
