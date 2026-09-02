"""P0 #9: empirical source-diversity diagnostic.

Verifies the diagnostic distinguishes DECLARED independence (config
labels, safelie.governance.auditor) from MEASURED independence (this
module) -- the scientific point being that a config can declare M
distinct independence classes while the sources' actual errors are
strongly correlated in practice (e.g. the P0 #7 self-peer-leakage bug:
two "independent" sources that are, in fact, the same critic evaluated
on the same observation).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from safelie.governance.empirical_diversity import (
    _participation_ratio,
    compute_empirical_diversity,
    format_report,
    write_report,
)


class TestParticipationRatio:
    def test_identity_matrix_gives_full_nominal_diversity(self):
        m = 5
        assert _participation_ratio(np.eye(m)) == pytest.approx(m, rel=1e-6)

    def test_all_ones_matrix_gives_minimal_diversity(self):
        m = 4
        corr = np.ones((m, m))
        assert _participation_ratio(corr) == pytest.approx(1.0, rel=1e-6)

    def test_partial_correlation_lands_strictly_between(self):
        corr = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
        pr = _participation_ratio(corr)
        assert 1.0 < pr < 3.0

    def test_single_source_is_trivially_diverse_at_one(self):
        assert _participation_ratio(np.eye(1)) == pytest.approx(1.0)


def _write_synthetic_run(tmp_path, owner_true, source_reports_by_round):
    """Build a minimal rounds.jsonl/oracle.jsonl pair by hand, so the
    diagnostic can be tested against known ground truth without running
    a full experiment. `source_reports_by_round[k]` = {source_id: value};
    `owner_true[k]` = true_cost_return for that round."""
    run_dir = tmp_path / "synthetic_run"
    run_dir.mkdir()
    with open(run_dir / "rounds.jsonl", "w", encoding="utf-8") as fh:
        for k, reports in enumerate(source_reports_by_round):
            record = {
                "round_k": k,
                "constraints": {
                    "agent_0": {"reports": [{"source_id": sid, "value": v} for sid, v in reports.items()]}
                },
            }
            fh.write(json.dumps(record) + "\n")
    with open(run_dir / "oracle.jsonl", "w", encoding="utf-8") as fh:
        for k, true_val in enumerate(owner_true):
            record = {"round_k": k, "agents": {"agent_0": {"true_cost_return": true_val}}}
            fh.write(json.dumps(record) + "\n")
    return run_dir


class TestComputeEmpiricalDiversityOnSyntheticLogs:
    def test_two_identical_sources_are_detected_as_non_diverse(self, tmp_path):
        """The P0 #7 self-peer-leakage signature: 'own_critic' and
        'peer_critic_k' report the EXACT SAME value every round (as they
        would if a peer source silently resolved to the owner itself).
        Measured diversity must reflect that, regardless of their
        declared (distinct) independence classes."""
        rng = np.random.default_rng(0)
        true_vals = list(rng.normal(20.0, 3.0, size=30))
        reports = []
        for t in true_vals:
            noisy = t + rng.normal(0, 1.0)
            reports.append({"own_critic": noisy, "peer_critic_1": noisy, "monitor_1": t + rng.normal(0, 1.0)})
        run_dir = _write_synthetic_run(tmp_path, true_vals, reports)

        report = compute_empirical_diversity(
            run_dir, nominal_effective_m=3,
            source_specs={
                "own_critic": ("own_critic", "ic_own"),
                "peer_critic_1": ("peer_critic", "ic_peer_1"),
                "monitor_1": ("monitor", "ic_monitor_1"),
            },
        )
        assert report.nominal_m == 3
        pair_corr = report.cross_source_error_correlation["agent_0"]
        assert pair_corr["own_critic|peer_critic_1"] == pytest.approx(1.0, abs=1e-6)
        # Measured diversity must be well below the nominal 3 distinct
        # classes, since two of the three sources are literally identical.
        assert report.statistical_effective_m_overall < 2.5
        assert report.assumption_1ii_supported is False

    def test_three_genuinely_independent_sources_measure_close_to_nominal(self, tmp_path):
        rng = np.random.default_rng(1)
        true_vals = list(rng.normal(20.0, 3.0, size=200))
        reports = []
        for t in true_vals:
            reports.append(
                {
                    "own_critic": t + rng.normal(0, 1.0),
                    "peer_critic_1": t + rng.normal(0, 1.0),
                    "monitor_1": t + rng.normal(0, 1.0),
                }
            )
        run_dir = _write_synthetic_run(tmp_path, true_vals, reports)
        report = compute_empirical_diversity(run_dir, nominal_effective_m=3, tolerance=0.75)
        assert report.statistical_effective_m_overall > 2.0
        assert report.assumption_1ii_supported is True

    def test_per_source_bias_and_correlation_are_measured_correctly(self, tmp_path):
        """A source with a constant +5 bias must show bias~5, near-zero
        MAE variance contribution, and correlation ~1 with the truth
        (since it tracks truth exactly, just offset)."""
        rng = np.random.default_rng(2)
        true_vals = list(rng.normal(20.0, 5.0, size=50))
        reports = [{"biased_source": t + 5.0, "unbiased_source": t} for t in true_vals]
        run_dir = _write_synthetic_run(tmp_path, true_vals, reports)
        report = compute_empirical_diversity(run_dir)
        by_id = {s.source_id: s for s in report.per_source}
        assert by_id["biased_source"].bias == pytest.approx(5.0, abs=1e-6)
        assert by_id["biased_source"].correlation_with_true == pytest.approx(1.0, abs=1e-6)
        assert by_id["unbiased_source"].bias == pytest.approx(0.0, abs=1e-6)

    def test_low_round_count_warns_and_forces_assumption_unsupported(self, tmp_path):
        """At n=2, |r|=1 for any pair by construction -- the report must
        not let that manufacture a confident 'supported=True' verdict."""
        rng = np.random.default_rng(4)
        true_vals = list(rng.normal(20.0, 3.0, size=2))
        reports = [{"a": t + rng.normal(0, 1), "b": t + rng.normal(0, 1), "c": t + rng.normal(0, 1)} for t in true_vals]
        run_dir = _write_synthetic_run(tmp_path, true_vals, reports)
        report = compute_empirical_diversity(run_dir, nominal_effective_m=3)
        assert report.assumption_1ii_supported is False
        assert any("not reliable" in w for w in report.warnings)

    def test_json_report_round_trips(self, tmp_path):
        rng = np.random.default_rng(3)
        true_vals = list(rng.normal(20.0, 3.0, size=20))
        reports = [{"a": t + rng.normal(0, 1), "b": t + rng.normal(0, 1)} for t in true_vals]
        run_dir = _write_synthetic_run(tmp_path, true_vals, reports)
        report = compute_empirical_diversity(run_dir)
        out = write_report(report, tmp_path / "out.json")
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["nominal_m"] == 2
        assert len(loaded["per_source"]) == 2
        text = format_report(report)
        assert "statistical effective M" in text


class TestIntegrationWithARealRun:
    def test_runs_end_to_end_against_a_real_tiny_experiment(self, tiny_config_factory):
        from safelie.experiment import run_experiment_with_oracle

        cfg = tiny_config_factory(run_id="empirical_diversity_integration")
        out_dir = run_experiment_with_oracle(cfg)
        source_specs = {s.source_id: (s.source_type, s.independence_class) for s in cfg.sources.sources}
        report = compute_empirical_diversity(out_dir, nominal_effective_m=cfg.sources.effective_M, source_specs=source_specs)
        assert report.nominal_m == 3
        assert len(report.per_source) == 3 * 3  # 3 sources x 3 agents (tiny_config_factory's n_agents)
        for s in report.per_source:
            assert s.source_type != "unknown"
        # Only 2 rounds in this tiny config: the low-sample-size warning
        # must fire, and assumption_1ii_supported must not be reported as
        # a positive finding from a sample this small.
        assert report.n_rounds_used == 2
        assert report.assumption_1ii_supported is False
        assert any("not reliable" in w for w in report.warnings)
