"""P0 #8 regression: `guarantee_in_force` must not be vacuous under attack.

Bug fixed: `ExperimentRun` accumulated `clean_run_disagreements` only when
`cfg.attack.name == "none"`. An attacked run has `attack.name != "none"`
for its entire length, so that list never received a sample;
`calibrate_epsilon_offline([])` returns 0.0, and
`compute_guarantee_in_force(applied_margin, 0.0)` is `applied_margin >=
0.0` -- true every round, since a margin can't be negative. Every
attacked RCE run therefore reported `guarantee_in_force=True`
unconditionally. The fix (`safelie.eval.calibration`) runs a dedicated,
attack-disabled calibration phase before round 0 and uses ITS
`epsilon_offline` for the entire run, attacked or not.
"""

from __future__ import annotations

import json
import math

import pytest

from safelie.eval.calibration import run_clean_calibration
from safelie.eval.margin import compute_guarantee_in_force
from safelie.training.loop import ExperimentRun
from safelie.utils.config import ExperimentConfig
from safelie.utils.logging import read_jsonl


def _rce_config(tmp_path, run_id: str, attack_name: str = "none", seed: int = 0, calibration_rounds: int = 5) -> ExperimentConfig:
    return ExperimentConfig(
        run_id=run_id,
        seed=seed,
        env={"name": "synthetic_constrained_marl", "n_agents": 3, "budget": 25.0, "horizon": 16, "obs_dim": 6, "action_dim": 2},
        topology={"name": "ring", "n_agents": 3},
        sources={
            "sources": [
                {"source_id": "own_critic", "source_type": "own_critic", "independence_class": "ic_own"},
                {"source_id": "peer_critic_1", "source_type": "peer_critic", "independence_class": "ic_peer1"},
                {"source_id": "peer_critic_2", "source_type": "peer_critic", "independence_class": "ic_peer2"},
                {"source_id": "monitor_1", "source_type": "monitor", "independence_class": "ic_mon1"},
                {"source_id": "monitor_2", "source_type": "monitor", "independence_class": "ic_mon2"},
            ]
        },
        attack={"name": attack_name, "f": 1, "budget_ratio": 0.5} if attack_name != "none" else {"name": "none"},
        defense={"name": "rce", "f": 1, "beta": 1.5, "calibration_rounds": calibration_rounds},
        ppo={"epochs": 1, "minibatches": 2, "hidden_dim": 8},
        total_steps=32,
        rollout_length=16,
        output_dir=str(tmp_path / "runs"),
    )


class TestVacuousGuaranteeRegression:
    """The exact bug: an attacked run's epsilon_offline must never be the
    silent 0.0 default, and guarantee_in_force must not be trivially True
    every round purely because it was never given a real threshold."""

    def test_attacked_run_epsilon_offline_is_never_the_vacuous_zero_default(self, tmp_path):
        cfg = _rce_config(tmp_path, "guarantee_attacked", attack_name="primary")
        run = ExperimentRun(cfg)
        for _ in range(2):
            record = run.run_round()
            for c in record["constraints"].values():
                assert c["epsilon_offline"] is not None
                assert c["epsilon_offline"] > 0.0, (
                    "epsilon_offline is exactly the vacuous 0.0 default this fix removes -- "
                    "an attacked run must calibrate from a dedicated clean phase, not its "
                    "own (empty) attack-disabled history"
                )

    def test_compute_guarantee_in_force_is_not_trivially_true_at_a_realistic_epsilon(self):
        """Sanity check on the primitive itself: at epsilon_offline=0.0
        (the bug), any non-negative margin passes. At a realistic
        (nonzero) epsilon, an inadequate margin must fail."""
        assert compute_guarantee_in_force(applied_margin=0.01, epsilon_offline=0.0) is True  # the bug, isolated
        assert compute_guarantee_in_force(applied_margin=0.01, epsilon_offline=0.5) is False  # the fix's effect
        assert compute_guarantee_in_force(applied_margin=1.0, epsilon_offline=0.5) is True

    def test_attacked_and_clean_runs_of_the_same_structure_get_the_same_calibration_recipe(self, tmp_path):
        """Not bit-identical epsilon values (each run's calibration phase
        is independently rolled out) but the same n_rounds/alpha recipe,
        so guarantee_in_force is comparable across a matrix's conditions
        rather than an artifact of how much clean history one condition
        happened to accumulate online."""
        clean_cfg = _rce_config(tmp_path, "guarantee_clean", attack_name="none")
        attacked_cfg = _rce_config(tmp_path, "guarantee_attacked2", attack_name="primary")
        clean_run = ExperimentRun(clean_cfg)
        attacked_run = ExperimentRun(attacked_cfg)
        assert clean_run.calibration is not None
        assert attacked_run.calibration is not None
        assert clean_run.calibration.n_rounds == attacked_run.calibration.n_rounds == 5
        assert clean_run.calibration.alpha == attacked_run.calibration.alpha


class TestCalibrationTraceability:
    """'log the components used to compute it' -- the task's explicit
    requirement. A Boolean must never stand alone."""

    def test_calibration_report_is_written_and_matches_the_run(self, tmp_path):
        cfg = _rce_config(tmp_path, "guarantee_traceable", attack_name="primary")
        run = ExperimentRun(cfg)
        report_path = run.output_dir / "guarantee_calibration.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["epsilon_offline"] == run.calibration.epsilon_offline
        assert report["n_rounds"] == 5
        assert report["source_config_run_id"] == "guarantee_traceable"
        assert report["source_run_id"] == "calib_guarantee_traceable"
        assert "disagreement_mean" in report and "disagreement_std" in report

    def test_calibration_phase_leaves_its_own_inspectable_artifact(self, tmp_path):
        """The calibration is a real rollout, not a black box: its own
        rounds.jsonl exists and shows attack.name=='none' behaviour
        (no corrupted sources in any round)."""
        cfg = _rce_config(tmp_path, "guarantee_inspectable", attack_name="primary")
        run = ExperimentRun(cfg)
        calib_dir = tmp_path / "runs" / "calib_guarantee_inspectable"
        assert calib_dir.exists()
        calib_rounds = read_jsonl(calib_dir / "rounds.jsonl")
        assert len(calib_rounds) == run.calibration.n_rounds
        for rec in calib_rounds:
            for c in rec["constraints"].values():
                assert c["corrupted_source_ids"] == []

    def test_round_records_log_epsilon_offline_every_round(self, tmp_path):
        cfg = _rce_config(tmp_path, "guarantee_logged", attack_name="primary")
        run = ExperimentRun(cfg)
        record = run.run_round()
        for c in record["constraints"].values():
            assert math.isclose(c["epsilon_offline"], run.calibration.epsilon_offline)


class TestCalibrationDoesNotBreakDeterminismOrRecurse:
    def test_same_seed_rce_runs_are_still_bitwise_identical(self, tmp_path):
        """The calibration phase runs (and uses RNG) BEFORE
        seed_everything(cfg.seed) inside __init__, specifically so it
        cannot perturb the main run's own determinism guarantee.

        Matches tests/smoke/test_determinism.py's own pattern: construct
        AND fully run run_a before even constructing run_b. Both runs
        read GaussianPolicy.act's samples from torch's *global* default
        generator (not an explicit per-object Generator), which
        seed_everything resets deterministically but which is a single
        process-wide resource -- constructing run_b before run_a has
        finished consuming that shared generator would interleave their
        draws and break bitwise equality for reasons unrelated to this
        fix (this is exactly why the pre-existing S3 test never
        interleaves construction and running across two runs either).
        """
        cfg_a = _rce_config(tmp_path, "guarantee_det_a", attack_name="primary", seed=7)
        cfg_b = _rce_config(tmp_path, "guarantee_det_b", attack_name="primary", seed=7)
        run_a = ExperimentRun(cfg_a)
        run_a.run()
        run_b = ExperimentRun(cfg_b)
        run_b.run()
        assert read_jsonl(run_a.output_dir / "rounds.jsonl") == read_jsonl(run_b.output_dir / "rounds.jsonl")

    def test_calibration_sub_run_does_not_recursively_calibrate_itself(self, tmp_path):
        """_skip_calibration must break the recursion: the nested
        ExperimentRun that run_clean_calibration constructs also has
        defense.name=='rce' (inherited), which would otherwise try to
        calibrate itself, then that one would too, forever."""
        cfg = _rce_config(tmp_path, "guarantee_no_recursion", attack_name="none", calibration_rounds=3)
        calibration = run_clean_calibration(cfg)
        assert calibration.n_rounds == 3
        assert calibration.epsilon_offline >= 0.0

    def test_non_rce_defense_never_triggers_calibration(self, tmp_path):
        cfg = _rce_config(tmp_path, "guarantee_not_rce", attack_name="none")
        cfg = cfg.model_copy(update={"defense": cfg.defense.model_copy(update={"name": "mean"})})
        run = ExperimentRun(cfg)
        assert run.calibration is None
        record = run.run_round()
        for c in record["constraints"].values():
            assert c["guarantee_in_force"] is None
            assert c["epsilon_offline"] is None
        assert not (run.output_dir / "guarantee_calibration.json").exists()


class TestConfigDefaults:
    def test_calibration_rounds_and_alpha_have_sane_defaults(self):
        from safelie.utils.config import DefenseConfig

        d = DefenseConfig(name="rce", f=1)
        assert d.calibration_rounds >= 1
        assert 0.0 < d.calibration_alpha < 1.0

    def test_calibration_rounds_must_be_positive(self):
        from pydantic import ValidationError

        from safelie.utils.config import DefenseConfig

        with pytest.raises(ValidationError):
            DefenseConfig(name="rce", f=1, calibration_rounds=0)
