"""A2 regression: the RCE calibration clone must be a VALID config.

Three defects lived in `run_clean_calibration` and none of them was
reachable by the existing tests, because those used the pre-G9 source
architecture (own_critic + peers + monitors, no `validation_rounds`, no
`corrupted_source_ids`). Under the frozen G9/G10 architecture that A1 and
A2 actually run, every one of them was fatal:

  1. the clone inherits the parent's `validation_rounds` ([25, 75, 125,
     175, 225]) while being only `calibration_rounds` (20) long, so
     `ExperimentConfig`'s range check rejects it;
  2. the clone sets `attack.f = 0` but keeps `attack.corrupted_source_ids`,
     which the len(ids) == f check rejects;
  3. neither is caught in the parent, because `model_copy` skips
     validators -- they surface inside the source workers, which DO
     re-validate, and a `multiprocessing.Pool` whose initializer raises
     respawns workers forever while the parent blocks. An unbounded hang,
     not a crash.

Together these meant NO RCE run could start under the architecture A1
froze. These tests pin the clone's validity without paying for a run.
"""

from __future__ import annotations

import pytest

from safelie.eval.calibration import calibration_clone
from safelie.utils.config import ExperimentConfig


def _batch_rce_config(**overrides) -> ExperimentConfig:
    """The A2 production shape: parallel trajectory-batch sources, M=3,
    validation rounds beyond the calibration window, RCE defense."""
    base = dict(
        run_id="calib_clone_probe",
        seed=0,
        env={"name": "synthetic_constrained_marl", "n_agents": 3, "budget": 25.0,
             "horizon": 16, "obs_dim": 6, "action_dim": 2},
        topology={"name": "ring", "n_agents": 3},
        sources={"sources": [
            {"source_id": f"batch_{i}", "source_type": "trajectory_batch",
             "independence_class": f"ic_batch{i}"} for i in (1, 2, 3)]},
        source_collection={"mode": "parallel_trajectory_batch", "M": 3, "R_m": 2,
                           "workers": 1, "chunks_per_worker": 1,
                           "validation_rounds": [25, 75, 125, 175, 225], "R_ref": 4},
        attack={"name": "none"},
        defense={"name": "rce", "f": 1, "beta": 1.5, "calibration_rounds": 20},
        ppo={"epochs": 1, "minibatches": 2, "hidden_dim": 8},
        total_steps=250 * 16,
        rollout_length=16,
        output_dir="runs",
    )
    base.update(overrides)
    return ExperimentConfig.model_validate(base)


class TestCalibrationCloneIsValid:
    def test_clean_parent_produces_a_valid_clone(self):
        cfg = _batch_rce_config()
        clone = calibration_clone(cfg)
        # calibration_clone re-validates internally; assert the properties
        # that make it valid, so a regression names the cause.
        assert clone.total_steps // clone.rollout_length == cfg.defense.calibration_rounds
        assert clone.source_collection.validation_rounds == []
        assert clone.attack.name == "none"

    def test_attacked_parent_produces_a_valid_clone(self):
        """Defect 2. attack.f -> 0 must clear corrupted_source_ids with it."""
        cfg = _batch_rce_config(
            attack={"name": "primary", "f": 1, "budget_ratio": 0.5,
                    "direction": "negative", "support": "persistent",
                    "corrupted_source_ids": ["batch_1"]})
        clone = calibration_clone(cfg)
        assert clone.attack.f == 0
        assert clone.attack.corrupted_source_ids is None

    def test_validation_rounds_outside_the_calibration_window_are_dropped(self):
        """Defect 1. The clone is shorter than its parent, so the parent's
        validation rounds are out of ITS range."""
        cfg = _batch_rce_config()
        assert max(cfg.source_collection.validation_rounds) > cfg.defense.calibration_rounds
        assert calibration_clone(cfg).source_collection.validation_rounds == []

    @pytest.mark.parametrize("calibration_rounds", [1, 5, 20])
    def test_clone_survives_worker_side_revalidation(self, calibration_rounds):
        """Defect 3. `safelie.training.source_batch._init_worker` calls
        `ExperimentConfig.model_validate_json` on the serialized config.
        A clone that fails there hangs the pool instead of raising."""
        cfg = _batch_rce_config(
            defense={"name": "rce", "f": 1, "beta": 1.5,
                     "calibration_rounds": calibration_rounds},
            attack={"name": "primary", "f": 1, "budget_ratio": 0.5,
                    "direction": "negative", "support": "persistent",
                    "corrupted_source_ids": ["batch_2"]})
        clone = calibration_clone(cfg)
        ExperimentConfig.model_validate_json(clone.model_dump_json())

    def test_parent_config_is_not_mutated(self):
        cfg = _batch_rce_config(
            attack={"name": "primary", "f": 1, "budget_ratio": 0.5,
                    "direction": "negative", "support": "persistent",
                    "corrupted_source_ids": ["batch_3"]})
        before_rounds = list(cfg.source_collection.validation_rounds)
        before_ids = list(cfg.attack.corrupted_source_ids)
        calibration_clone(cfg)
        assert cfg.source_collection.validation_rounds == before_rounds
        assert cfg.attack.corrupted_source_ids == before_ids
        assert cfg.attack.name == "primary" and cfg.attack.f == 1


class TestRceIsDegenerateAtMEquals3:
    """docs/a2_rce_gates.md section 4, pinned as a test.

    At M=3, f=1 the trimmed set holds one value, so MAD is structurally 0,
    the spread is floored to sigma_min, and the margin is the constant
    beta*sigma_min. A2's entire interpretation rests on this, so it is
    asserted rather than described.
    """

    def test_rce_reduces_to_median_plus_constant(self):
        import numpy as np

        from safelie.defenses.rce import rce_aggregate

        rng = np.random.default_rng(0)
        beta, sigma_min = 1.5, 1e-3
        with pytest.warns(RuntimeWarning):
            rce_aggregate(np.array([1.0, 2.0, 3.0]), 1, beta, sigma_min, 3)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(2000):
                v = rng.normal(25.0, 8.0, size=3)
                r = rce_aggregate(v, 1, beta, sigma_min, 3)
                assert r.retained_n == 1
                assert r.degenerate is True
                assert r.spread == sigma_min
                assert r.applied_margin == pytest.approx(beta * sigma_min, abs=0.0)
                assert r.point_estimate == pytest.approx(float(np.median(v)), abs=0.0)
                assert r.pessimistic_estimate == pytest.approx(
                    float(np.median(v)) + beta * sigma_min, abs=0.0)
