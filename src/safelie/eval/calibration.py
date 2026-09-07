"""Dedicated clean calibration phase for `guarantee_in_force` (P0 #8).

**The bug this replaces.** `safelie.training.loop.ExperimentRun` used to
accumulate `clean_run_disagreements` online, appending one `agg.spread`
sample per round but only `if cfg.attack.name == "none"`. For an attacked
run, `cfg.attack.name` is fixed for the run's entire length, so that list
never received a single sample. `calibrate_epsilon_offline` on an empty
list returns `0.0` by construction (see `safelie.eval.margin`), and
`compute_guarantee_in_force(applied_margin, 0.0)` is `applied_margin >=
0.0` -- true every round, since a margin is never negative. Every attacked
RCE run therefore reported `guarantee_in_force=True` unconditionally,
regardless of whether the empirical margin was actually adequate. This is
exactly the failure mode PROJECT_REPORT.md warns against: a Boolean
certificate that looks like a runtime check but is actually vacuous.

**The fix.** `epsilon_offline` must come from a clean (attack-disabled)
reference distribution that exists independently of whatever the run
being certified is doing -- never from that run's own history. This
module runs a short, dedicated, attack-disabled instance of the same
config (same env/topology/sources/defense/ppo/dual, `attack.name` forced
to `"none"`) for `cfg.defense.calibration_rounds` rounds, and calibrates
`epsilon_offline` from its observed disagreement distribution. That
calibration is computed once per (config structure, seed) before the real
run's first round and never updated online -- it is used identically
whether the real run turns out to be clean or attacked, which is also
what makes `guarantee_in_force` comparable across a pilot matrix's
conditions rather than an artifact of how much clean history each
condition happened to accumulate on its own.

Every component feeding the resulting `epsilon_offline` is recorded in
`RceCalibration` and written to `<run output_dir>/guarantee_calibration.json`
(`write_calibration_report`), so the number is traceable to an actual
measurement rather than merely present.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from safelie.eval.margin import calibrate_epsilon_offline
from safelie.utils.config import ExperimentConfig


@dataclass(frozen=True)
class RceCalibration:
    """Every component of the offline epsilon calibration, kept together
    so `guarantee_in_force` is always traceable to how it was produced."""

    epsilon_offline: float
    n_rounds: int
    alpha: float
    disagreement_mean: float
    disagreement_std: float
    disagreement_min: float
    disagreement_max: float
    source_run_id: str
    source_config_run_id: str

    def to_dict(self) -> dict:
        return asdict(self)


def calibration_clone(cfg: ExperimentConfig) -> ExperimentConfig:
    """The attack-disabled, shortened clone of `cfg` the calibration phase
    actually runs.

    Split out of `run_clean_calibration` so the clone's VALIDITY can be
    tested without paying for a training run: every defect fixed here was
    invisible in the parent process (`model_copy` skips validators) and
    surfaced only inside the source worker pool, hours into a queue.
    """
    n_rounds = cfg.defense.calibration_rounds
    calib_cfg = cfg.model_copy(
        update={
            "run_id": f"calib_{cfg.run_id}",
            # A2. `corrupted_source_ids` must be cleared alongside `f`, not
            # left behind: `ExperimentConfig` requires len(ids) == attack.f,
            # so an f=0 clone still naming one source is an invalid config.
            # `model_copy` does not re-run validators, so this stayed
            # invisible in the parent process and surfaced only inside the
            # source workers, which DO re-validate (`_init_worker` calls
            # `model_validate_json`). Every attacked RCE run therefore died
            # in its calibration phase.
            "attack": cfg.attack.model_copy(
                update={"name": "none", "f": 0, "corrupted_source_ids": None}
            ),
            "total_steps": n_rounds * cfg.rollout_length,
            # A2. The clone is SHORTER than the run it calibrates, so the
            # parent's `validation_rounds` are generally outside its own
            # 0..n_rounds-1 range and `ExperimentConfig`'s cross-field
            # validator rejects the clone outright -- which made every RCE
            # run under the G9/G10 architecture fail in
            # `ExperimentRun.__init__` before round 0 (the frozen configs
            # use validation_rounds [25, 75, 125, 175, 225] against
            # calibration_rounds=20). The withheld R_ref reference exists to
            # separate source-implementation failure from policy-control
            # failure in a RUN; the calibration phase is neither -- it reads
            # only `agg.spread` -- so it needs no reference at all. Emptying
            # the list is what the clone actually wants, and it also stops
            # the phase from paying for R_ref=120 collections it discards.
            "source_collection": cfg.source_collection.model_copy(
                update={"validation_rounds": []}
            ),
        }
    )

    # A2. `model_copy` does NOT re-run validators, but the source workers
    # DO (`safelie.training.source_batch._init_worker` calls
    # `model_validate_json` on the serialized config). An invalid clone
    # therefore does not raise here -- it raises inside every worker, and a
    # `multiprocessing.Pool` whose initializer raises respawns workers
    # forever while the parent blocks on the first chunk. That is an
    # unbounded hang, not a crash, which is the worst possible failure mode
    # for a multi-day run queue. Re-validate in the parent so a bad clone is
    # an immediate, readable error instead.
    return ExperimentConfig.model_validate_json(calib_cfg.model_dump_json())


def run_clean_calibration(cfg: ExperimentConfig) -> RceCalibration:
    """Run a dedicated, attack-disabled clone of `cfg` and calibrate
    `epsilon_offline` from its own observed source-disagreement (MAD)
    distribution. See the module docstring for why this must never reuse
    the disagreement history of the run it will be attached to.
    """
    from safelie.training.loop import (
        ExperimentRun,  # local: eval -> training is fine; avoids a load-time cycle
    )

    n_rounds = cfg.defense.calibration_rounds
    alpha = cfg.defense.calibration_alpha
    calib_cfg = calibration_clone(cfg)

    run = ExperimentRun(calib_cfg, _skip_calibration=True)
    try:
        for _ in range(n_rounds):
            run.run_round()
    finally:
        # A2. Release the calibration run's worker pool before the real run
        # starts. Under the G9 parallel trajectory-batch architecture this
        # phase opens `source_collection.workers` (12) OS processes, each
        # holding its own MuJoCo environment and policy copy; only
        # `round_logger` was being closed, so those 12 stayed resident for
        # the whole ~10 h run that follows and then spawned 12 more. This
        # is resource cleanup only: `ExperimentRun.close` touches the pool
        # and the JSONL loggers and nothing else, and in particular does
        # not touch `clean_run_disagreements`, which is read below. No
        # calibrated number changes (docs/a2_rce_gates.md A2-G1-v).
        run.round_logger.close()
        run.close()

    disagreements = np.asarray(run.clean_run_disagreements, dtype=float)
    if len(disagreements) < n_rounds:
        raise RuntimeError(
            f"Clean calibration phase for {cfg.run_id!r} produced "
            f"{len(disagreements)} disagreement samples after {n_rounds} rounds "
            f"(expected {n_rounds}, one per round: every round of a "
            f"attack.name=='none' run appends one spread value in "
            f"ExperimentRun.run_round). Refusing to silently produce a partial "
            f"or empty calibration rather than a real epsilon_offline=0.0 in "
            f"disguise."
        )
    epsilon_offline = calibrate_epsilon_offline(list(disagreements), alpha=alpha)

    return RceCalibration(
        epsilon_offline=epsilon_offline,
        n_rounds=n_rounds,
        alpha=alpha,
        disagreement_mean=float(disagreements.mean()),
        disagreement_std=float(disagreements.std()),
        disagreement_min=float(disagreements.min()),
        disagreement_max=float(disagreements.max()),
        source_run_id=calib_cfg.run_id,
        source_config_run_id=cfg.run_id,
    )


def write_calibration_report(calibration: RceCalibration, output_dir: str | Path) -> Path:
    """Persist every component of the calibration next to the run it
    certifies, so `guarantee_in_force` is auditable without re-deriving it."""
    path = Path(output_dir) / "guarantee_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration.to_dict(), indent=2), encoding="utf-8")
    return path
