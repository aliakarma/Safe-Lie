"""Smoke tests S3 (deterministic seeding) and S14 (checkpoint restore).

Report reference: PROJECT_REPORT.md §R4.1 — "Two runs, same seed/config
gives bitwise-identical logs over 100 iterations" (S3); "Policy, critics,
ensemble, lambda, and RNG states save and restore; a restored run
continues bitwise-identically to an uninterrupted one" (S14).
"""

from __future__ import annotations

from safelie.training.loop import ExperimentRun
from safelie.utils.logging import read_jsonl


def _rounds_are_bitwise_identical(path_a, path_b) -> bool:
    a, b = read_jsonl(path_a), read_jsonl(path_b)
    return a == b


def test_same_seed_same_config_gives_bitwise_identical_logs(tiny_config_factory):
    cfg_a = tiny_config_factory(run_id="s3_run_a", seed=7)
    cfg_b = tiny_config_factory(run_id="s3_run_b", seed=7)

    run_a = ExperimentRun(cfg_a)
    run_a.run()
    run_b = ExperimentRun(cfg_b)
    run_b.run()

    assert _rounds_are_bitwise_identical(
        run_a.output_dir / "rounds.jsonl", run_b.output_dir / "rounds.jsonl"
    )


def test_different_seed_gives_different_logs(tiny_config_factory):
    """Sanity check on the *other* direction: if determinism were broken by
    (e.g.) an unseeded global RNG leaking in, this would also spuriously
    pass -- so we also assert seed=0 differs from seed=1."""
    cfg_a = tiny_config_factory(run_id="s3_seed0", seed=0)
    cfg_b = tiny_config_factory(run_id="s3_seed1", seed=1)

    run_a = ExperimentRun(cfg_a)
    run_a.run()
    run_b = ExperimentRun(cfg_b)
    run_b.run()

    assert not _rounds_are_bitwise_identical(
        run_a.output_dir / "rounds.jsonl", run_b.output_dir / "rounds.jsonl"
    )


def test_checkpoint_restore_continues_bitwise_identically(tiny_config_factory, tmp_path):
    total_rounds = 4  # 4 * rollout_length(16) = 64 steps
    cfg_uninterrupted = tiny_config_factory(run_id="s14_uninterrupted", seed=3)
    cfg_uninterrupted = cfg_uninterrupted.model_copy(update={"total_steps": total_rounds * cfg_uninterrupted.rollout_length})

    run_uninterrupted = ExperimentRun(cfg_uninterrupted)
    for _ in range(total_rounds):
        run_uninterrupted.run_round()
    run_uninterrupted.round_logger.close()
    uninterrupted_records = read_jsonl(run_uninterrupted.output_dir / "rounds.jsonl")

    cfg_resumed = tiny_config_factory(run_id="s14_resumed", seed=3)
    cfg_resumed = cfg_resumed.model_copy(update={"total_steps": total_rounds * cfg_resumed.rollout_length})
    run_resumed = ExperimentRun(cfg_resumed)
    for _ in range(2):  # first half
        run_resumed.run_round()
    ckpt_path = tmp_path / "ckpt.pt"
    run_resumed.checkpoint(ckpt_path)
    run_resumed.round_logger.close()

    # Simulate a fresh process picking the run back up.
    run_resumed_2 = ExperimentRun(cfg_resumed)
    run_resumed_2.restore(ckpt_path)
    # Re-open the logger in append mode at the right position (JsonlLogger
    # already opens in append mode; re-point it to the same file).
    run_resumed_2.round_logger = type(run_resumed.round_logger)(run_resumed.output_dir / "rounds.jsonl")
    for _ in range(2):  # second half
        run_resumed_2.run_round()
    run_resumed_2.round_logger.close()

    resumed_records = read_jsonl(run_resumed.output_dir / "rounds.jsonl")

    assert len(resumed_records) == len(uninterrupted_records) == total_rounds
    assert resumed_records == uninterrupted_records
