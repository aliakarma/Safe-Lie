"""A3-G1-iii keys duplication on the full trajectory seed pair.

docs/a3_gates.md section 17. A source trajectory is fixed by its
`(env_seed, torch_seed)` pair: `collect_one_trajectory` resets the environment
with the first and the torch generator with the second. The queue used to halt
on the collector's `duplicate_seed_events`, which counts SCALAR repeats -- a
torch seed recurring with a different env seed, in another stream, under
another policy, is not a repeated trajectory, and at M=5 about half of all
seeds contain one by chance.

These tests pin the corrected key, the reporting of all three repeat counts,
and the factual claim section 17 rests on: replayed through the real
collector, seed 0 has one scalar torch repeat and no repeated pair.

Nothing here touches `results/runs_a3` or trains anything.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "scripts" / "a3_run_queue.py"


def _load_queue_module():
    spec = importlib.util.spec_from_file_location("a3_run_queue", QUEUE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The counting itself
# ---------------------------------------------------------------------------


def test_scalar_torch_repeat_with_a_different_env_seed_is_not_a_duplicate(tmp_path):
    q = _load_queue_module()
    p = _write_rows(tmp_path / "source_seeds.jsonl", [
        {"round_k": 0, "seeds": {"batch_1": [[1, 7], [2, 8]]}, "reference_seeds": []},
        {"round_k": 1, "seeds": {"batch_3": [[3, 7]]}, "reference_seeds": []},
    ])
    r = q.seed_repeats(p)
    assert r == {"n_pairs": 3, "pair_repeats": 0, "env_repeats": 0, "torch_repeats": 1}


def test_scalar_env_repeat_with_a_different_torch_seed_is_not_a_duplicate(tmp_path):
    q = _load_queue_module()
    p = _write_rows(tmp_path / "source_seeds.jsonl", [
        {"round_k": 0, "seeds": {"batch_1": [[5, 1]], "batch_2": [[5, 2]]}, "reference_seeds": []},
    ])
    r = q.seed_repeats(p)
    assert (r["pair_repeats"], r["env_repeats"], r["torch_repeats"]) == (0, 1, 0)


def test_a_repeated_full_pair_is_a_duplicate(tmp_path):
    q = _load_queue_module()
    p = _write_rows(tmp_path / "source_seeds.jsonl", [
        {"round_k": 0, "seeds": {"batch_1": [[1, 7]]}, "reference_seeds": []},
        {"round_k": 9, "seeds": {"batch_4": [[1, 7]]}, "reference_seeds": []},
    ])
    r = q.seed_repeats(p)
    assert (r["pair_repeats"], r["env_repeats"], r["torch_repeats"]) == (1, 1, 1)


def test_reference_trajectories_are_inside_the_uniqueness_set(tmp_path):
    """The withheld reference stream issues trajectories too; a replica pair
    reappearing as a reference pair is a duplicate."""
    q = _load_queue_module()
    p = _write_rows(tmp_path / "source_seeds.jsonl", [
        {"round_k": 25, "seeds": {"batch_1": [[11, 12]]}, "reference_seeds": [[11, 12]]},
    ])
    assert q.seed_repeats(p)["pair_repeats"] == 1


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def _synthetic_complete_clean_run(run_dir: Path, seed_rows: list[dict], duplicate_seed_events: int) -> Path:
    """A minimal finished undefended run: enough for `run_gates` on condition A'."""
    run_dir.mkdir(parents=True)
    _write_rows(run_dir / "rounds.jsonl",
                [{"round_k": k, "source_batch": {"env_steps": 5 * 30 * 2000}} for k in range(250)])
    (run_dir / "run_metadata.json").write_text(json.dumps({
        "config_snapshot": {
            "rollout_length": 2000,
            "source_collection": {"M": 5},
            "attack": {"name": "none", "f": 0},
            "defense": {"name": "mean", "f": 0},
        },
        "source_seed_audit": {"duplicate_seed_events": duplicate_seed_events},
    }), encoding="utf-8")
    _write_rows(run_dir / "source_seeds.jsonl", seed_rows)
    return run_dir


def test_gate_passes_a_run_whose_only_repeat_is_a_scalar_torch_seed(tmp_path):
    q = _load_queue_module()
    run = _synthetic_complete_clean_run(tmp_path / "A_seed0", [
        {"round_k": 52, "seeds": {"batch_1": [[1738743935, 1495645152]]}, "reference_seeds": []},
        {"round_k": 209, "seeds": {"batch_3": [[1515124702, 1495645152]]}, "reference_seeds": []},
    ], duplicate_seed_events=1)
    ok, why = q.run_gates("a", 0, run, tmp_path / "log.txt")
    assert ok, why
    # all three repeat counts, and the collector's scalar counter, are reported
    for fragment in ("repeated pairs=0", "repeated env seeds=0", "repeated torch seeds=1",
                     "duplicate_seed_events=1"):
        assert fragment in why, why


def test_gate_fails_a_run_with_a_repeated_trajectory_pair(tmp_path):
    q = _load_queue_module()
    run = _synthetic_complete_clean_run(tmp_path / "A_seed0", [
        {"round_k": 3, "seeds": {"batch_1": [[42, 43]]}, "reference_seeds": []},
        {"round_k": 4, "seeds": {"batch_2": [[42, 43]]}, "reference_seeds": []},
    ], duplicate_seed_events=2)
    ok, why = q.run_gates("a", 0, run, tmp_path / "log.txt")
    assert not ok
    assert why.startswith("A3-G1-iii") and "repeated pairs=1" in why, why


def test_gate_fails_when_the_seed_log_is_missing(tmp_path):
    q = _load_queue_module()
    run = _synthetic_complete_clean_run(tmp_path / "A_seed0", [], duplicate_seed_events=0)
    (run / "source_seeds.jsonl").unlink()
    ok, why = q.run_gates("a", 0, run, tmp_path / "log.txt")
    assert not ok and "source_seeds.jsonl" in why, why


# ---------------------------------------------------------------------------
# Section 17's factual claim, through the real collector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed, expected", [
    (0, {"pair_repeats": 0, "env_repeats": 0, "torch_repeats": 1}),
    (1, {"pair_repeats": 0, "env_repeats": 0, "torch_repeats": 0}),
    (2, {"pair_repeats": 0, "env_repeats": 0, "torch_repeats": 0}),
])
def test_production_seed_draws_replayed_through_the_real_collector(tmp_path, seed, expected):
    from safelie.training.source_batch import ParallelBatchSourceCollector
    from safelie.utils.config import load_experiment_config

    q = _load_queue_module()
    cfg = load_experiment_config(str(ROOT / f"configs/experiment/a3/a_seed{seed}.yaml"))
    sc = cfg.source_collection
    col = ParallelBatchSourceCollector(cfg, [f"agent_{i}" for i in range(cfg.env.n_agents)])
    rows = []
    for k in range(cfg.total_steps // cfg.rollout_length):
        row = {"round_k": k, "seeds": {}, "reference_seeds": []}
        for m, rid in enumerate(col.replica_ids):          # the order `collect` draws in
            row["seeds"][rid] = [list(p) for p in col._draw_seeds(col._replica_rngs[m], sc.R_m)]
        if k in set(sc.validation_rounds):
            row["reference_seeds"] = [list(p) for p in col._draw_seeds(col._reference_rng, sc.R_ref)]
        rows.append(row)
    r = q.seed_repeats(_write_rows(tmp_path / "source_seeds.jsonl", rows))
    assert r["n_pairs"] == 5 * 30 * 250 + 5 * 120
    assert {k: r[k] for k in expected} == expected
    # the collector's scalar counter is what the old gate halted on
    assert col.seed_audit()["duplicate_seed_events"] == expected["env_repeats"] + expected["torch_repeats"]


# ---------------------------------------------------------------------------
# Status bookkeeping names the machine that is actually running the queue
# ---------------------------------------------------------------------------


def test_status_machine_label_is_the_executing_host_not_a_hardcoded_platform(monkeypatch):
    q = _load_queue_module()
    monkeypatch.setenv("SAFELIE_MACHINE_TYPE", "Mac16,10 (Mac mini, Apple M4 4P+6E, 16 GB)")
    assert q.machine_label().startswith("Mac16,10 (Mac mini, Apple M4 4P+6E, 16 GB)")
    monkeypatch.delenv("SAFELIE_MACHINE_TYPE")
    import platform
    assert platform.machine() in q.machine_label()
    assert "gcp t2d-standard-60 (AMD Milan" not in QUEUE.read_text(encoding="utf-8")
