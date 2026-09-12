"""`workers` is a compute-only knob; the rest of `source_collection` is frozen.

docs/a3_gates.md sections 15 and 16. A3's platform moved to a 60-vCPU GCP
instance, which meant raising `workers` from 12 -- and
`scripts/a3_verify_frozen.py` would have rejected that, halting the queue
before any run started, because it inherited A2's rule that NOTHING in
`source_collection` may move except `M` and `seed_entropy`.

Section 16 then moved the production value from 30 to **20**, so that all
three seeds run concurrently (3 x 20 = 60 vCPU) rather than two at 30 with
seed 2 waiting. That is a scheduling change; the invariance argument below is
what makes it free of scientific consequence.

The justification for relaxing it, recorded here as well as in the script:

    Worker count is scientifically invariant because trajectory seeds are
    generated in the parent process and each worker is a pure function of the
    same (policy, env_seed, torch_seed) tuple.

That is a structural property of `safelie.training.source_batch`, and it was
also MEASURED at A3's own operating point (M=5, f=1) on 2026-09-09, and
re-measured on 2026-09-12 to cover the production value 20 and its neighbour
30 -- neither of which the original sweep had reached:

    workers in {1, 2, 4, 5, 6, 8, 12, 20, 30}; both dispatch paths (workers=1
    bypasses mp.Pool and runs in-process, workers>=2 does not); chunk
    partitions from 40 to 120; at R_m=8 and at production R_m=30 -- all 30
    source means (5 sources x 6 owners) bitwise identical, and every
    per-trajectory value identical, max abs difference exactly 0.0.

`tests/unit/test_source_batch.py` pins the invariance itself. This file pins
the *policy*: that the verifier permits the knob to move, and still refuses
every neighbouring field that is NOT a compute knob.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "a3_verify_frozen", ROOT / "scripts" / "a3_verify_frozen.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


VERIFIER = _load_verifier()


# ---------------------------------------------------------------------------
# The permission policy itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workers", [1, 2, 5, 12, 20, 30])
def test_workers_is_permitted_as_a_compute_variant(workers):
    """Every worker count A3 might plausibly run at is a legal compute variant.

    1 is the pool-bypass path, 12 was the laptop/Mac value, 20 is the GCP
    production value (three concurrent seeds x 20 on a 60-vCPU instance), 30
    was the previous two-concurrent-seeds value, and 2/5 are intermediate
    counts with different chunk partitions.
    """
    assert VERIFIER.illegal_keys(["source_collection.workers"]) == []
    # and it stays legal alongside the other declared A3 changes
    differing = ["run_id", "output_dir", "attack.name", "defense.name",
                 "sources.sources", "source_collection.M",
                 "source_collection.seed_entropy", "source_collection.workers"]
    assert VERIFIER.illegal_keys(differing) == []


@pytest.mark.parametrize(
    "field",
    ["R_m", "chunks_per_worker", "validation_rounds", "R_ref", "mode"],
)
def test_the_rest_of_source_collection_is_still_frozen(field):
    """Relaxing `workers` must not have relaxed its neighbours.

    R_m and R_ref change the estimator's variance, validation_rounds changes
    when the withheld reference is collected, and mode changes the source
    architecture outright. None is a compute knob, and a change to any of them
    must still halt the queue.
    """
    assert VERIFIER.illegal_keys([f"source_collection.{field}"]) == [
        f"source_collection.{field}"
    ]


def test_permitted_set_is_exactly_the_three_declared_keys():
    assert VERIFIER.PERMITTED_SOURCE_COLLECTION == {"M", "seed_entropy", "workers"}


def test_workers_is_not_permitted_to_differ_within_a_seed():
    """A whole seed must run on one machine, so its four conditions must agree
    on the worker count too. `workers` is a compute knob ACROSS campaigns, not
    a free parameter WITHIN a paired seed -- CRN pairing is the reason the
    within-seed rule is stricter than the vs-A2 rule."""
    assert "source_collection" not in VERIFIER.PERMITTED_WITHIN_SEED
    assert VERIFIER.PERMITTED_WITHIN_SEED == {"run_id", "attack", "defense"}


# ---------------------------------------------------------------------------
# End to end against the real twelve production configs
# ---------------------------------------------------------------------------


def test_the_twelve_production_configs_are_all_at_workers_20():
    sys.path.insert(0, str(ROOT / "src"))
    from safelie.utils.config import load_experiment_config

    for cond in ("a", "b", "c", "e"):
        for seed in (0, 1, 2):
            cfg = load_experiment_config(
                str(ROOT / f"configs/experiment/a3/{cond}_seed{seed}.yaml")
            )
            sc = cfg.source_collection
            assert sc is not None
            assert sc.workers == 20, f"{cond}_seed{seed} is at workers={sc.workers}"
            # the scientific fields are unmoved
            assert (sc.M, sc.R_m, sc.R_ref, sc.chunks_per_worker) == (5, 30, 120, 4)
            assert cfg.rollout_length == 2000
            assert cfg.total_steps == 500_000


def test_verifier_passes_end_to_end_at_workers_20():
    """The whole script, on the real configs, exits 0."""
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/a3_verify_frozen.py")],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ALL CONFIGS PASS" in r.stdout
