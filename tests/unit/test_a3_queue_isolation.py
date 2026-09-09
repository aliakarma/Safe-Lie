"""Two A3 queue processes, one per seed, must not touch each other.

docs/a3_gates.md section 15. The GCP layout runs seed 0 and seed 1 side by
side on one 60-vCPU instance. Before that change the queue had one hardcoded
`a3_queue_status.json` and `write_status` rewrote the whole document, so two
processes would have silently lost each other's updates -- and the status file
is what records gate outcomes and the halt reason, so a lost update destroys
the campaign's audit trail rather than merely confusing the operator.

Everything here is scheduling and bookkeeping. No test in this file trains
anything: they use `--dry-run`, which walks the real queue, writes the real
status files, and starts no run. The scientific run order within a seed
(A' -> B' -> C' -> E') is asserted, not changed.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "scripts" / "a3_run_queue.py"
OUT = ROOT / "results" / "runs_a3"


def _load_queue_module():
    """Import the queue script as a module, for the tests that exercise one of
    its functions directly rather than through a subprocess."""
    spec = importlib.util.spec_from_file_location("a3_run_queue", QUEUE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def run_queue(*args: str, expect_ok: bool = True) -> subprocess.CompletedProcess:
    """Run the queue and, by default, REQUIRE it to exit 0.

    Checking the return code here rather than at each call site is deliberate.
    An earlier version left it to the caller, and two tests did not check --
    which let a crashed queue process pass as a success and hid an
    intermittent `os.replace` failure that only showed up as a stray `.tmp`
    file several tests later. A queue that dies is never an acceptable
    outcome except where a test explicitly asks for one (`expect_ok=False`).
    """
    r = subprocess.run(
        [sys.executable, str(QUEUE), *args],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300,
    )
    if expect_ok:
        assert r.returncode == 0, (
            f"queue exited {r.returncode} for args {args}\n"
            f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
        )
    return r


@pytest.fixture
def clean_status():
    """Remove per-seed status/lock files before and after, leaving any real
    campaign status file alone."""
    def _paths():
        return [OUT / f"a3_queue_status_seed{s}.json" for s in (0, 1, 2)] + \
               [OUT / f".a3_queue_seed{s}.lock" for s in (0, 1, 2)] + \
               [OUT / ".a3_queue.lock"] + list(OUT.glob("*.tmp"))

    for p in _paths():
        p.unlink(missing_ok=True)
    yield
    for p in _paths():
        p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Seed filtering and status isolation
# ---------------------------------------------------------------------------


def test_seed_filter_selects_only_that_seeds_four_runs(clean_status):
    for seed in (0, 1, 2):
        r = run_queue("--seed", str(seed), "--dry-run")
        assert r.returncode == 0, r.stdout + r.stderr
        state = json.loads((OUT / f"a3_queue_status_seed{seed}.json").read_text())
        assert state["seed_scope"] == seed
        assert state["queue"] == [f"{c}_seed{seed}" for c in ("A", "B", "C", "E")]
        # no run from another seed was selected
        assert all(f"seed{seed}" in name for name in state["runs"])


def test_run_order_within_a_seed_is_unchanged(clean_status):
    """A' -> B' -> C' -> E'. The undefended references must come first: the
    mechanism validator gates C' against B' and E' against A'."""
    r = run_queue("--seed", "0", "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    state = json.loads((OUT / "a3_queue_status_seed0.json").read_text())
    assert state["queue"] == ["A_seed0", "B_seed0", "C_seed0", "E_seed0"]


def test_each_seed_writes_only_its_own_status_file(clean_status):
    run_queue("--seed", "0", "--dry-run")
    assert (OUT / "a3_queue_status_seed0.json").exists()
    assert not (OUT / "a3_queue_status_seed1.json").exists()
    assert not (OUT / "a3_queue_status_seed2.json").exists()

    run_queue("--seed", "1", "--dry-run")
    assert (OUT / "a3_queue_status_seed1.json").exists()
    s0 = json.loads((OUT / "a3_queue_status_seed0.json").read_text())
    assert s0["seed_scope"] == 0          # seed 1's run did not rewrite seed 0's file
    assert s0["queue"] == ["A_seed0", "B_seed0", "C_seed0", "E_seed0"]


def test_two_seeds_concurrently_do_not_corrupt_either_status_file(clean_status):
    """The actual GCP layout: two queue processes at once."""
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(run_queue, "--seed", str(s), "--dry-run") for s in (0, 1)]
        results = [f.result() for f in futures]
    for r in results:
        assert r.returncode == 0, r.stdout + r.stderr

    for seed in (0, 1):
        raw = (OUT / f"a3_queue_status_seed{seed}.json").read_text()
        state = json.loads(raw)          # parses => not truncated or interleaved
        assert state["seed_scope"] == seed
        assert state["queue"] == [f"{c}_seed{seed}" for c in ("A", "B", "C", "E")]
        assert state["halted"] is None


def test_no_temp_files_are_left_behind(clean_status):
    run_queue("--seed", "0", "--dry-run")
    assert not list(OUT.glob("*.tmp")), "atomic-rewrite temp file was not replaced"


def test_write_status_survives_a_transient_replace_failure(tmp_path, monkeypatch):
    """A bookkeeping write must not be able to end a multi-day campaign.

    On Windows `os.replace` raises PermissionError if anything holds a
    momentary handle to either path -- a virus scanner or the search indexer
    is enough, and it happens under load. Unhandled, that kills the queue
    process and halts a seed over a status write. This is the regression: two
    failures then success, no exception, no stray temp file.
    """
    queue = _load_queue_module()
    monkeypatch.setattr(queue, "OUT_ROOT", tmp_path)
    target = tmp_path / "a3_queue_status_seed0.json"

    real_replace = queue.os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("simulated transient sharing violation")
        return real_replace(src, dst)

    monkeypatch.setattr(queue.os, "replace", flaky)
    monkeypatch.setattr(queue.time, "sleep", lambda _s: None)   # no real delay

    queue.write_status({"seed_scope": 0, "runs": {}}, target)

    assert calls["n"] == 3                       # two failures, then success
    assert json.loads(target.read_text())["seed_scope"] == 0
    assert not list(tmp_path.glob("*.tmp"))


def test_write_status_cleans_up_when_replace_never_succeeds(tmp_path, monkeypatch):
    """If it genuinely cannot write, the error still propagates -- silently
    losing the audit trail would be worse than failing -- but no temp file is
    left behind for a later reader to mistake for status."""
    queue = _load_queue_module()
    monkeypatch.setattr(queue, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(queue.time, "sleep", lambda _s: None)

    def always_fails(src, dst):
        raise PermissionError("permanent")

    monkeypatch.setattr(queue.os, "replace", always_fails)

    with pytest.raises(PermissionError):
        queue.write_status({"seed_scope": 0}, tmp_path / "a3_queue_status_seed0.json")
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------


def test_a_second_process_on_the_same_seed_is_refused(clean_status):
    """Per-seed status files stop two processes clobbering each other's
    bookkeeping; they do NOT stop two processes running the same seed and
    racing on its run directories and checkpoints. The lock does."""
    lock = OUT / ".a3_queue_seed0.lock"
    OUT.mkdir(parents=True, exist_ok=True)
    lock.write_text("pid=999999 started=test", encoding="utf-8")
    try:
        r = run_queue("--seed", "0", "--dry-run", expect_ok=False)
        assert r.returncode != 0
        assert "already exists" in (r.stdout + r.stderr)
        assert "pid=999999" in (r.stdout + r.stderr)
    finally:
        lock.unlink(missing_ok=True)


def test_different_seeds_take_different_locks(clean_status):
    lock0 = OUT / ".a3_queue_seed0.lock"
    OUT.mkdir(parents=True, exist_ok=True)
    lock0.write_text("pid=999999 started=test", encoding="utf-8")
    try:
        r = run_queue("--seed", "1", "--dry-run")   # seed 1 is unaffected
        assert r.returncode == 0, r.stdout + r.stderr
    finally:
        lock0.unlink(missing_ok=True)


def test_lock_is_released_on_normal_exit(clean_status):
    run_queue("--seed", "0", "--dry-run")
    assert not (OUT / ".a3_queue_seed0.lock").exists()
    r = run_queue("--seed", "0", "--dry-run")       # so a rerun is possible
    assert r.returncode == 0, r.stdout + r.stderr


# ---------------------------------------------------------------------------
# Independent stop / resume
# ---------------------------------------------------------------------------


def test_halting_one_seed_leaves_the_other_intact(clean_status):
    """Simulate seed 0 halted mid-campaign, then run seed 1. Seed 0's recorded
    halt state must survive untouched -- an operator has to be able to see WHY
    a seed stopped after the others have moved on."""
    run_queue("--seed", "0", "--dry-run")
    p0 = OUT / "a3_queue_status_seed0.json"
    halted = json.loads(p0.read_text())
    halted["halted"] = "A_seed0: simulated stop condition"
    halted["runs"] = {"A_seed0": {"status": "gate-failed", "gates_pass": False}}
    p0.write_text(json.dumps(halted, indent=2), encoding="utf-8")

    r = run_queue("--seed", "1", "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr

    after = json.loads(p0.read_text())
    assert after["halted"] == "A_seed0: simulated stop condition"
    assert after["runs"]["A_seed0"]["gates_pass"] is False


def test_each_seed_resumes_from_its_own_status_file(clean_status):
    """Restarting a seed reloads that seed's state and clears only its halt."""
    run_queue("--seed", "0", "--dry-run")
    p0 = OUT / "a3_queue_status_seed0.json"
    state = json.loads(p0.read_text())
    state["halted"] = "simulated"
    state["runs"]["_marker"] = {"kept": True}
    p0.write_text(json.dumps(state, indent=2), encoding="utf-8")

    r = run_queue("--seed", "0", "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    resumed = json.loads(p0.read_text())
    assert resumed["halted"] is None                  # cleared on restart
    assert "restarted" in resumed                     # and recorded as a restart
    assert resumed["runs"]["_marker"] == {"kept": True}   # prior state preserved


# ---------------------------------------------------------------------------
# The whole-campaign invocation still works and stays separate
# ---------------------------------------------------------------------------


def test_unfiltered_queue_still_covers_all_twelve_and_uses_the_original_file(clean_status):
    original = OUT / "a3_queue_status.json"
    backup = original.read_text(encoding="utf-8") if original.exists() else None
    try:
        r = run_queue("--dry-run")
        assert r.returncode == 0, r.stdout + r.stderr
        state = json.loads(original.read_text())
        assert state["seed_scope"] is None
        assert len(state["queue"]) == 12
        assert state["queue"][:4] == ["A_seed0", "B_seed0", "C_seed0", "E_seed0"]
    finally:
        if backup is not None:
            original.write_text(backup, encoding="utf-8")
        else:
            original.unlink(missing_ok=True)


def test_dry_run_creates_no_production_results(clean_status):
    before = {p.name for p in OUT.iterdir()} if OUT.exists() else set()
    run_queue("--seed", "2", "--dry-run")
    after = {p.name for p in OUT.iterdir()}
    new = after - before
    # only bookkeeping may appear -- never a run directory
    assert all(n.startswith("a3_queue_status") or n.startswith(".a3_queue") for n in new), new
    for cond in ("A", "B", "C", "E"):
        assert not (OUT / f"{cond}_seed2" / "rounds.jsonl").exists()


def test_smoke_configs_are_unreachable_from_the_queue():
    """The queue builds config paths as `{cond}_seed{seed}.yaml`, so a
    `_smoke_*.yaml` in the same directory can never be selected."""
    src = QUEUE.read_text(encoding="utf-8")
    assert 'configs/experiment/a3/{cond}_seed{seed}.yaml' in src
    assert "_smoke" not in src
    smoke = sorted(p.name for p in (ROOT / "configs/experiment/a3").glob("_smoke*.yaml"))
    assert smoke, "expected the smoke configs to exist alongside the production ones"
    for cond in ("a", "b", "c", "e"):
        for seed in (0, 1, 2):
            assert f"{cond}_seed{seed}.yaml" not in smoke
