#!/usr/bin/env python
"""A3 production run queue -- gated and resumable. One machine, all twelve runs.

Runs the twelve A3 runs and enforces docs/a3_gates.md section 7 after each,
so a stop condition halts the queue instead of spending another day and a
half producing artifacts already known to be invalid.

**All twelve on one machine (BRANCH-B', gates doc section 12.4, platform
amended in section 15).** The cross-platform probe showed two different
architectures do not agree bitwise -- torch initialises different weights on
arm64 -- so every contrast must live on one machine. The production platform
is one GCP `t2d-standard-60` (AMD Milan, x86-64); with every run on one
instance there is no machine factor to bound.

**Order: complete seeds, A' B' C' E'.** Every contrast A3 reports is a
within-seed difference, so finishing whole seeds makes every intermediate
stopping point a complete, analysable design at 1, 2 or 3 seeds instead of
three partial ones. Within a seed the undefended references come first
because the mechanism validator gates C' against B' and E' against A'.

**Sequential WITHIN a seed, optionally concurrent ACROSS seeds.** The four
conditions of a seed must stay in order and on one machine -- the mechanism
validator gates C' against B' and E' against A', and the CRN pairing is what
makes the contrasts paired. Different seeds share nothing but the code, so
they may run side by side when a machine has the cores for it.

`--seed N` restricts this process to one seed and gives it its own status
file, `a3_queue_status_seed<N>.json`. That is the whole of the concurrency
support: scheduling and bookkeeping only, no change to what any run does.
A lock file makes two processes claiming the same seed an immediate,
explicit failure rather than a silent race on the status file.

Usage:
    python scripts/a3_run_queue.py                    # all 12, sequential
    python scripts/a3_run_queue.py --seed 0           # just seed 0's four runs
    python scripts/a3_run_queue.py --seed 0 --only C_seed0
    python scripts/a3_run_queue.py --seed 0 --dry-run

    # GCP t2d-standard-60: all three seeds concurrently, 20 workers each
    python scripts/a3_run_queue.py --seed 0 &
    python scripts/a3_run_queue.py --seed 1 &
    python scripts/a3_run_queue.py --seed 2 &
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT_ROOT = ROOT / "results" / "runs_a3"


def status_path(seed: int | None) -> Path:
    """One status file per queue process.

    A whole-campaign invocation keeps the original `a3_queue_status.json`;
    `--seed N` writes `a3_queue_status_seed<N>.json`. Two processes therefore
    never share a status file, which is what makes concurrent seeds safe:
    `write_status` rewrites the whole document, so a shared file would lose
    updates and could be read mid-write as truncated JSON. The status file
    records gate outcomes and the halt reason, so corrupting it would destroy
    the campaign's audit trail rather than merely inconvenience the operator.
    """
    return OUT_ROOT / (f"a3_queue_status_seed{seed}.json" if seed is not None
                       else "a3_queue_status.json")


def lock_path(seed: int | None) -> Path:
    return OUT_ROOT / (f".a3_queue_seed{seed}.lock" if seed is not None
                       else ".a3_queue.lock")


ATTACKED = {0: "batch_1", 1: "batch_3", 2: "batch_5"}       # gates doc section 5
CONDS = ("a", "b", "c", "e")
LABEL = {"a": "A", "b": "B", "c": "C", "e": "E"}
# Complete seeds, undefended references first -- see the module docstring.
QUEUE = [(c, s) for s in (0, 1, 2) for c in CONDS]

EXPECTED_ROUNDS = 250
EXPECTED_PPO_STEPS = 500_000
EXPECTED_SOURCE_STEPS = 5 * 30 * 2000 * 250      # M=5: 75,000,000
EXPECTED_ENV_SEEDS = (5 + 1) * 30 * 250 + 0      # replicas only; reference adds R_ref draws
SIGMA_MIN = 0.001


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(state: dict, path: Path, _attempts: int = 5) -> None:
    """Atomic rewrite: write a temp file beside the target, then replace.

    `os.replace` is atomic on POSIX and on Windows, so a reader never sees a
    half-written document even if it polls while the queue is writing. The
    temp name carries the pid so two processes cannot collide on it either.

    Two hardening details, both learned from an intermittent failure of
    `tests/unit/test_a3_queue_isolation.py::test_no_temp_files_are_left_behind`
    that appeared only under full-suite load:

      - On Windows `os.replace` raises PermissionError if anything holds a
        transient handle to either path -- a virus scanner or the search
        indexer touching a file the queue just closed is enough. Left
        unhandled that kills the queue process, and with it a seed of a
        multi-day campaign, over a bookkeeping write. It is retried briefly.
      - The temp file is removed in `finally`, so a failed write never leaves
        a stray `.tmp` that a later reader might mistake for status.

    Bookkeeping must not be able to end a run: if the replace genuinely
    cannot succeed the exception still propagates, but only after the retries
    and the cleanup.
    """
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        for attempt in range(_attempts):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == _attempts - 1:
                    raise
                time.sleep(0.1 * (attempt + 1))
    finally:
        tmp.unlink(missing_ok=True)


class QueueLock:
    """Refuse to start a second queue process for the same seed.

    Per-seed status files stop two processes from clobbering each other's
    bookkeeping, but they do not stop two processes from running the SAME
    seed and racing on its run directories and checkpoints. This does.

    `O_CREAT | O_EXCL` is atomic on both platforms; a stale lock (a killed
    queue) is reported with the pid that held it and must be removed by hand,
    because silently stealing it is how two trainers end up writing one
    checkpoint.
    """

    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> QueueLock:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                held = self.path.read_text(encoding="utf-8").strip()
            except OSError:
                held = "unknown"
            raise SystemExit(
                f"HALT: {self.path.name} already exists (held by {held}).\n"
                f"Another queue process is running this seed, or a previous one "
                f"was killed. Check, then remove the lock file to proceed."
            ) from None
        os.write(self.fd, f"pid={os.getpid()} started={now()}".encode())
        return self

    def __exit__(self, *exc) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.path.unlink(missing_ok=True)


def rounds_done(run_dir: Path) -> int:
    f = run_dir / "rounds.jsonl"
    if not f.exists():
        return 0
    with f.open(encoding="utf-8") as fh:
        return sum(1 for ln in fh if ln.strip())


def run_gates(cond: str, seed: int, run_dir: Path, log: Path) -> tuple[bool, str]:
    """Section 7 stop conditions for one finished run."""
    k = rounds_done(run_dir)
    if k != EXPECTED_ROUNDS:
        return False, f"expected {EXPECTED_ROUNDS} rounds, found {k}"

    md = run_dir / "run_metadata.json"
    if not md.exists():
        return False, "run_metadata.json missing"
    meta = json.loads(md.read_text(encoding="utf-8"))
    cfg_snap = meta.get("config_snapshot", {})

    # A3-G1-iv from PRIMARY EVIDENCE, not the metadata summary.
    rollout = int(cfg_snap.get("rollout_length", 0))
    if k * rollout != EXPECTED_PPO_STEPS:
        return False, (f"A3-G1-iv: derived PPO env steps {k * rollout} "
                       f"({k} x {rollout}) != {EXPECTED_PPO_STEPS}")
    src_steps = 0
    with (run_dir / "rounds.jsonl").open(encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                src_steps += int((json.loads(ln).get("source_batch") or {}).get("env_steps", 0))
    if src_steps != EXPECTED_SOURCE_STEPS:
        return False, (f"A3-G1-iv: derived source env steps {src_steps} != "
                       f"{EXPECTED_SOURCE_STEPS}")

    audit = meta.get("source_seed_audit", {})
    if audit.get("duplicate_seed_events", -1) != 0:
        return False, f"A3-G1-iii: {audit.get('duplicate_seed_events')} duplicate source seeds"

    if int(cfg_snap.get("source_collection", {}).get("M", 0)) != 5:
        return False, f"A3: M is not 5 ({cfg_snap.get('source_collection', {}).get('M')})"

    atk = cfg_snap.get("attack", {})
    if cond in ("b", "c"):
        if atk.get("corrupted_source_ids") != [ATTACKED[seed]]:
            return False, (f"section 5 mapping violated: corrupted "
                           f"{atk.get('corrupted_source_ids')}, expected [{ATTACKED[seed]!r}]")
        if atk.get("name") != "primary" or atk.get("budget_ratio") != 0.5:
            return False, f"A3-G2-i: attack block is not the primary attack: {atk}"
    else:
        if atk.get("name") != "none" or atk.get("f") != 0:
            return False, f"A3-G2-iii: {LABEL[cond]}' must be clean, got {atk}"

    dfn = cfg_snap.get("defense", {})
    if cond in ("c", "e"):
        if dfn.get("name") != "rce" or dfn.get("f") != 1 or dfn.get("beta") != 1.5:
            return False, f"A3-G1-i: defense is not the declared RCE block: {dfn}"
        calib = run_dir / "guarantee_calibration.json"
        if not calib.exists():
            return False, "A3-G1-v: guarantee_calibration.json missing from an RCE run"
        eps = json.loads(calib.read_text(encoding="utf-8")).get("epsilon_offline")
        # A3 inverts A2's expectation: the MAD is live, so this must NOT be
        # the sigma_min constant. If it is, the retained set degenerated and
        # A3 has not tested the margin.
        if eps is None or float(eps) <= SIGMA_MIN + 1e-12:
            return False, (f"A3-G1-v: epsilon_offline={eps} <= sigma_min={SIGMA_MIN}; the "
                           f"MAD is not live and A3 has NOT tested the margin")
    else:
        if dfn.get("name") != "mean" or dfn.get("f") != 0:
            return False, f"A3: {LABEL[cond]}' must be undefended, got {dfn}"

    # The mechanism validator runs only for the RCE conditions, and needs its
    # same-seed undefended partner to already exist.
    if cond in ("c", "e"):
        paired = OUT_ROOT / (f"B_seed{seed}" if cond == "c" else f"A_seed{seed}")
        if rounds_done(paired) < EXPECTED_ROUNDS:
            return False, f"paired reference {paired.name} is not complete; cannot gate"
        report = run_dir / "a3_mechanism_report.json"
        cmd = [PY, str(ROOT / "scripts/a3_rce_mechanism_validation.py"),
               "--run", str(run_dir), "--condition", LABEL[cond],
               "--paired", str(paired), "--out", str(report)]
        if cond == "c":
            cmd += ["--attacked-source", ATTACKED[seed]]
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"\n=== gates {LABEL[cond]}_seed{seed} {now()} ===\n")
            fh.flush()
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
        if rc != 0:
            return False, f"M=5 mechanism validation failed (rc={rc}); see {report}"

    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="run just this one, e.g. C_seed0")
    ap.add_argument("--seed", type=int, default=None, choices=[0, 1, 2],
                    help="restrict this process to one seed's four runs, and give it "
                         "its own status file. Required for concurrent execution.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Scheduling only: the same QUEUE, filtered. The order within a seed is
    # untouched (A' -> B' -> C' -> E'), and so is every run's config.
    queue = [(c, s) for c, s in QUEUE if args.seed is None or s == args.seed]
    if not queue:
        print(f"no runs match --seed {args.seed}")
        return 1

    status = status_path(args.seed)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_dir = OUT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    with QueueLock(lock_path(args.seed)):
        return _run_queue(args, queue, status, log_dir)


def _run_queue(args, queue, status: Path, log_dir: Path) -> int:
    state = {"started": now(),
             "machine": "gcp t2d-standard-60 (AMD Milan, x86-64) -- gates doc section 15",
             "seed_scope": args.seed,
             "queue": [f"{LABEL[c]}_seed{s}" for c, s in queue],
             "runs": {}, "halted": None}
    if status.exists():
        try:
            state.update(json.loads(status.read_text(encoding="utf-8")))
            state["halted"] = None
            state["restarted"] = now()
        except Exception:
            pass
    write_status(state, status)

    scope = "all 12" if args.seed is None else f"seed {args.seed} ({len(queue)} runs)"
    print(f"[{now()}] A3 queue scope: {scope} -> {status.name}")
    print(f"[{now()}] A3-G1-i: verifying the twelve configs change exactly what A3 declares")
    if not args.dry_run:
        rc = subprocess.call([PY, str(ROOT / "scripts/a3_verify_frozen.py")], cwd=str(ROOT))
        if rc != 0:
            state["halted"] = "A3-G1-i failed: configs are not the declared A3 design"
            write_status(state, status)
            print("HALT:", state["halted"])
            return 1

    for cond, seed in queue:
        name = f"{LABEL[cond]}_seed{seed}"
        if args.only and args.only != name:
            continue
        run_dir = OUT_ROOT / name
        cfg = ROOT / f"configs/experiment/a3/{cond}_seed{seed}.yaml"
        log = log_dir / f"{name}.log"

        if state["runs"].get(name, {}).get("gates_pass") is True:
            print(f"[{now()}] {name}: already complete and gated, skipping")
            continue

        done = rounds_done(run_dir)

        # Never re-enter a finished run: resuming trains zero rounds and still
        # rewrites run_metadata.json with this invocation's env_steps and a
        # near-zero timing block. A complete run needs gating, not training.
        if done >= EXPECTED_ROUNDS:
            print(f"[{now()}] {name}: already at {done} rounds -- gating only")
            ok, why = run_gates(cond, seed, run_dir, log)
            state["runs"].setdefault(name, {}).update(
                status="complete" if ok else "gate-failed", gates_pass=ok,
                gate_note=why, gated_without_retraining=True)
            write_status(state, status)
            print(f"[{now()}] {name}: {'gates PASS' if ok else 'GATES FAIL -- ' + why}")
            if not ok:
                state["halted"] = f"{name}: {why}"
                write_status(state, status)
                return 1
            continue

        note = f" (resuming from round {done})" if done else ""
        print(f"[{now()}] {name}: launching{note}  cfg={cfg.name}"
              + (f"  attacked={ATTACKED[seed]}" if cond in ("b", "c") else "  (clean)"))
        state["runs"][name] = {
            "started": now(), "config": str(cfg.relative_to(ROOT)),
            "condition": LABEL[cond] + "'", "seed": seed,
            "attacked_source": ATTACKED[seed] if cond in ("b", "c") else None,
            "resumed_from_round": done, "status": "running"}
        write_status(state, status)
        if args.dry_run:
            state["runs"][name]["status"] = "dry-run"
            write_status(state, status)
            continue

        t0 = time.time()
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"\n=== {name} start {now()} ===\n")
            fh.flush()
            rc = subprocess.call(
                [PY, str(ROOT / "scripts/train.py"), "--config", str(cfg), "--eval-every", "1"],
                stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
        elapsed = time.time() - t0
        state["runs"][name].update(finished=now(), wall_clock_s=round(elapsed, 1), returncode=rc)

        if rc != 0:
            state["runs"][name]["status"] = "failed"
            state["halted"] = f"{name}: training exited rc={rc}; see {log}"
            write_status(state, status)
            print("HALT:", state["halted"])
            return 1

        ok, why = run_gates(cond, seed, run_dir, log)
        state["runs"][name].update(status="complete" if ok else "gate-failed",
                                   gates_pass=ok, gate_note=why)
        write_status(state, status)
        print(f"[{now()}] {name}: {'gates PASS' if ok else 'GATES FAIL -- ' + why}  "
              f"({elapsed / 3600:.2f} h)")
        if not ok:
            state["halted"] = f"{name}: {why}"
            write_status(state, status)
            print("HALT: a section 7 stop condition fired. Fix the implementation "
                  "before continuing.")
            return 1

    state["finished"] = now()
    write_status(state, status)
    print(f"[{now()}] A3 queue complete. Analyse with: python scripts/analyze_a3.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
