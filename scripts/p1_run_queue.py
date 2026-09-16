#!/usr/bin/env python
"""P1 production run queue — sequential, gated, resumable.

docs/p1_concentrated_attack_gates.md Sections 5, 13, 14.

Order is paired by seed, ring first, as specified in the campaign brief:

    R_seed0, I_seed0, R_seed1, I_seed1, R_seed2, I_seed2

so that at every intermediate stopping point the completed work is a set of
COMPLETE (ring, W=I) pairs at matched seeds -- which is the unit the
localization-vs-spreading contrast is made on. Stopping after an odd number
of runs leaves a seed unpaired and answers nothing.

Sequential, not parallel: the source collector already uses all 12 logical
CPUs (`workers: 12, chunks_per_worker: 4`), so two concurrent runs would not
finish sooner and would make the per-run stop rule unusable.

After each run `scripts/p1_verify_run.py` enforces the structural gates. A
failure HALTS the queue rather than spending another ~10 h producing
artifacts already known to be invalid. A completed, verified run is never
overwritten or rerun.

Usage:
    python scripts/p1_run_queue.py
    python scripts/p1_run_queue.py --dry-run
    python scripts/p1_run_queue.py --only R_seed0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT_ROOT = ROOT / "results" / "runs_p1"
STATUS = OUT_ROOT / "p1_queue_status.json"
EXPECTED_ROUNDS = 250

# Paired by seed, ring first.
QUEUE = [(c, s) for s in (0, 1, 2) for c in ("R", "I")]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cfg_path(cond: str, seed: int) -> Path:
    return ROOT / f"configs/experiment/p1/{cond.lower()}_seed{seed}.yaml"


def run_dir(cond: str, seed: int) -> Path:
    return OUT_ROOT / f"{cond}_seed{seed}"


def rounds_done(d: Path) -> int:
    f = d / "rounds.jsonl"
    if not f.exists():
        return 0
    with f.open(encoding="utf-8") as fh:
        return sum(1 for ln in fh if ln.strip())


def write_status(state: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(state, indent=2), encoding="utf-8")


def verified(d: Path) -> bool:
    r = d / "p1_run_verification.json"
    if not r.exists():
        return False
    try:
        return bool(json.loads(r.read_text(encoding="utf-8")).get("pass"))
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="Run a single run_id, e.g. R_seed0")
    args = ap.parse_args()

    queue = [(c, s) for c, s in QUEUE if args.only in (None, f"{c}_seed{s}")]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_dir = OUT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    state = {"campaign": "P1", "started_at": now(), "queue": [f"{c}_seed{s}" for c, s in queue],
             "expected_rounds": EXPECTED_ROUNDS, "runs": {}}
    if STATUS.exists():
        try:
            prev = json.loads(STATUS.read_text(encoding="utf-8"))
            state["runs"] = prev.get("runs", {})
        except Exception:
            pass

    print(f"P1 queue: {len(queue)} run(s), sequential, ~9.8 h each "
          f"(~{9.76 * len(queue):.0f} h total)")
    for c, s in queue:
        print(f"   {c}_seed{s}  <- {cfg_path(c, s).relative_to(ROOT)}")
    print()
    if args.dry_run:
        for c, s in queue:
            p = cfg_path(c, s)
            assert p.exists(), f"missing config {p}"
        print("dry-run: all configs present.")
        return 0

    for cond, seed in queue:
        rid = f"{cond}_seed{seed}"
        d = run_dir(cond, seed)
        done = rounds_done(d)

        if verified(d) and done >= EXPECTED_ROUNDS:
            print(f"[{now()}] {rid}: already complete and verified ({done} rounds) -- skipping.")
            state["runs"][rid] = {**state["runs"].get(rid, {}), "status": "complete", "rounds": done}
            write_status(state)
            continue

        print(f"[{now()}] {rid}: starting ({done}/{EXPECTED_ROUNDS} rounds present, resuming if any).")
        state["runs"][rid] = {"status": "running", "started_at": now(), "rounds_at_start": done}
        write_status(state)

        log = log_dir / f"{rid}.log"
        t0 = time.time()
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"\n===== {rid} launched {now()} =====\n")
            fh.flush()
            proc = subprocess.run(
                [PY, str(ROOT / "scripts/train.py"), "--config", str(cfg_path(cond, seed))],
                cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT, check=False,
            )
        dt = time.time() - t0

        if proc.returncode != 0:
            state["runs"][rid] = {**state["runs"][rid], "status": "FAILED_TRAIN",
                                  "returncode": proc.returncode, "wall_clock_s": dt,
                                  "ended_at": now(), "log": str(log.relative_to(ROOT))}
            write_status(state)
            print(f"[{now()}] {rid}: TRAINING FAILED (rc={proc.returncode}). HALTING. See {log}")
            return 1

        # ---- structural gates -------------------------------------------
        vreport = d / "p1_run_verification.json"
        v = subprocess.run(
            [PY, str(ROOT / "scripts/p1_verify_run.py"), "--run", str(d),
             "--expected-rounds", str(EXPECTED_ROUNDS), "--report", str(vreport)],
            cwd=str(ROOT), capture_output=True, text=True, check=False,
        )
        print(v.stdout)
        if v.returncode != 0:
            print(v.stderr)
            state["runs"][rid] = {**state["runs"][rid], "status": "FAILED_GATES",
                                  "wall_clock_s": dt, "ended_at": now(),
                                  "verification": str(vreport.relative_to(ROOT))}
            write_status(state)
            print(f"[{now()}] {rid}: STRUCTURAL GATES FAILED. HALTING -- not patching around it.")
            return 1

        state["runs"][rid] = {**state["runs"][rid], "status": "complete",
                              "rounds": rounds_done(d), "wall_clock_s": dt,
                              "hours": round(dt / 3600, 2), "ended_at": now(),
                              "verification": str(vreport.relative_to(ROOT))}
        write_status(state)
        print(f"[{now()}] {rid}: COMPLETE and verified in {dt/3600:.2f} h.\n")

        # Refresh the analysis after every run so a partial campaign is
        # still readable without waiting for the queue to drain.
        subprocess.run([PY, str(ROOT / "scripts/p1_analyze.py")],
                       cwd=str(ROOT), capture_output=True, text=True, check=False)

    state["finished_at"] = now()
    write_status(state)
    print(f"[{now()}] P1 queue finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
