#!/usr/bin/env python
"""Unconstrained-control run queue — sequential, gated, resumable.

Runs the three condition-U seeds one after another and enforces
docs/unconstrained_control.md section 8 (U-S1..U-S7) after each one, so a
structural defect halts the queue after ~8 hours instead of after ~25.

**Seed 0 first, and alone until it passes.** Section 10 of the
pre-declaration requires it: a defect in the treatment — a multiplier that
is not actually pinned, an attack that crept in, a config that did not
load as intended — is a defect in all three runs, and there is no reason
to pay for it three times before noticing.

**Sequential, not parallel.** The source collector already occupies all 12
logical CPUs (`workers: 12, chunks_per_worker: 4`), so two concurrent runs
would not finish sooner and would make the per-run stop rule unusable.
Same reasoning as `scripts/a1_run_queue.py`.

**Resumable.** `run_experiment_with_oracle` resumes from the checkpoint in
the run's output directory, so re-invoking this script after an
interruption continues rather than restarting, and already-verified runs
are skipped. Safe to re-run at any time.

Usage:
    python scripts/unconstrained_run_queue.py
    python scripts/unconstrained_run_queue.py --only U_seed0
    python scripts/unconstrained_run_queue.py --dry-run
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
OUT_ROOT = ROOT / "results" / "runs_unconstrained"
STATUS = OUT_ROOT / "queue_status.json"
EXPECTED_ROUNDS = 250
SEEDS = (0, 1, 2)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(state: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATUS)


def rounds_done(run_dir: Path) -> int:
    f = run_dir / "rounds.jsonl"
    if not f.exists():
        return 0
    with f.open(encoding="utf-8") as fh:
        return sum(1 for ln in fh if ln.strip())


def verify(run_dir: Path, log: Path) -> tuple[bool, str]:
    cmd = [PY, str(ROOT / "scripts/verify_unconstrained_run.py"), "--run", str(run_dir)]
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n\n===== verify {run_dir.name} @ {now()} =====\n")
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    vf = run_dir / "unconstrained_run_verification.json"
    detail = ""
    if vf.exists():
        v = json.loads(vf.read_text(encoding="utf-8"))
        failed = [g["gate"] for g in v.get("gates", []) if not g["pass"]]
        detail = "; ".join(failed) if failed else "all gates pass"
    return rc == 0, detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="run just this run_id, e.g. U_seed0")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-config-check", action="store_true",
                    help="skip the pre-launch config verification (NOT recommended)")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    logs = OUT_ROOT / "logs"
    logs.mkdir(exist_ok=True)

    queue = [s for s in SEEDS if args.only is None or args.only == f"U_seed{s}"]
    if not queue:
        print(f"nothing matches --only {args.only}")
        return 2

    state: dict = {
        "predeclaration": "docs/unconstrained_control.md",
        "started_at": now(),
        "queue": [f"U_seed{s}" for s in queue],
        "runs": {},
    }
    if STATUS.exists():
        try:
            prev = json.loads(STATUS.read_text(encoding="utf-8"))
            state["runs"] = prev.get("runs", {})
            state["started_at"] = prev.get("started_at", state["started_at"])
        except Exception:
            pass

    # ---- pre-launch: the loaded ExperimentConfig must be the control
    if not args.skip_config_check:
        rc = subprocess.call([PY, str(ROOT / "scripts/verify_unconstrained_config.py")],
                             cwd=str(ROOT))
        if rc != 0:
            print("\nHALT: pre-launch config verification failed. Nothing launched.")
            state["halted"] = "config verification failed"
            write_status(state)
            return 1
    state["config_verification"] = "results/runs_unconstrained/config_verification.json"

    if args.dry_run:
        for s in queue:
            cfg = f"configs/experiment/unconstrained/u_seed{s}.yaml"
            print(f"  would run: {PY} scripts/train.py --config {cfg}")
            print(f"             -> {OUT_ROOT / f'U_seed{s}'}  ({rounds_done(OUT_ROOT / f'U_seed{s}')}/250 done)")
        return 0

    for s in queue:
        rid = f"U_seed{s}"
        run_dir = OUT_ROOT / rid
        cfg = ROOT / f"configs/experiment/unconstrained/u_seed{s}.yaml"
        log = logs / f"{rid}.log"

        done = rounds_done(run_dir)
        if done >= EXPECTED_ROUNDS:
            ok, detail = verify(run_dir, log)
            state["runs"][rid] = {"status": "verified" if ok else "VERIFY FAILED",
                                  "rounds": done, "detail": detail, "at": now()}
            write_status(state)
            print(f"[{rid}] already complete ({done} rounds) -- {detail}")
            if not ok:
                print(f"\nHALT: {rid} failed structural verification. Queue stopped.")
                state["halted"] = f"{rid} failed verification"
                write_status(state)
                return 1
            continue

        print(f"[{rid}] launching ({done}/{EXPECTED_ROUNDS} rounds already on disk) "
              f"-> {log}")
        state["runs"][rid] = {"status": "running", "rounds_at_start": done,
                              "started": now()}
        write_status(state)

        t0 = time.time()
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"\n\n===== train {rid} @ {now()} =====\n")
            fh.flush()
            rc = subprocess.call(
                [PY, str(ROOT / "scripts/train.py"), "--config", str(cfg)],
                stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
        elapsed = time.time() - t0

        if rc != 0:
            state["runs"][rid] = {"status": "TRAIN FAILED", "returncode": rc,
                                  "elapsed_s": elapsed, "at": now()}
            state["halted"] = f"{rid} training exited {rc}"
            write_status(state)
            print(f"\nHALT: {rid} training exited {rc}. Queue stopped. See {log}")
            return 1

        ok, detail = verify(run_dir, log)
        state["runs"][rid] = {"status": "verified" if ok else "VERIFY FAILED",
                              "rounds": rounds_done(run_dir), "elapsed_s": elapsed,
                              "detail": detail, "at": now()}
        write_status(state)
        print(f"[{rid}] finished in {elapsed/3600:.2f} h -- {detail}")

        if not ok:
            state["halted"] = f"{rid} failed structural verification"
            write_status(state)
            print(f"\nHALT: {rid} failed structural verification. Queue stopped. "
                  f"Seeds after it were NOT launched. See {log}")
            return 1

    state["finished_at"] = now()
    write_status(state)
    print("\nQueue complete. Next: python scripts/analyze_unconstrained.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
