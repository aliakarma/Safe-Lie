#!/usr/bin/env python
"""A1 run watchdog: stall detection + per-seed verification.

Runs alongside `scripts/a1_run_queue.py` and does three things the queue
driver cannot do from inside a blocking `subprocess.call`:

1. **Flags stalls while they are happening.** Round 55 of B_seed0 took
   9,372 s instead of 78 s because the machine slept for 2 h 34 min
   (Kernel-Power 42 at 06:34:14 UTC, resume 107 at 09:08:43 UTC). That was
   only discovered hours later. Two rules now catch it:

   * *retrospective* -- a completed round whose source wall clock exceeds
     `--factor` x the median of that run's rounds so far;
   * *live* -- no new round for longer than `--factor` x median, reported
     while the stall is still in progress.

   The median needs `--min-rounds` samples before either rule arms, so a
   cold start does not produce noise.

2. **Verifies each seed the moment it finishes**, by invoking
   `scripts/a1_verify_run.py`, so a defect surfaces after ~7 h rather than
   after ~39 h. Only the structural gates run here; the scientific
   contrasts stay in `analyze_a1.py`, where they are evaluated over all
   three seeds at once. Per-seed verification must not become a way to peek
   at the outcome one seed at a time.

3. **Asks Windows not to sleep.** `SetThreadExecutionState(ES_CONTINUOUS |
   ES_SYSTEM_REQUIRED)` is held for the watchdog's lifetime. This blocks
   *idle* sleep only -- it cannot stop a manual sleep, a lid-close action,
   or a critical-battery hibernate. On battery the job will still die; the
   fix for that is mains power, not this flag.

Nothing here writes to any training artifact; it is read-only over
`rounds.jsonl` plus the verification JSON the verifier itself emits.

Usage:
    python scripts/a1_watchdog.py                    # poll until all 6 runs verify
    python scripts/a1_watchdog.py --factor 5 --interval 60
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT_ROOT = ROOT / "results" / "runs_a1"
STATE = OUT_ROOT / "a1_watchdog.json"
EVENTS = OUT_ROOT / "logs" / "a1_watchdog.log"

EXPECTED_ROUNDS = 250
RUNS = [f"{c}_seed{s}" for c in ("B", "D") for s in (0, 1, 2)]

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def keep_awake() -> bool:
    """Block idle sleep for this process's lifetime. Windows only."""
    try:
        rv = ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        return bool(rv)
    except Exception:
        return False


def release_awake() -> None:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass


def round_times(run_dir: Path) -> tuple[list[int], list[float]]:
    """(round indices, source wall clock per round) -- cheap, streaming."""
    f = run_dir / "rounds.jsonl"
    if not f.exists():
        return [], []
    ks, ts = [], []
    with f.open(encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                break                      # partial final line while writing
            sb = r.get("source_batch") or {}
            ks.append(r["round_k"])
            ts.append(float(sb.get("wall_clock_s", 0.0))
                      + float(sb.get("reference_wall_clock_s") or 0.0))
    return ks, ts


def verify_run(name: str) -> dict:
    run_dir = OUT_ROOT / name
    rc = subprocess.call(
        [PY, str(ROOT / "scripts/a1_verify_run.py"), "--run", str(run_dir)],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rep_path = run_dir / "a1_run_verification.json"
    summary = {"returncode": rc, "pass": rc == 0}
    if rep_path.exists():
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        summary["failed_gates"] = [c["gate"] for c in rep["checks"] if c["pass"] is False]
        summary["rounds"] = rep["rounds"]
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--factor", type=float, default=5.0,
                    help="stall threshold, as a multiple of the run's median round time")
    ap.add_argument("--interval", type=float, default=60.0, help="poll seconds")
    ap.add_argument("--min-rounds", type=int, default=10,
                    help="rounds needed before the median is trusted")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    args = ap.parse_args()

    awake = keep_awake()
    log(f"watchdog start: factor={args.factor}x median, poll={args.interval}s, "
        f"idle-sleep blocked={awake}")
    if not awake:
        log("WARNING: could not acquire the no-sleep request; the machine may idle-sleep")

    state: dict = {"started": now(), "factor": args.factor, "runs": {}, "events": []}
    if STATE.exists():
        try:
            state.update(json.loads(STATE.read_text(encoding="utf-8")))
        except Exception:
            pass
    state.setdefault("runs", {})
    reported_stalls: dict[str, set] = {n: set() for n in RUNS}
    last_seen: dict[str, tuple[int, float]] = {}
    live_warned: dict[str, float] = {}

    try:
        while True:
            for name in RUNS:
                run_dir = OUT_ROOT / name
                if not run_dir.exists():
                    continue
                ks, ts = round_times(run_dir)
                n = len(ks)
                if n == 0:
                    continue
                rec = state["runs"].setdefault(name, {})
                rec["rounds"] = n

                med = float(np.median(ts)) if n >= args.min_rounds else None
                if med:
                    rec["median_round_s"] = round(med, 1)
                    thresh = args.factor * med
                    # (a) retrospective: a completed round that overran
                    for k, t in zip(ks, ts, strict=True):
                        if t > thresh and k not in reported_stalls[name]:
                            reported_stalls[name].add(k)
                            ev = {"kind": "stall", "run": name, "round": k,
                                  "seconds": round(t, 1), "median": round(med, 1),
                                  "factor": round(t / med, 1), "at": now()}
                            state["events"].append(ev)
                            log(f"STALL  {name} round {k}: {t:.0f} s "
                                f"= {t / med:.0f}x median ({med:.0f} s)")
                    # (b) live: no new round for longer than the threshold.
                    # Suppressed once the run is complete -- a finished run
                    # stops producing rounds by definition, and warning about
                    # that at the end of all six runs would train the reader
                    # to ignore the warning that matters.
                    prev = last_seen.get(name)
                    if prev and prev[0] == n and n < EXPECTED_ROUNDS:
                        waiting = time.time() - prev[1]
                        if waiting > thresh and live_warned.get(name, 0) < prev[1]:
                            live_warned[name] = prev[1]
                            ev = {"kind": "stall_in_progress", "run": name,
                                  "stuck_after_round": ks[-1],
                                  "waiting_s": round(waiting), "median": round(med, 1),
                                  "at": now()}
                            state["events"].append(ev)
                            log(f"STALL IN PROGRESS  {name}: no new round for "
                                f"{waiting:.0f} s after round {ks[-1]} "
                                f"({waiting / med:.0f}x median) -- check the machine")
                    if not prev or prev[0] != n:
                        last_seen[name] = (n, time.time())
                else:
                    last_seen[name] = (n, last_seen.get(name, (0, time.time()))[1]
                                       if last_seen.get(name, (0, 0))[0] == n else time.time())

                # completed -> verify once
                if n >= EXPECTED_ROUNDS and not rec.get("verified"):
                    if not (run_dir / "run_metadata.json").exists():
                        continue
                    meta = json.loads((run_dir / "run_metadata.json").read_text(
                        encoding="utf-8"))
                    if not meta.get("env_steps"):
                        continue                      # still finalising
                    log(f"{name} reached {n} rounds -- running full verification")
                    res = verify_run(name)
                    rec["verified"] = True
                    rec["verification"] = res
                    state["events"].append({"kind": "verification", "run": name,
                                            "at": now(), **res})
                    if res["pass"]:
                        log(f"VERIFIED  {name}: all structural gates pass")
                    else:
                        log(f"VERIFICATION FAILED  {name}: {res.get('failed_gates')}")

            state["updated"] = now()
            OUT_ROOT.mkdir(parents=True, exist_ok=True)
            STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")

            done = [n for n in RUNS if state["runs"].get(n, {}).get("verified")]
            if len(done) == len(RUNS):
                log("all six runs verified -- watchdog exiting")
                return 0
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log("watchdog interrupted")
        return 130
    finally:
        release_awake()


if __name__ == "__main__":
    raise SystemExit(main())
