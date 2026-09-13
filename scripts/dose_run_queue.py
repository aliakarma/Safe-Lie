#!/usr/bin/env python
"""DR dose-sweep production queue -- sequential, gated, resumable.

Runs the six new dose training runs one after another and enforces
docs/dose_response_gates.md section 7 after each one, so a stop condition
halts the queue instead of spending another 38 hours producing artifacts
that are already known to be invalid.

**Order is deliberate (gates doc section 13):**

    1. B100_seed0     2. B025_seed0
    3. B100_seed1     4. B025_seed1
    5. B100_seed2     6. B025_seed2

`B/d = 1.0` runs first at each seed because it is the condition most
likely to behave qualitatively differently, and a design problem there
should surface after ~8 h rather than after ~23 h. Pairing it immediately
with that seed's 0.25 run means that after every EVEN-numbered run the
four-dose grid is complete for one more seed -- no intermediate stopping
point leaves a dose present at some seeds and absent at others.

**Sequential, not parallel.** The source collector already uses all 12
logical CPUs (`workers: 12, chunks_per_worker: 4`); two concurrent runs
would not finish sooner and would make the per-run stop rule unusable.

**Never re-enters a finished run.** `run_experiment_with_oracle` resumes,
executes zero rounds, and still rewrites `run_metadata.json` with the
resuming invocation's `env_steps` and `timing` -- the defect that
clobbered A1's `D_seed0` metadata (gates doc section 12, still unfixed at
HEAD by deliberate decision). A complete run needs gating, not training.

Usage:
    python scripts/dose_run_queue.py
    python scripts/dose_run_queue.py --only B100_seed0
    python scripts/dose_run_queue.py --dry-run
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
OUT_ROOT = ROOT / "results" / "runs_dose"
STATUS = OUT_ROOT / "dose_queue_status.json"

# docs/dose_response_gates.md section 3. Do not tune here.
DOSE = {"025": 0.25, "100": 1.0}
D_BUDGET = 25.0
ATTACKED = {0: "batch_1", 1: "batch_2", 2: "batch_3"}
CLEAN_DIR = {
    0: ROOT / "results/runs_constraint_batch_g9/g9_batch_clean",
    1: ROOT / "results/runs_constraint_batch_g10/seed1",
    2: ROOT / "results/runs_constraint_batch_g10/seed2",
}

# Section 13's order: the high dose first at each seed, then its 0.25 pair.
QUEUE = [(tag, s) for s in (0, 1, 2) for tag in ("100", "025")]

EXPECTED_ROUNDS = 250


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(state: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(state, indent=2), encoding="utf-8")


def rounds_done(run_dir: Path) -> int:
    f = run_dir / "rounds.jsonl"
    if not f.exists():
        return 0
    with f.open(encoding="utf-8") as fh:
        return sum(1 for ln in fh if ln.strip())


def run_gates(tag: str, seed: int, run_dir: Path, log: Path) -> tuple[bool, str]:
    """Section 7 stop conditions for one finished run.

    Two independent checkers, both required:

    * `dose_verify_run.py` -- the full section 7 gate set, with B derived
      from the run's own config snapshot.
    * `a1_mechanism_validation.py` -- A1's committed mechanism checker,
      reused unmodified at this dose's `--B`. Reusing it rather than
      reimplementing it is the point: the identity that gated A1 gates
      these runs too, at a different magnitude.
    """
    B = DOSE[tag] * D_BUDGET

    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== gates B{tag}_seed{seed} {now()} ===\n")
        fh.flush()
        rc = subprocess.call(
            [PY, str(ROOT / "scripts/dose_verify_run.py"), "--run", str(run_dir)],
            stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
        if rc != 0:
            return False, (f"dose_verify_run failed (rc={rc}); see "
                           f"{run_dir / 'dose_run_verification.json'}")

        report = run_dir / "dose_mechanism_report.json"
        fh.write(f"\n=== a1_mechanism_validation B{tag}_seed{seed} at B={B} ===\n")
        fh.flush()
        rc = subprocess.call(
            [PY, str(ROOT / "scripts/a1_mechanism_validation.py"),
             "--clean", str(CLEAN_DIR[seed]),
             "--attack", str(run_dir),
             "--B", str(B), "--M", "3",
             "--attacked-source", ATTACKED[seed],
             "--out", str(report)],
            stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    if rc != 0:
        return False, f"mechanism validation failed (rc={rc}); see {report}"

    k = rounds_done(run_dir)
    if k != EXPECTED_ROUNDS:
        return False, f"expected {EXPECTED_ROUNDS} rounds, found {k}"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="run just this one, e.g. B100_seed0")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_dir = OUT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    state = {"predeclaration": "docs/dose_response_gates.md",
             "started": now(),
             "queue": [f"B{t}_seed{s}" for t, s in QUEUE],
             "runs": {}, "halted": None}
    if STATUS.exists():
        try:
            state.update(json.loads(STATUS.read_text(encoding="utf-8")))
            state["halted"] = None
            state["restarted"] = now()
        except Exception:
            pass
    write_status(state)

    # DR-S10, once, before anything runs.
    print(f"[{now()}] DR-S10: verifying the dose configs differ from A1 "
          f"condition B only in budget_ratio")
    if not args.dry_run:
        rc = subprocess.call([PY, str(ROOT / "scripts/dose_verify_frozen.py")],
                             cwd=str(ROOT))
        if rc != 0:
            state["halted"] = ("DR-S10 failed: dose configs differ from A1 "
                               "condition B beyond budget_ratio")
            write_status(state)
            print("HALT:", state["halted"])
            return 1

    for tag, seed in QUEUE:
        name = f"B{tag}_seed{seed}"
        if args.only and args.only != name:
            continue
        run_dir = OUT_ROOT / name
        cfg = ROOT / f"configs/experiment/dose/b{tag}_seed{seed}.yaml"
        log = log_dir / f"{name}.log"

        prev = state["runs"].get(name, {})
        if prev.get("gates_pass") is True:
            print(f"[{now()}] {name}: already complete and gated, skipping")
            continue

        done = rounds_done(run_dir)

        # Never re-enter a finished run -- see the module docstring.
        if done >= EXPECTED_ROUNDS:
            print(f"[{now()}] {name}: already at {done} rounds -- gating only, "
                  f"not re-entering training")
            ok, why = run_gates(tag, seed, run_dir, log)
            state["runs"].setdefault(name, {}).update(
                status="complete" if ok else "gate-failed",
                gates_pass=ok, gate_note=why, gated_without_retraining=True)
            write_status(state)
            print(f"[{now()}] {name}: {'gates PASS' if ok else 'GATES FAIL -- ' + why}")
            if not ok:
                state["halted"] = f"{name}: {why}"
                write_status(state)
                return 1
            continue

        note = f" (resuming from round {done})" if done else ""
        print(f"[{now()}] {name}: launching{note}  cfg={cfg.name}  "
              f"B/d={DOSE[tag]}  B={DOSE[tag] * D_BUDGET}  "
              f"corrupted={ATTACKED[seed]}")
        state["runs"][name] = {"started": now(),
                               "config": str(cfg.relative_to(ROOT)),
                               "budget_ratio": DOSE[tag],
                               "B": DOSE[tag] * D_BUDGET,
                               "corrupted_source": ATTACKED[seed],
                               "resumed_from_round": done, "status": "running"}
        write_status(state)
        if args.dry_run:
            state["runs"][name]["status"] = "dry-run"
            write_status(state)
            continue

        t0 = time.time()
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"\n=== {name} start {now()} ===\n")
            fh.flush()
            rc = subprocess.call(
                [PY, str(ROOT / "scripts/train.py"), "--config", str(cfg),
                 "--eval-every", "1"],
                stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
        elapsed = time.time() - t0
        state["runs"][name].update(finished=now(), wall_clock_s=round(elapsed, 1),
                                   returncode=rc)

        if rc != 0:
            state["runs"][name]["status"] = "failed"
            state["halted"] = f"{name}: training exited rc={rc}; see {log}"
            write_status(state)
            print("HALT:", state["halted"])
            return 1

        ok, why = run_gates(tag, seed, run_dir, log)
        state["runs"][name].update(status="complete" if ok else "gate-failed",
                                   gates_pass=ok, gate_note=why)
        write_status(state)
        print(f"[{now()}] {name}: {'gates PASS' if ok else 'GATES FAIL -- ' + why}  "
              f"({elapsed / 3600:.2f} h)")
        if not ok:
            state["halted"] = f"{name}: {why}"
            write_status(state)
            print("HALT: a section 7 stop condition fired. Fix the implementation "
                  "before continuing.")
            return 1

    state["finished"] = now()
    write_status(state)
    print(f"[{now()}] DR queue complete. Analyse with: "
          f"python scripts/analyze_dose.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
