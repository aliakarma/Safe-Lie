#!/usr/bin/env python
"""A1 production run queue — sequential, gated, resumable.

Runs the six A1 training runs one after another and enforces
docs/a1_attack_gates.md §7 after each one, so a stop condition halts the
queue instead of spending another 38 hours producing artifacts that are
already known to be invalid.

**Order is deliberate: all three B runs first.** Condition A already
exists, so finishing B0/B1/B2 completes the *primary* contrast B−A even
if the machine is taken back before the D runs finish. Running them
interleaved would leave the primary contrast incomplete at every
intermediate stopping point.

**Sequential, not parallel.** The source collector already uses all 12
logical CPUs (`workers: 12, chunks_per_worker: 4`); two concurrent runs
would not finish sooner and would make the per-run stop rule unusable.

**Resumable.** `run_experiment_with_oracle` resumes from the checkpoint in
the run's output directory, so re-invoking this script after an
interruption continues rather than restarting, and already-gated runs are
skipped.

Usage:
    python scripts/a1_run_queue.py
    python scripts/a1_run_queue.py --only B_seed0
    python scripts/a1_run_queue.py --dry-run
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
OUT_ROOT = ROOT / "results" / "runs_a1"
STATUS = OUT_ROOT / "a1_queue_status.json"

# §3 balanced mapping. (condition, seed) -> attacked source.
ATTACKED = {0: "batch_1", 1: "batch_2", 2: "batch_3"}
CLEAN_DIR = {
    0: ROOT / "results/runs_constraint_batch_g9/g9_batch_clean",
    1: ROOT / "results/runs_constraint_batch_g10/seed1",
    2: ROOT / "results/runs_constraint_batch_g10/seed2",
}
# B first -- see the module docstring.
QUEUE = [("B", s) for s in (0, 1, 2)] + [("D", s) for s in (0, 1, 2)]

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


def run_gates(cond: str, seed: int, run_dir: Path, log: Path) -> tuple[bool, str]:
    """§7 stop conditions for one finished run, against its clean pair."""
    report = run_dir / "a1_mechanism_report.json"
    cmd = [
        PY, str(ROOT / "scripts/a1_mechanism_validation.py"),
        "--clean", str(CLEAN_DIR[seed]),
        "--attack" if cond == "B" else "--noise", str(run_dir),
        "--B", "12.5", "--M", "3",
        "--attacked-source", ATTACKED[seed],
        "--out", str(report),
    ]
    # The validator always needs an --attack path; for a D run point it at
    # the clean run so the attack-specific identities are trivially the
    # no-shift case, and let --noise carry the D-specific checks.
    if cond == "D":
        cmd = [
            PY, str(ROOT / "scripts/a1_mechanism_validation.py"),
            "--clean", str(CLEAN_DIR[seed]),
            "--attack", str(CLEAN_DIR[seed]),
            "--noise", str(run_dir),
            "--B", "12.5", "--M", "3",
            "--attacked-source", ATTACKED[seed],
            "--out", str(report),
        ]
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== gates {cond}_seed{seed} {now()} ===\n")
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    if rc != 0:
        return False, f"mechanism validation failed (rc={rc}); see {report}"

    k = rounds_done(run_dir)
    if k != EXPECTED_ROUNDS:
        return False, f"expected {EXPECTED_ROUNDS} rounds, found {k}"

    md = run_dir / "run_metadata.json"
    if not md.exists():
        return False, "run_metadata.json missing"
    meta = json.loads(md.read_text(encoding="utf-8"))
    steps = meta.get("env_steps", {})
    if steps.get("ppo") != 500_000:
        return False, f"A1-S4: PPO env steps {steps.get('ppo')} != 500000"
    audit = meta.get("source_seed_audit", {})
    if audit.get("duplicate_seed_events", -1) != 0:
        return False, f"A1-S5: {audit.get('duplicate_seed_events')} duplicate source seeds"
    if audit.get("n_env_seeds_issued") != 23_100:
        return False, f"A1-S5: {audit.get('n_env_seeds_issued')} env seeds issued, expected 23100"
    cfg = meta.get("config_snapshot", {})
    if cfg.get("attack", {}).get("corrupted_source_ids") != [ATTACKED[seed]]:
        return False, ("§3 mapping violated: run corrupted "
                       f"{cfg.get('attack', {}).get('corrupted_source_ids')}, "
                       f"expected [{ATTACKED[seed]!r}]")
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="run just this one, e.g. B_seed0")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_dir = OUT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    state = {"started": now(), "queue": [f"{c}_seed{s}" for c, s in QUEUE],
             "runs": {}, "halted": None}
    if STATUS.exists():
        try:
            state.update(json.loads(STATUS.read_text(encoding="utf-8")))
            state["halted"] = None
            state["restarted"] = now()
        except Exception:
            pass
    write_status(state)

    # A1-S10, once, before anything runs.
    print(f"[{now()}] A1-S10: verifying the configs are frozen G10 + attack only")
    if not args.dry_run:
        rc = subprocess.call([PY, str(ROOT / "scripts/a1_verify_frozen.py")], cwd=str(ROOT))
        if rc != 0:
            state["halted"] = "A1-S10 failed: configs differ from G10 beyond the treatment"
            write_status(state)
            print("HALT:", state["halted"])
            return 1

    for cond, seed in QUEUE:
        name = f"{cond}_seed{seed}"
        if args.only and args.only != name:
            continue
        run_dir = OUT_ROOT / name
        cfg = ROOT / f"configs/experiment/a1/{cond.lower()}_seed{seed}.yaml"
        log = log_dir / f"{name}.log"

        prev = state["runs"].get(name, {})
        if prev.get("gates_pass") is True:
            print(f"[{now()}] {name}: already complete and gated, skipping")
            continue

        done = rounds_done(run_dir)
        note = f" (resuming from round {done})" if done else ""
        print(f"[{now()}] {name}: launching{note}  cfg={cfg.name}  "
              f"attacked={ATTACKED[seed]}")
        state["runs"][name] = {"started": now(), "config": str(cfg.relative_to(ROOT)),
                               "attacked_source": ATTACKED[seed],
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
                [PY, str(ROOT / "scripts/train.py"), "--config", str(cfg), "--eval-every", "1"],
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

        ok, why = run_gates(cond, seed, run_dir, log)
        state["runs"][name].update(status="complete" if ok else "gate-failed",
                                   gates_pass=ok, gate_note=why)
        write_status(state)
        print(f"[{now()}] {name}: {'gates PASS' if ok else 'GATES FAIL -- ' + why}  "
              f"({elapsed / 3600:.2f} h)")
        if not ok:
            state["halted"] = f"{name}: {why}"
            write_status(state)
            print("HALT: a §7 stop condition fired. Fix the implementation before continuing.")
            return 1

    state["finished"] = now()
    write_status(state)
    print(f"[{now()}] A1 queue complete. Analyse with: python scripts/analyze_a1.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
