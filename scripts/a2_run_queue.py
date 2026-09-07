#!/usr/bin/env python
"""A2 production run queue -- sequential, gated, resumable.

Runs the six A2 training runs one after another and enforces
docs/a2_rce_gates.md section 7 after each, so a stop condition halts the
queue instead of spending another 45 hours producing artifacts already
known to be invalid.

**Order is deliberate: C and E are interleaved by seed, C first.**
Unlike A1 -- where finishing all three B runs completed the primary
contrast -- A2's primary quantity is an interaction, `(C-E)-(B-A)`, which
needs BOTH new conditions at the SAME seed before it means anything. The
order C0, E0, C1, E1, C2, E2 therefore makes every intermediate stopping
point a complete, analysable design at 1, 2 or 3 seeds, instead of three
half-designs. Running all C first would leave the headline number
uncomputable until the very last run.

**Sequential, not parallel.** The source collector already uses all 12
logical CPUs; two concurrent runs would not finish sooner and would make
the per-run stop rule unusable.

**Resumable.** `run_experiment_with_oracle` resumes from the checkpoint in
the run's output directory, and already-gated runs are skipped. Note that
an RCE run re-executes its 20-round clean calibration phase on every
resume (it lives in `ExperimentRun.__init__`, before `restore`); that is
about 0.6 h per resume and it cannot change a number, since
`epsilon_offline` is a quantile of a constant vector at M=3, f=1.

Usage:
    python scripts/a2_run_queue.py
    python scripts/a2_run_queue.py --only C_seed0
    python scripts/a2_run_queue.py --dry-run
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
OUT_ROOT = ROOT / "results" / "runs_a2"
STATUS = OUT_ROOT / "a2_queue_status.json"

ATTACKED = {0: "batch_1", 1: "batch_2", 2: "batch_3"}      # gates doc section 3
CLEAN_DIR = {
    0: ROOT / "results/runs_constraint_batch_g9/g9_batch_clean",
    1: ROOT / "results/runs_constraint_batch_g10/seed1",
    2: ROOT / "results/runs_constraint_batch_g10/seed2",
}
ATTACK_DIR = {s: ROOT / f"results/runs_a1/B_seed{s}" for s in (0, 1, 2)}

# Interleaved by seed -- see the module docstring.
QUEUE = [(c, s) for s in (0, 1, 2) for c in ("C", "E")]

EXPECTED_ROUNDS = 250
EXPECTED_PPO_STEPS = 500_000
EXPECTED_SOURCE_STEPS = 45_000_000
EXPECTED_ENV_SEEDS = 23_100
SIGMA_MIN = 0.001


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
    """Section 7 stop conditions for one finished run, against its no-RCE pair."""
    paired = ATTACK_DIR[seed] if cond == "C" else CLEAN_DIR[seed]
    report = run_dir / "a2_mechanism_report.json"
    cmd = [
        PY, str(ROOT / "scripts/a2_rce_mechanism_validation.py"),
        "--run", str(run_dir), "--condition", cond,
        "--paired", str(paired),
        "--out", str(report),
    ]
    if cond == "C":
        cmd += ["--attacked-source", ATTACKED[seed]]
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== gates {cond}_seed{seed} {now()} ===\n")
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    if rc != 0:
        return False, f"RCE mechanism validation failed (rc={rc}); see {report}"

    k = rounds_done(run_dir)
    if k != EXPECTED_ROUNDS:
        return False, f"expected {EXPECTED_ROUNDS} rounds, found {k}"

    md = run_dir / "run_metadata.json"
    if not md.exists():
        return False, "run_metadata.json missing"
    meta = json.loads(md.read_text(encoding="utf-8"))
    cfg_snap = meta.get("config_snapshot", {})

    # A2-G1-iv from PRIMARY EVIDENCE, not from the metadata summary: each
    # round is one PPO rollout of `rollout_length` steps, and each round's
    # own source cost is in rounds.jsonl. Both are independent of whatever
    # a resume may have written into env_steps.
    rollout = int(cfg_snap.get("rollout_length", 0))
    if k * rollout != EXPECTED_PPO_STEPS:
        return False, (f"A2-G1-iv: derived PPO env steps {k * rollout} "
                       f"({k} rounds x {rollout}) != {EXPECTED_PPO_STEPS}")
    src_steps = 0
    with (run_dir / "rounds.jsonl").open(encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                src_steps += int((json.loads(ln).get("source_batch") or {}).get("env_steps", 0))
    if src_steps != EXPECTED_SOURCE_STEPS:
        return False, f"A2-G1-iv: derived source env steps {src_steps} != {EXPECTED_SOURCE_STEPS}"

    audit = meta.get("source_seed_audit", {})
    if audit.get("duplicate_seed_events", -1) != 0:
        return False, f"A2-G1-iii: {audit.get('duplicate_seed_events')} duplicate source seeds"
    if audit.get("n_env_seeds_issued") != EXPECTED_ENV_SEEDS:
        return False, (f"A2-G1-iii: {audit.get('n_env_seeds_issued')} env seeds issued, "
                       f"expected {EXPECTED_ENV_SEEDS}")

    # A2-G2-i / section 7 item 3: the run really carried the treatment its
    # condition claims, and the mapped source really was the attacked one.
    atk = cfg_snap.get("attack", {})
    if cond == "C":
        if atk.get("corrupted_source_ids") != [ATTACKED[seed]]:
            return False, (f"section 3 mapping violated: run corrupted "
                           f"{atk.get('corrupted_source_ids')}, expected [{ATTACKED[seed]!r}]")
        if atk.get("name") != "primary" or atk.get("budget_ratio") != 0.5:
            return False, f"A2-G2-i: attack block is not the A1 primary attack: {atk}"
    else:
        if atk.get("name") != "none" or atk.get("f") != 0:
            return False, f"A2-G2-iii: condition E must be clean, got {atk}"

    dfn = cfg_snap.get("defense", {})
    if dfn.get("name") != "rce" or dfn.get("f") != 1 or dfn.get("beta") != 1.5:
        return False, f"A2-G1-i: defense is not the declared RCE block: {dfn}"

    # A2-G1-v: the disclosed calibration phase produced the constant it is
    # mathematically obliged to produce. Anything else means the audited
    # degeneracy does not hold and the section 4 analysis is wrong.
    calib = run_dir / "guarantee_calibration.json"
    if not calib.exists():
        return False, "A2-G1-v: guarantee_calibration.json missing from an RCE run"
    eps = json.loads(calib.read_text(encoding="utf-8")).get("epsilon_offline")
    if eps is None or abs(float(eps) - SIGMA_MIN) > 1e-12:
        return False, (f"A2-G1-v: epsilon_offline={eps}, expected exactly sigma_min="
                       f"{SIGMA_MIN}; the section 4 degeneracy analysis does not hold")

    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="run just this one, e.g. C_seed0")
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

    # A2-G1-i / A2-G2-i, once, before anything runs.
    print(f"[{now()}] A2-G1-i: verifying the configs differ from their frozen "
          f"references only in the defense block")
    if not args.dry_run:
        rc = subprocess.call([PY, str(ROOT / "scripts/a2_verify_frozen.py")], cwd=str(ROOT))
        if rc != 0:
            state["halted"] = "A2-G1-i failed: configs differ beyond the defense block"
            write_status(state)
            print("HALT:", state["halted"])
            return 1

    for cond, seed in QUEUE:
        name = f"{cond}_seed{seed}"
        if args.only and args.only != name:
            continue
        run_dir = OUT_ROOT / name
        cfg = ROOT / f"configs/experiment/a2/{cond.lower()}_seed{seed}.yaml"
        log = log_dir / f"{name}.log"

        prev = state["runs"].get(name, {})
        if prev.get("gates_pass") is True:
            print(f"[{now()}] {name}: already complete and gated, skipping")
            continue

        done = rounds_done(run_dir)

        # Never re-enter a finished run (the A1 lesson): resuming a complete
        # run trains zero rounds and still rewrites run_metadata.json with
        # THIS invocation's env_steps and a near-zero timing block, which is
        # what clobbered A1's D_seed0 metadata. A complete run needs gating,
        # not training. For an RCE run this also avoids paying 0.6 h for a
        # calibration phase whose only consumer is a logged Boolean.
        if done >= EXPECTED_ROUNDS:
            print(f"[{now()}] {name}: already at {done} rounds -- gating only, "
                  f"not re-entering training")
            ok, why = run_gates(cond, seed, run_dir, log)
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
        pairing = ATTACK_DIR[seed].name if cond == "C" else CLEAN_DIR[seed].name
        print(f"[{now()}] {name}: launching{note}  cfg={cfg.name}  paired-with={pairing}"
              + (f"  attacked={ATTACKED[seed]}" if cond == "C" else "  (clean)"))
        state["runs"][name] = {"started": now(), "config": str(cfg.relative_to(ROOT)),
                               "condition": cond, "seed": seed,
                               "attacked_source": ATTACKED[seed] if cond == "C" else None,
                               "paired_no_rce_run": str(
                                   (ATTACK_DIR[seed] if cond == "C" else CLEAN_DIR[seed])
                                   .relative_to(ROOT)),
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
            print("HALT: a section 7 stop condition fired. Fix the implementation "
                  "before continuing.")
            return 1

    state["finished"] = now()
    write_status(state)
    print(f"[{now()}] A2 queue complete. Analyse with: python scripts/analyze_a2.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
