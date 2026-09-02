#!/usr/bin/env python
"""Run the pilot condition matrix as a priority-ordered job queue.

Usage:
    python scripts/run_matrix.py --plan overnight --workers 6
    python scripts/run_matrix.py --plan overnight --workers 6 --dry-run

Why a queue rather than a shell loop: the matrix is 20+ runs of ~80
minutes each, far longer than any one sitting, and the runs are not
equally valuable. `PROJECT_REPORT.md` §R8.3 is explicit that condition D
(the falsification control) must be read before condition C -- if the
benign control does not separate from the attacked condition, the defense
comparison is moot and the remaining runs are wasted. So jobs are emitted
in priority order and dispatched to a fixed worker pool: whenever the
queue is interrupted, what has completed is the most decisive subset
rather than an arbitrary one.

Each job is an independent `scripts/train.py` process pinned to one torch
thread (verified bitwise-identical to the torch default on this pipeline;
see that script's --threads help). Runs are resumable -- `train.py`
auto-resumes from `checkpoint.pt` -- so re-running this script after an
interruption continues rather than restarting, and completed runs are
skipped outright.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs" / "experiment"
RESULTS = REPO / "results" / "runs"
LOGS = REPO / "results" / "logs"

CONDITIONS = {
    "A": "pilot_A_clean",
    "B": "pilot_B_attack",
    "C": "pilot_C_rce",
    "D": "pilot_D_benign",
    "E": "pilot_E_clean_rce",
}

# Priority tiers. Earlier tiers finish first, so an interrupted night still
# answers the most important question it had time for.
PLANS: dict[str, list[list[tuple[str, int]]]] = {
    "overnight": [
        # 1. The §R8.3 falsification gate: bring the attacked condition and
        #    the benign control to n=3 so their separation can be judged
        #    against the clean condition's own seed spread.
        [("B", 1), ("B", 2), ("D", 1), ("D", 2)],
        # 2. The defense claim at n=3.
        [("C", 0), ("C", 1), ("C", 2), ("E", 0), ("E", 1), ("E", 2)],
        # 3-4. Power. The dual variable oscillates with a ~100-round period,
        #    so rounds within a run are heavily autocorrelated and seeds --
        #    not rounds -- are the independent samples. Extending every
        #    condition together keeps the design balanced.
        [(c, 3) for c in CONDITIONS],
        [(c, 4) for c in CONDITIONS],
    ],
    "gate": [[("B", 1), ("B", 2), ("D", 1), ("D", 2)]],
}


def run_dir(cond: str, seed: int) -> Path:
    return RESULTS / f"{CONDITIONS[cond]}_seed{seed}"


def is_complete(cond: str, seed: int, expected_rounds: int) -> bool:
    rounds = run_dir(cond, seed) / "rounds.jsonl"
    if not rounds.exists():
        return False
    with rounds.open(encoding="utf-8") as fh:
        return sum(1 for _ in fh) >= expected_rounds


def expected_rounds(cond: str) -> int:
    import yaml

    cfg = yaml.safe_load((CONFIG_DIR / f"{CONDITIONS[cond]}.yaml").read_text(encoding="utf-8"))
    return max(1, cfg["total_steps"] // cfg["rollout_length"])


def launch(cond: str, seed: int) -> subprocess.Popen:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{CONDITIONS[cond]}_seed{seed}.log"
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "train.py"),
        "--config", str(CONFIG_DIR / f"{CONDITIONS[cond]}.yaml"),
        "--seed", str(seed),
        "--threads", "1",
    ]
    fh = log.open("a", encoding="utf-8")
    fh.write(f"\n=== launched {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
    fh.flush()
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO))
    proc._safelie_job = (cond, seed)  # type: ignore[attr-defined]
    proc._safelie_log = fh  # type: ignore[attr-defined]
    return proc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default="overnight", choices=sorted(PLANS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status-file", default=str(REPO / "results" / "logs" / "matrix_status.json"))
    args = ap.parse_args()

    jobs = [job for tier in PLANS[args.plan] for job in tier]
    need = [j for j in jobs if not is_complete(*j, expected_rounds(j[0]))]

    print(f"plan={args.plan}  workers={args.workers}")
    print(f"{len(jobs)} jobs, {len(jobs) - len(need)} already complete, {len(need)} to run")
    for c, s in need:
        print(f"  {CONDITIONS[c]} seed {s}")
    if args.dry_run:
        return 0

    pending = list(need)
    running: list[subprocess.Popen] = []
    done: list[tuple[str, int, int]] = []
    started = time.time()

    def write_status() -> None:
        Path(args.status_file).write_text(
            json.dumps(
                {
                    "updated": datetime.now().isoformat(timespec="seconds"),
                    "elapsed_min": round((time.time() - started) / 60, 1),
                    "running": [
                        f"{CONDITIONS[p._safelie_job[0]]}_seed{p._safelie_job[1]}"  # type: ignore[attr-defined]
                        for p in running
                    ],
                    "pending": [f"{CONDITIONS[c]}_seed{s}" for c, s in pending],
                    "done": [f"{CONDITIONS[c]}_seed{s} rc={rc}" for c, s, rc in done],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    while pending or running:
        while pending and len(running) < args.workers:
            cond, seed = pending.pop(0)
            running.append(launch(cond, seed))
            print(f"[{datetime.now():%H:%M:%S}] launched {CONDITIONS[cond]} seed {seed}")
        write_status()

        time.sleep(20)
        for proc in list(running):
            if proc.poll() is None:
                continue
            cond, seed = proc._safelie_job  # type: ignore[attr-defined]
            proc._safelie_log.close()  # type: ignore[attr-defined]
            running.remove(proc)
            done.append((cond, seed, proc.returncode))
            state = "OK" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
            print(f"[{datetime.now():%H:%M:%S}] {CONDITIONS[cond]} seed {seed} {state}")

    write_status()
    failed = [(c, s) for c, s, rc in done if rc != 0]
    print(f"\nmatrix finished in {(time.time() - started) / 60:.1f} min")
    print(f"  completed: {len(done) - len(failed)}   failed: {len(failed)}")
    for c, s in failed:
        print(f"  FAILED: {CONDITIONS[c]} seed {s} -- see results/logs/{CONDITIONS[c]}_seed{s}.log")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
