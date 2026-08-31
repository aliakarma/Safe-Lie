"""Preflight: the CRITICAL smoke subset, runnable on Colab (or anywhere).

Report reference: PROJECT_REPORT.md §R7.3, cell 5 — "python -m
safelie.preflight -- runs the CRITICAL smoke subset (S1-S12, S16-S28) on
Colab, so that a laptop-vs-Colab environment difference cannot silently
invalidate the run. Training does not start unless preflight passes."

This module is a thin, importable wrapper around the same pytest suites
`scripts/smoke_test.py` runs locally (`tests/unit`, `tests/property`,
`tests/theory`, `tests/isolation`), so that the exact same tests that
gate local development also gate the Colab runtime before any GPU time
is spent. It intentionally excludes `tests/integration` (config-loading
convenience tests, not part of the CRITICAL gate) and `tests/smoke`'s
determinism/end-to-end suite when `--fast` is passed, since those are
slower and already covered locally.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CRITICAL_SUITES = ["tests/unit", "tests/property", "tests/theory", "tests/isolation"]
FULL_SUITES = CRITICAL_SUITES + ["tests/smoke"]


def run_preflight(fast: bool = False, repo_root: Path | None = None) -> int:
    root = repo_root or Path(__file__).resolve().parents[2]
    suites = CRITICAL_SUITES if fast else FULL_SUITES
    cmd = [sys.executable, "-m", "pytest", *suites, "-q"]
    print(f"[safelie.preflight] running: {' '.join(cmd)} (cwd={root})")
    result = subprocess.run(cmd, cwd=root)
    if result.returncode == 0:
        print("[safelie.preflight] PASS -- safe to proceed to training.")
    else:
        print("[safelie.preflight] FAIL -- do not start training on this runtime.")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="Skip tests/smoke (slower determinism/e2e checks)")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    args = parser.parse_args()

    if args.dry_run:
        suites = CRITICAL_SUITES if args.fast else FULL_SUITES
        print(f"[safelie.preflight] dry-run: would execute pytest over {suites}")
        sys.exit(0)

    sys.exit(run_preflight(fast=args.fast))


if __name__ == "__main__":
    main()
