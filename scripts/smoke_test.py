#!/usr/bin/env python
"""The GREEN SIGNAL gate (PROJECT_REPORT.md §R5).

Runs every CRITICAL local test (unit, property, theory, smoke, isolation)
and reports GREEN or RED against the eight conditions G-1..G-8. This is a
technical-readiness gate, not evidence about the hypothesis -- see the
module's own printed banner and §R5's own warning against conflating the
two.

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TEST_DIRS = ["tests/unit", "tests/property", "tests/theory", "tests/smoke", "tests/isolation"]

CONDITIONS = [
    ("G-1", "All unit, property, and theory tests pass", "tests/unit, tests/property, tests/theory"),
    ("G-2", "No critical implementation failures remain", "pytest exit code across all suites"),
    ("G-3", "Oracle / true-cost isolation passes", "tests/isolation"),
    ("G-4", "Attack changes the safety-feedback pathway in the intended direction", "tests/property/test_attacks.py"),
    ("G-5", "RCE behaves in the intended qualitative direction", "tests/property/test_aggregators.py, test_dual_update.py"),
    ("G-6", "Clean training completes end-to-end locally at tiny scale", "tests/smoke/test_end_to_end.py"),
    ("G-7", "Results reproduce across repeated local runs", "tests/smoke/test_determinism.py"),
    ("G-8", "No unresolved specification gap makes the pilot scientifically ambiguous", "manual: docs/decisions/ + docs/assumptions.md"),
]


def main() -> int:
    print("=" * 78)
    print("GREEN SIGNAL GATE -- technical readiness only, NOT evidence about the")
    print("hypothesis (PROJECT_REPORT.md §R5). 'All numbers look good' is not the")
    print("criterion.")
    print("=" * 78)

    cmd = [sys.executable, "-m", "pytest", *TEST_DIRS, "-q"]
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    passed = result.returncode == 0

    print("\n" + "-" * 78)
    print(f"{'Condition':<6}{'Description':<70}")
    print("-" * 78)
    for cid, desc, source in CONDITIONS:
        status = "PASS" if passed else "CHECK"
        if cid == "G-8":
            status = "MANUAL"  # this one is not automatable; see docs/decisions/
        print(f"{cid:<6}{desc}")
        print(f"      -> {source} [{status}]")

    print("-" * 78)
    if passed:
        print(
            "\nAll automated CRITICAL tests pass. G-8 (specification-gap closure) is a "
            "manual review -- confirm every entry in docs/decisions/ and "
            "docs/assumptions.md is current before declaring GREEN.\n"
        )
        print("RESULT: technical-readiness checks GREEN (pending manual G-8 review).")
        return 0
    print("\nRESULT: RED. Do not proceed past this gate. Fix the failing test(s) above,")
    print("re-run the *whole* suite, and re-evaluate (PROJECT_REPORT.md §R5.3).\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
