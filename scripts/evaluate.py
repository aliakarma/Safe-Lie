#!/usr/bin/env python
"""Regenerate a MEASURED summary table from one or more completed runs.

Usage:
    python scripts/evaluate.py --run clean=results/runs/local_demo_clean \\
                                --run attack=results/runs/local_demo_attack \\
                                --run rce=results/runs/local_demo_rce \\
                                --budget 25.0

This never reads or compares against main_iclr.tex's projected numbers
(see safelie.analysis.tables' module docstring).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safelie.analysis.tables import build_summary_table  # noqa: E402


def _parse_run(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected label=path, got '{spec}'")
    label, path = spec.split("=", 1)
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", type=_parse_run, required=True, dest="runs",
                         help="label=path_to_run_output_dir, repeatable")
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--last-n-rounds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write the table to")
    args = parser.parse_args()

    run_dirs = dict(args.runs)
    table = build_summary_table(run_dirs, budget=args.budget, last_n_rounds=args.last_n_rounds)
    print(table)
    if args.out:
        args.out.write_text(table, encoding="utf-8")
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
