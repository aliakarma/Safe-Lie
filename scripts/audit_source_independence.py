#!/usr/bin/env python
"""Measured source diversity for a COMPLETED run (P0 #9).

Usage:
    python scripts/audit_source_independence.py --run-dir results/runs/pilot_A_clean_seed0 \
        --config configs/experiment/pilot_A_clean.yaml

Complements, and never replaces, `scripts/audit_sources.py`. That script
checks a CONFIG's *declared* independence classes before any round runs
(`safelie.governance.auditor`); this one checks what a run's own logs say
actually happened: are the M sources' errors (against the withheld
oracle's true cost) behaving as if they were independent, or are several
of them moving together regardless of what their independence-class
labels claim?

Writes a machine-readable JSON report (`<run-dir>/source_diversity.json`
by default) and prints a human-readable summary. Exits 1 if
`assumption_1ii_supported` is False, so this can gate a pipeline the same
way `audit_sources.py` does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safelie.governance.empirical_diversity import (  # noqa: E402
    compute_empirical_diversity,
    format_report,
    write_report,
)
from safelie.sources.registry import effective_M  # noqa: E402
from safelie.utils.config import load_experiment_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="Path to a completed run's output directory")
    ap.add_argument("--config", help="The experiment config the run used (for source_type/independence_class labels)")
    ap.add_argument("--tolerance", type=float, default=0.5, help="Max (nominal_effective_M - measured) to still call Assumption 1(ii) supported")
    ap.add_argument("--out", help="Where to write the JSON report (default: <run-dir>/source_diversity.json)")
    args = ap.parse_args()

    source_specs = None
    nominal_effective_m = None
    if args.config:
        cfg = load_experiment_config(args.config)
        source_specs = {s.source_id: (s.source_type, s.independence_class) for s in cfg.sources.sources}
        nominal_effective_m = effective_M(cfg.sources.sources)

    report = compute_empirical_diversity(
        args.run_dir,
        nominal_effective_m=nominal_effective_m,
        tolerance=args.tolerance,
        source_specs=source_specs,
    )
    out_path = Path(args.out) if args.out else Path(args.run_dir) / "source_diversity.json"
    write_report(report, out_path)

    print(format_report(report))
    print(f"\nMachine-readable report written to {out_path}")
    return 0 if report.assumption_1ii_supported else 1


if __name__ == "__main__":
    raise SystemExit(main())
