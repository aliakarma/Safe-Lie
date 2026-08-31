#!/usr/bin/env python
"""The M >= 2f+1 deployment checklist (Phase 10's SourceAuditor).

Usage:
    python scripts/audit_sources.py --config configs/experiment/pilot_C_rce.yaml --assumed-f 1
    python scripts/audit_sources.py --preset m5_two_agent --assumed-f 2

Report reference: PROJECT_REPORT.md §2.6 -- "the most transferable
artifact ... count your genuinely independent reporting sources M and
your tolerable compromised-source count f; if M < 2f+1, your constraint
enforcement has no robustness guarantee at all." This counts independence
classes, not raw report IDs (W4) -- see safelie.governance.auditor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safelie.governance.auditor import audit_sources, format_audit_report  # noqa: E402
from safelie.sources.registry import default_m5_sources, default_m7_sources  # noqa: E402
from safelie.utils.config import load_experiment_config  # noqa: E402

PRESETS = {
    "m7_primary": default_m7_sources,
    "m5_two_agent": default_m5_sources,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="Path to an experiment config; audits its `sources` section")
    group.add_argument("--preset", choices=list(PRESETS), help="Audit one of the paper's named source configurations")
    parser.add_argument("--assumed-f", type=int, required=True, help="Assumed number of compromised sources")
    args = parser.parse_args()

    if args.config:
        cfg = load_experiment_config(args.config)
        sources = cfg.sources
        label = cfg.run_id
    else:
        sources = PRESETS[args.preset]()
        label = args.preset

    result = audit_sources(sources, assumed_f=args.assumed_f)
    print(format_audit_report(result, label=label))
    sys.exit(0 if result.passes else 1)


if __name__ == "__main__":
    main()
