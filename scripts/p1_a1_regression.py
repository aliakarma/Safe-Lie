#!/usr/bin/env python
"""P1 reproducibility regression: the A1 code path must not have moved.

docs/p1_concentrated_attack_gates.md Section 10.

P1 adds one field to `AttackConfig` and one owner-scoped branch to
`safelie.training.loop`. Both are supposed to be inert when the field is
absent, which is the case for every A1/A2/A3/U/dose config ever run. This
script proves that behaviourally rather than by reading the diff: it
re-runs the FIRST `--rounds` rounds of a committed A1 production config
with today's code and compares every numeric field of the resulting round
records against the committed `rounds.jsonl`, bit for bit.

It never writes into a production directory: output goes to `--out`, which
must not be under `results/runs_a1`. The committed artifact is opened
read-only.

Usage:
    python scripts/p1_a1_regression.py --config configs/experiment/a1/b_seed0.yaml \
        --reference results/runs_a1/B_seed0 --rounds 2 --out <scratch dir>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safelie.training.loop import ExperimentRun  # noqa: E402
from safelie.utils.config import load_experiment_config  # noqa: E402

# Fields P1 adds. They cannot exist in a pre-P1 artifact, so their presence
# in the fresh record is expected and their absence from the reference is
# not a mismatch. Everything else must match exactly.
P1_NEW_FIELDS = {"injected_delta", "owner_targeted", "clean_point_estimate"}


def flatten(obj, prefix=""):
    """Every leaf of a round record as {dotted path: value}."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}{i}."))
    else:
        out[prefix.rstrip(".")] = obj
    return out


def compare(fresh: dict, ref: dict) -> list[str]:
    a, b = flatten(fresh), flatten(ref)
    problems = []
    for key in sorted(set(a) | set(b)):
        leaf = key.rsplit(".", 1)[-1]
        if leaf in P1_NEW_FIELDS:
            if key not in a:
                problems.append(f"{key}: P1 field missing from the fresh run")
            continue
        if key not in a:
            problems.append(f"{key}: present in reference, absent from fresh run")
            continue
        if key not in b:
            problems.append(f"{key}: NEW field not declared in P1_NEW_FIELDS (schema drift)")
            continue
        x, y = a[key], b[key]
        if isinstance(x, float) or isinstance(y, float):
            # Bitwise: `==` on floats, with NaN treated as equal to NaN.
            if not (x == y or (isinstance(x, float) and isinstance(y, float)
                               and math.isnan(x) and math.isnan(y))):
                problems.append(f"{key}: fresh={x!r} reference={y!r} (delta={x - y!r})")
        elif x != y:
            problems.append(f"{key}: fresh={x!r} reference={y!r}")
    return problems


# Wall-clock and scheduling fields are machine-dependent, not results.
VOLATILE_PREFIXES = (
    "source_batch.wall_clock_s",
    "source_batch.reference_wall_clock_s",
    "source_batch.n_chunks",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    if "runs_a1" in out.parts or "runs_a2" in out.parts or "runs_a3" in out.parts:
        print(f"REFUSING: --out {out} is inside a production run tree.", file=sys.stderr)
        return 2

    ref_file = Path(args.reference) / "rounds.jsonl"
    reference = []
    with ref_file.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= args.rounds:
                break
            reference.append(json.loads(line))
    if len(reference) < args.rounds:
        print(f"Reference has only {len(reference)} rounds", file=sys.stderr)
        return 2

    cfg = load_experiment_config(args.config)
    cfg = cfg.model_copy(update={"output_dir": str(out), "run_id": "p1_a1_regression"})
    print(f"config    : {args.config}")
    print(f"reference : {ref_file}")
    print(f"rounds    : {args.rounds}")
    print(f"attack    : name={cfg.attack.name} f={cfg.attack.f} "
          f"sources={cfg.attack.corrupted_source_ids} owners={cfg.attack.corrupted_owner_ids}")
    assert cfg.attack.corrupted_owner_ids is None, (
        "This regression is only meaningful on a config that does NOT set the new field."
    )

    run = ExperimentRun(cfg)
    try:
        fresh = [run.run_round() for _ in range(args.rounds)]
    finally:
        run.close()

    total = 0
    for k, (f, r) in enumerate(zip(fresh, reference, strict=True)):
        problems = [p for p in compare(f, r)
                    if not p.startswith(VOLATILE_PREFIXES)]
        total += len(problems)
        status = "IDENTICAL" if not problems else f"{len(problems)} MISMATCHES"
        print(f"  round {k}: {status}")
        for p in problems[:20]:
            print(f"      {p}")

    print()
    if total == 0:
        print(f"PASS: {args.rounds} rounds reproduce the committed A1 artifact bitwise.")
        return 0
    print(f"FAIL: {total} mismatching fields.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
