#!/usr/bin/env python
"""Assert that two runs produced numerically identical `rounds.jsonl`.

Written to answer one question: did a code change move an experiment that
is already frozen? Point it at an artifact produced BEFORE the change and
one produced after, from the same config and seed, and it either proves
they agree exactly or names the first field that does not.

Comparison is exact, not approximate. Floats are compared by their repr,
so `0.1 + 0.2` and `0.30000000000000004` do not quietly pass as equal.

Wall-clock fields are excluded and nothing else is. They are the only
values in a round record that are *expected* to vary between two identical
runs, and they are listed explicitly below rather than matched by a
pattern, so a future field that happens to contain "time" in its name
cannot silently drop out of the comparison.

Usage:
    python scripts/compare_runs_identical.py --before RUN_A --after RUN_B
    python scripts/compare_runs_identical.py --before A --after B \
        --allow-new-keys pid     # fields the change is EXPECTED to add
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Values that legitimately differ between two identical runs.
VOLATILE = frozenset({"wall_clock_s", "reference_wall_clock_s"})


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def flatten(obj, prefix: str = "") -> dict[str, str]:
    """Every leaf, addressed by path, rendered as an exact string."""
    out: dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in VOLATILE:
                continue
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = repr(obj)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True, type=Path)
    ap.add_argument("--after", required=True, type=Path)
    ap.add_argument("--allow-new-keys", nargs="*", default=[],
                    help="Leaf-path segments the AFTER run may add. Use for fields a change "
                         "deliberately introduces; anything else new is still a failure.")
    args = ap.parse_args()

    a = read_jsonl(args.before / "rounds.jsonl")
    b = read_jsonl(args.after / "rounds.jsonl")

    print(f"  before: {args.before}  ({len(a)} rounds)")
    print(f"  after : {args.after}  ({len(b)} rounds)")

    if len(a) != len(b):
        print(f"  [FAIL] round count differs: {len(a)} vs {len(b)}")
        sys.exit(1)

    allowed = tuple(args.allow_new_keys)
    n_compared = 0
    mismatches: list[str] = []
    added: list[str] = []
    removed: list[str] = []

    for k, (ra, rb) in enumerate(zip(a, b)):
        fa, fb = flatten(ra), flatten(rb)
        for path in sorted(set(fa) | set(fb)):
            in_a, in_b = path in fa, path in fb
            if in_a and in_b:
                n_compared += 1
                if fa[path] != fb[path]:
                    mismatches.append(f"round {k}: {path}: {fa[path]} -> {fb[path]}")
            elif in_b:
                if not any(f".{seg}." in f".{path}." for seg in allowed):
                    added.append(f"round {k}: {path}")
            else:
                removed.append(f"round {k}: {path}")

    ok = not mismatches and not added and not removed
    print(f"  leaves compared: {n_compared}")
    print(f"  mismatched: {len(mismatches)}   unexpectedly added: {len(added)}   removed: {len(removed)}")
    for line in (mismatches + added + removed)[:15]:
        print(f"     {line}")
    if len(mismatches + added + removed) > 15:
        print(f"     ... and {len(mismatches + added + removed) - 15} more")

    print("  [PASS] the two runs are numerically identical" if ok
          else "  [FAIL] the runs differ -- the code change moved the experiment")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
