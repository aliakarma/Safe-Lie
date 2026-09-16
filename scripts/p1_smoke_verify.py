#!/usr/bin/env python
"""P1 pre-run smoke gate: inspect the ACTUAL per-owner perturbation vector.

docs/p1_concentrated_attack_gates.md Section 9, gate P1-SMOKE.

This is the gate the whole campaign turns on. It runs real rounds of a real
P1 production config and prints delta_k, the vector whose i-th entry is the
shift the i-th constraint owner's dual coordinate actually received, measured
as `post-attack aggregate - same-aggregator-on-uncorrupted-reports`.

It PASSES only if that vector is concentrated,

    delta_k ~ [0, ..., -B/M, ..., 0]

and FAILS if it is uniform,

    delta_k ~ [-B/M, -B/M, ..., -B/M]

which is what A1/A2/A3 actually ran (measured on the committed A1 logs at
-4.1666... on all six coordinates, to 9.8e-15).

Nothing here is written to a production directory.

Usage:
    python scripts/p1_smoke_verify.py --config configs/experiment/p1/r_seed0.yaml \
        --rounds 1 --out <scratch dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safelie.eval.monitor import concentration_ratio  # noqa: E402
from safelie.training.loop import ExperimentRun  # noqa: E402
from safelie.utils.config import load_experiment_config  # noqa: E402

TOL = 1e-9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None, help="Optional JSON report path")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    if any(part.startswith("runs_") and part != "runs_p1_smoke" for part in out.parts):
        print(f"REFUSING: --out {out} looks like a production run tree.", file=sys.stderr)
        return 2

    cfg = load_experiment_config(args.config)
    cfg = cfg.model_copy(update={"output_dir": str(out), "run_id": "p1_smoke"})

    n = cfg.env.n_agents
    m = len(cfg.sources.sources)
    B = cfg.attack.budget_ratio * cfg.env.budget
    expected_shift = -B / m
    owners = cfg.attack.corrupted_owner_ids or []

    print("=" * 74)
    print("P1-SMOKE: per-owner perturbation vector on the real production config")
    print("=" * 74)
    print(f"config            : {args.config}")
    print(f"topology          : {cfg.topology.name}   N={n}")
    print(f"attack            : {cfg.attack.name}, f={cfg.attack.f}, "
          f"direction={cfg.attack.direction}, support={cfg.attack.support}")
    print(f"corrupted source  : {cfg.attack.corrupted_source_ids}")
    print(f"corrupted owners  : {owners}")
    print(f"B = {cfg.attack.budget_ratio} * d = {B}    M = {m}")
    print(f"predicted shift on a targeted owner : -B/M = {expected_shift:.10f}")
    print("predicted shift on every other owner: 0.0 exactly")
    print()

    run = ExperimentRun(cfg)
    try:
        records = [run.run_round() for _ in range(args.rounds)]
    finally:
        run.close()

    aids = sorted(records[0]["constraints"])
    target_idx = [aids.index(o) for o in owners]
    ok = True
    rows = []

    for rec in records:
        k = rec["round_k"]
        delta = np.array([rec["constraints"][a]["injected_delta"] for a in aids])
        conc = float(concentration_ratio(delta)[0])
        nonzero = np.flatnonzero(np.abs(delta) > 1e-12).tolist()

        expected = np.zeros(n)
        for i in target_idx:
            expected[i] = expected_shift

        concentrated = np.allclose(delta, expected, atol=TOL)
        uniform = bool(np.allclose(delta, np.full(n, delta[0]), atol=TOL) and abs(delta[0]) > TOL)
        exact_zero_off_target = all(
            rec["constraints"][a]["injected_delta"] == 0.0
            for i, a in enumerate(aids) if i not in target_idx
        )

        print(f"round {k}:")
        print(f"  delta_k            = {np.array2string(delta, precision=10, suppress_small=False)}")
        print(f"  nonzero coordinates= {nonzero}   (expected {target_idx})")
        print(f"  concentration      = {conc:.6f}   (1.0 = one coordinate, {1/n:.4f} = uniform)")
        print(f"  off-target exactly zero (==0.0, not approx): {exact_zero_off_target}")
        print(f"  matches [0,..,-B/M,..,0] : {concentrated}")
        print(f"  matches [-B/M,..,-B/M]   : {uniform}   <-- must be False")
        if not concentrated or uniform or not exact_zero_off_target:
            ok = False
        rows.append({"round_k": k, "delta": delta.tolist(), "concentration": conc,
                     "nonzero": nonzero, "concentrated": bool(concentrated),
                     "uniform": uniform, "off_target_exactly_zero": exact_zero_off_target})

        # Cross-check against the untouched report stream: the targeted owner
        # must be the only one whose aggregate moved at all.
        for a in aids:
            blk = rec["constraints"][a]
            moved = blk["aggregate"]["point_estimate"] != blk["clean_point_estimate"]
            flag = "CORRUPTED" if a in owners else "clean"
            print(f"    {a:8s} [{flag:9s}] aggregate_moved={str(moved):5s} "
                  f"delta={blk['injected_delta']:+.10f} "
                  f"reported_sources={[round(r['value'], 4) for r in blk['reports']]}")
        print()

    print("=" * 74)
    print("P1-SMOKE: PASS" if ok else "P1-SMOKE: FAIL")
    print("=" * 74)
    if args.report:
        Path(args.report).write_text(json.dumps(
            {"config": args.config, "topology": cfg.topology.name, "B": B, "M": m,
             "expected_shift": expected_shift, "owners": owners,
             "agent_ids": aids, "rounds": rows, "pass": ok}, indent=2), encoding="utf-8")
        print(f"report written to {args.report}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
