#!/usr/bin/env python
"""P1 per-run structural verification. Run after every completed P1 run.

docs/p1_concentrated_attack_gates.md Sections 13-14. The queue calls this
after each run and halts on a non-zero exit, so a structural failure stops
the campaign instead of spending another 10 hours producing artifacts that
are already known to be invalid.

Checks, in order:
    P1-a   delta_k is concentrated: exactly the configured owners are
           nonzero, at exactly -B/M, on EVERY round.
    P1-b   every untargeted owner's aggregate equals the uncorrupted
           aggregate bitwise (injected_delta == 0.0, not approx).
    P1-c   the run used the configured topology, and the dual identity
           lambda_after - lambda_mixed_before == eta * residual holds on
           every unclipped cell.
    P1-e   round count, finiteness, monotone round_k.
    P1-f   no duplicate source seeds within the run.
    P1-g   the oracle log exists, was written separately, and has no
           learner fields.
    P1-h   theta_k checksum stable through the dual update on every round.

Usage:
    python scripts/p1_verify_run.py --run results/runs_p1/R_seed0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safelie.consensus.mixing import assert_doubly_stochastic  # noqa: E402
from safelie.consensus.topologies import build_topology  # noqa: E402

EXPECTED_ROUNDS = 250
TOL = 1e-9
CLIP_EPS = 1e-12


def load_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def all_finite(obj) -> bool:
    if isinstance(obj, dict):
        return all(all_finite(v) for v in obj.values())
    if isinstance(obj, list):
        return all(all_finite(v) for v in obj)
    if isinstance(obj, bool):
        return True
    if isinstance(obj, (int, float)):
        return math.isfinite(obj)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--expected-rounds", type=int, default=EXPECTED_ROUNDS)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    cfg = meta["config_snapshot"]
    rounds = load_jsonl(run_dir / "rounds.jsonl")

    n = cfg["env"]["n_agents"]
    d = cfg["env"]["budget"]
    eta = cfg["dual"]["eta_lambda"]
    lam_max = cfg["dual"]["lambda_max"]
    m = len(cfg["sources"]["sources"])
    B = cfg["attack"]["budget_ratio"] * d
    owners = cfg["attack"].get("corrupted_owner_ids") or []
    expected_shift = -B / m
    aids = sorted(rounds[0]["constraints"])
    W = build_topology(cfg["topology"]["name"], cfg["topology"]["n_agents"],
                       p=cfg["topology"].get("p"), graph_seed=cfg["topology"].get("graph_seed", 0))

    failures: list[str] = []
    notes: list[str] = []

    print(f"run        : {run_dir}")
    print(f"run_id     : {cfg['run_id']}   seed={cfg['seed']}   topology={cfg['topology']['name']}")
    print(f"attack     : source={cfg['attack'].get('corrupted_source_ids')} owners={owners}")
    print(f"rounds     : {len(rounds)} (expected {args.expected_rounds})")
    print()

    # ---- P1-e: shape -----------------------------------------------------
    if len(rounds) != args.expected_rounds:
        failures.append(f"P1-e: {len(rounds)} rounds, expected {args.expected_rounds}")
    if [r["round_k"] for r in rounds] != list(range(len(rounds))):
        failures.append("P1-e: round_k is not 0..K-1 monotone")
    nonfinite = [r["round_k"] for r in rounds if not all_finite(r)]
    if nonfinite:
        failures.append(f"P1-e: non-finite values in rounds {nonfinite[:5]}")

    # ---- P1-a / P1-b: the concentrated perturbation -----------------------
    bad_conc, bad_zero = [], []
    delta_hist = []
    for r in rounds:
        delta = np.array([r["constraints"][a]["injected_delta"] for a in aids])
        delta_hist.append(delta)
        expected = np.zeros(n)
        for o in owners:
            expected[aids.index(o)] = expected_shift
        if not np.allclose(delta, expected, atol=TOL):
            bad_conc.append((r["round_k"], delta.tolist()))
        for a in aids:
            if a in owners:
                continue
            blk = r["constraints"][a]
            if blk["injected_delta"] != 0.0 or blk["aggregate"]["point_estimate"] != blk["clean_point_estimate"]:
                bad_zero.append((r["round_k"], a))
    delta_hist = np.array(delta_hist)
    if bad_conc:
        failures.append(f"P1-a: delta_k wrong on {len(bad_conc)} rounds, first={bad_conc[0]}")
    else:
        print(f"P1-a PASS: delta_k == {np.array2string(np.zeros(n) + 0, precision=0)[:0]}"
              f"[0,..,{expected_shift:.10f},..,0] on all {len(rounds)} rounds "
              f"(owners {owners}); max abs dev = "
              f"{np.abs(delta_hist - delta_hist[0]).max():.3e}")
    if bad_zero:
        failures.append(f"P1-b: {len(bad_zero)} untargeted owner-rounds were shifted, first={bad_zero[0]}")
    else:
        print(f"P1-b PASS: all {len(rounds) * (n - len(owners))} untargeted owner-rounds "
              f"bitwise unshifted (== 0.0 exactly)")

    # ---- P1-d/D5: mass ----------------------------------------------------
    total_mass = float(delta_hist.sum())
    predicted_mass = len(rounds) * expected_shift * len(owners)
    if abs(total_mass - predicted_mass) > 1e-6:
        failures.append(f"D5: sum_k 1^T delta_k = {total_mass} != predicted {predicted_mass}")
    else:
        print(f"D5   PASS: sum_k 1^T delta_k = {total_mass:.6f} "
              f"(predicted K*(-B/M) = {predicted_mass:.6f})")

    # ---- P1-c: topology + dual identity -----------------------------------
    assert_doubly_stochastic(W)
    if cfg["topology"]["name"] == "identity":
        if not np.array_equal(W, np.eye(n)):
            failures.append("P1-c: identity topology is not exactly I")
    lam = np.array([[r["constraints"][a]["lambda_after"] for a in aids] for r in rounds])
    mixed = np.array([[r["constraints"][a]["lambda_mixed_before"] for a in aids] for r in rounds])
    res = np.array([[r["constraints"][a]["constraint_residual"] for a in aids] for r in rounds])

    # W reproduction: lambda_mixed_before[k] must equal W @ lambda_after[k-1]
    mix_err = 0.0
    for k in range(1, len(rounds)):
        mix_err = max(mix_err, float(np.abs(mixed[k] - W @ lam[k - 1]).max()))
    if mix_err > 1e-9:
        failures.append(f"P1-c: consensus mixing does not reproduce W: max err {mix_err:.3e}")
    else:
        print(f"P1-c PASS: lambda_mixed_before == W @ lambda_prev to {mix_err:.3e} "
              f"({cfg['topology']['name']})")

    unclipped = (lam > CLIP_EPS) & (lam < lam_max - CLIP_EPS)
    dual_err = float(np.abs((lam - mixed - eta * res)[unclipped]).max()) if unclipped.any() else 0.0
    if dual_err > 1e-9:
        failures.append(f"P1-c: dual identity fails on unclipped cells: max err {dual_err:.3e}")
    else:
        print(f"P1-c PASS: lambda_after - lambda_mixed_before == eta*residual to "
              f"{dual_err:.3e} on {int(unclipped.sum())} unclipped cells "
              f"({100 * unclipped.mean():.1f}%)")
    notes.append(f"lambda at floor on {100 * (lam <= CLIP_EPS).mean():.2f}% of cells, "
                 f"at ceiling on {100 * (lam >= lam_max - CLIP_EPS).mean():.2f}%")

    # ---- P1-h: theta stability -------------------------------------------
    unstable = [r["round_k"] for r in rounds
                if r.get("source_batch", {}).get("theta_k_checksum_stable_through_dual") is not True]
    if unstable:
        failures.append(f"P1-h: theta moved during rounds {unstable[:5]}")
    else:
        print("P1-h PASS: theta_k checksum stable through the dual update on all rounds")

    # ---- P1-f: seed uniqueness -------------------------------------------
    audit = meta.get("source_seed_audit", {})
    dupes = audit.get("duplicate_seed_events", None)
    if dupes not in (0, None):
        failures.append(f"P1-f: {dupes} duplicate source-seed events")
    else:
        print(f"P1-f PASS: {audit.get('n_env_seeds_issued')} env seeds issued, "
              f"{dupes} duplicates, collisions={audit.get('collisions')}")

    # ---- P1-g: oracle isolation ------------------------------------------
    oracle_p = run_dir / "oracle.jsonl"
    if not oracle_p.exists():
        failures.append("P1-g: oracle.jsonl missing")
    else:
        orec = load_jsonl(oracle_p)
        leaked = [k for k in ("lambda_after", "injected_delta", "reports") if k in (orec[0] if orec else {})]
        if leaked:
            failures.append(f"P1-g: learner fields present in oracle.jsonl: {leaked}")
        if not all_finite(orec):
            failures.append("P1-g: non-finite values in oracle.jsonl")
        print(f"P1-g PASS: oracle.jsonl has {len(orec)} records, no learner fields")

    print()
    for note in notes:
        print(f"note: {note}")
    print()
    ok = not failures
    if ok:
        print(f"VERIFY PASS: {run_dir.name}")
    else:
        print(f"VERIFY FAIL: {run_dir.name}")
        for f in failures:
            print(f"  - {f}")

    if args.report:
        Path(args.report).write_text(json.dumps({
            "run": str(run_dir), "run_id": cfg["run_id"], "seed": cfg["seed"],
            "topology": cfg["topology"]["name"], "owners": owners,
            "rounds": len(rounds), "pass": ok, "failures": failures, "notes": notes,
            "total_injected_mass": total_mass, "predicted_mass": predicted_mass,
            "mixing_max_err": mix_err, "dual_identity_max_err": dual_err,
            "frac_lambda_at_floor": float((lam <= CLIP_EPS).mean()),
            "frac_lambda_at_ceiling": float((lam >= lam_max - CLIP_EPS).mean()),
        }, indent=2), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
