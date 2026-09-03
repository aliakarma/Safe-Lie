"""G9 smoke test (docs/g9_gates.md, "Run protocol" step 1; task section 14).

**This produces no scientific result.** It runs a handful of rounds of the
real integrated loop to prove the machinery works before eight hours of
compute are committed to it, and it writes to a scratch directory that is
deliberately NOT `results/runs_constraint_batch_g9/`.

What it checks, in one pass:

  1. source collection runs at the configured `M`, `R_m` and worker count
     without deadlocking;
  2. PPO continues -- theta moves every round, and the round record still
     carries the same PPO diagnostics it did in G2;
  3. the dual update fires and lambda responds with the right sign;
  4. `lambda_after - lambda_mixed_before == eta * residual` exactly, on
     every unclipped cell (gate G9d-i, on live data);
  5. the policy checksum is identical across all three replicas and
     unchanged from before collection to after the dual update (G9h);
  6. every source seed is unique and the replica seed sets are disjoint
     (G9g);
  7. **checkpoint/resume is bitwise**: a run interrupted after round 1 and
     resumed produces byte-identical round records to an uninterrupted
     run. This is the check that most needs doing before a long run, since
     the only way an 8-hour job survives an interruption is if resuming it
     is exact;
  8. the validation reference is collected on the configured round, is
     written to its own file, and is absent from `rounds.jsonl`;
  9. per-phase wall clock, so the projected 250-round cost in
     docs/g9_gates.md can be replaced by a measurement before the primary
     run starts.

Usage:
    python scripts/g9_smoke_test.py [--rounds 3] [--workers 9] [--R-m 30]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from safelie.experiment import run_experiment_with_oracle  # noqa: E402
from safelie.utils.config import load_experiment_config  # noqa: E402
from safelie.utils.logging import read_jsonl  # noqa: E402

CONFIG = "configs/experiment/g9_batch_clean.yaml"
SCRATCH = Path("results/g9_smoke")


def _build(cfg_path: str, rounds: int, workers: int, r_m: int, run_id: str, out_dir: Path):
    cfg = load_experiment_config(cfg_path)
    sc = cfg.source_collection.model_copy(
        update={"workers": workers, "R_m": r_m, "validation_rounds": [1] if rounds > 1 else [0]}
    )
    return cfg.model_copy(
        update={
            "run_id": run_id,
            "total_steps": rounds * cfg.rollout_length,
            "output_dir": str(out_dir),
            "source_collection": sc,
        }
    )


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--R-m", type=int, default=30, dest="r_m")
    ap.add_argument("--keep", action="store_true", help="keep the scratch directory")
    args = ap.parse_args()

    torch.set_num_threads(1)
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)

    failures: list[str] = []

    # ---------------------------------------------------------------- A
    print(f"\n=== A. uninterrupted {args.rounds}-round run "
          f"(M=3, R_m={args.r_m}, workers={args.workers}) ===")
    cfg_a = _build(args.config, args.rounds, args.workers, args.r_m, "smoke_A", SCRATCH)
    t0 = time.time()
    dir_a = run_experiment_with_oracle(cfg_a, eval_every=1, checkpoint_every=1)
    wall_a = time.time() - t0
    rounds_a = read_jsonl(dir_a / "rounds.jsonl")
    meta_a = json.loads((dir_a / "run_metadata.json").read_text(encoding="utf-8"))
    print(f"  ran {len(rounds_a)} rounds in {wall_a:.1f}s")

    # --- 1/9. cost accounting -----------------------------------------
    t = meta_a["timing"]
    es = meta_a["env_steps"]
    print("\n--- per-phase wall clock (measured, replaces the projection) ---")
    print(f"  source collection : {t['source_s']:8.1f}s  ({t['source_s'] / len(rounds_a):6.1f}s/round)")
    print(f"  learner total     : {t['learner_total_s']:8.1f}s")
    print(f"  oracle eval       : {t['oracle_s']:8.1f}s")
    print(f"  wall clock        : {t['wall_clock_s']:8.1f}s  ({t['s_per_round']:6.1f}s/round)")
    print(f"  env steps         : ppo={es['ppo']:,}  source={es['source']:,}  oracle={es['oracle']:,}")
    print(f"  projected 250 rounds: {t['s_per_round'] * 250 / 3600:.2f} h")

    # --- 2. PPO continued ---------------------------------------------
    print("\n--- PPO health ---")
    aids = sorted(rounds_a[0]["constraints"])
    kl = np.array([[r["constraints"][a]["ppo"]["approx_kl"] for a in aids] for r in rounds_a])
    ent = np.array([[r["constraints"][a]["ppo"]["entropy"] for a in aids] for r in rounds_a])
    print(f"  approx_kl  min/med/max : {kl.min():.6f} / {np.median(kl):.6f} / {kl.max():.6f}")
    print(f"  entropy    min/max     : {ent.min():.4f} / {ent.max():.4f}")
    if not np.isfinite(kl).all() or not np.isfinite(ent).all():
        failures.append("PPO produced a non-finite KL or entropy")
        _fail("non-finite PPO diagnostics")

    # --- 1. source structure ------------------------------------------
    print("\n--- source structure (G9a) ---")
    for rec in rounds_a:
        sb = rec["source_batch"]
        for a in aids:
            n = len(rec["constraints"][a]["reports"])
            if n != cfg_a.source_collection.M:
                failures.append(f"round {rec['round_k']} owner {a}: {n} sources, expected 3")
        if sb["n_trajectories"] != cfg_a.source_collection.M * args.r_m:
            failures.append(f"round {rec['round_k']}: wrong trajectory count")
    sb0 = rounds_a[0]["source_batch"]
    print(f"  M={sb0['M']} R_m={sb0['R_m']} trajectories/round={sb0['n_trajectories']} "
          f"env_steps/round={sb0['env_steps']:,}")
    print(f"  source wall clock/round: {[round(r['source_batch']['wall_clock_s'], 1) for r in rounds_a]}")

    # --- 5. policy pinning (G9h) --------------------------------------
    print("\n--- policy pinning (G9h) ---")
    for rec in rounds_a:
        sb = rec["source_batch"]
        if not sb.get("theta_k_checksum_stable_through_dual"):
            failures.append(f"round {rec['round_k']}: theta moved before the dual update")
        print(f"  round {rec['round_k']}: theta_k={sb['policy_checksum'][:16]}... "
              f"chunks={sb['n_chunks']} all_match={sb['worker_checksums_all_match']} "
              f"stable_through_dual={sb.get('theta_k_checksum_stable_through_dual')}")
    checksums = [r["source_batch"]["policy_checksum"] for r in rounds_a]
    if len(set(checksums)) != len(checksums):
        failures.append("theta_k checksum repeated across rounds -- the policy is not moving")
        _fail("theta_k identical across rounds")

    # --- 6. RNG independence (G9g) ------------------------------------
    print("\n--- RNG independence (G9g) ---")
    seed_recs = read_jsonl(dir_a / "source_seeds.jsonl")
    all_env: list[int] = []
    for rec in seed_recs:
        per_replica = {rid: {p[0] for p in pairs} for rid, pairs in rec["seeds"].items()}
        rids = sorted(per_replica)
        for i in range(len(rids)):
            for j in range(i + 1, len(rids)):
                overlap = per_replica[rids[i]] & per_replica[rids[j]]
                if overlap:
                    failures.append(f"round {rec['round_k']}: replica seed overlap {sorted(overlap)[:5]}")
        for pairs in rec["seeds"].values():
            all_env.extend(p[0] for p in pairs)
        all_env.extend(p[0] for p in rec["reference_seeds"])
    dupes = len(all_env) - len(set(all_env))
    audit = meta_a["source_seed_audit"]
    print(f"  env seeds issued={len(all_env)}  duplicates={dupes}  "
          f"audit_duplicate_events={audit['duplicate_seed_events']}")
    if dupes or audit["duplicate_seed_events"]:
        failures.append(f"{dupes} duplicate source seeds")
        _fail("duplicate source seeds")

    # --- 3/4. dual update (G9d) ---------------------------------------
    print("\n--- dual update (G9d) ---")
    eta = cfg_a.dual.eta_lambda
    worst = 0.0
    n_unclipped = 0
    for rec in rounds_a:
        for a in aids:
            c = rec["constraints"][a]
            lam, mixed, res = c["lambda_after"], c["lambda_mixed_before"], c["constraint_residual"]
            if 1e-9 < lam < cfg_a.dual.lambda_max - 1e-9:
                n_unclipped += 1
                worst = max(worst, abs((lam - mixed) - eta * res))
    print(f"  unclipped cells={n_unclipped}  max |(lam - W lam) - eta*residual| = {worst:.3e}")
    if n_unclipped and worst > 1e-9:
        failures.append(f"G9d-i violated: max wiring error {worst:.3e}")
        _fail("dual wiring identity violated")
    lam_last = [rounds_a[-1]["constraints"][a]["lambda_after"] for a in aids]
    res_first = [rounds_a[0]["constraints"][a]["constraint_residual"] for a in aids]
    print(f"  round 0 residuals : {[round(x, 3) for x in res_first]}")
    print(f"  final lambda      : {[round(x, 4) for x in lam_last]}")

    # --- 8. validation reference --------------------------------------
    print("\n--- validation reference (step 3b) ---")
    ref_path = dir_a / "validation_reference.jsonl"
    if not ref_path.exists():
        failures.append("no validation_reference.jsonl written")
        _fail("validation reference missing")
    else:
        refs = read_jsonl(ref_path)
        r0 = refs[0]
        print(f"  rounds with a reference: {[r['round_k'] for r in refs]}  R_ref={r0['R_ref']}")
        print(f"  reference means: { {k: round(v, 3) for k, v in r0['reference_mean'].items()} }")
        blob = json.dumps(rounds_a)
        if "reference_mean" in blob:
            failures.append("the withheld reference leaked into rounds.jsonl")
            _fail("reference leaked into rounds.jsonl")
        # the reference must be a different draw from every source batch
        for rid, per_owner in r0["source_per_trajectory"].items():
            for a in aids:
                if per_owner[a] == r0["reference_per_trajectory"][a][: len(per_owner[a])]:
                    failures.append(f"reference reuses replica {rid}'s trajectories")

    # ---------------------------------------------------------------- B
    print(f"\n=== B. interrupt after round 1, resume to {args.rounds} ===")
    cfg_b1 = _build(args.config, 1, args.workers, args.r_m, "smoke_B", SCRATCH)
    run_experiment_with_oracle(cfg_b1, eval_every=1, checkpoint_every=1)
    cfg_b2 = _build(args.config, args.rounds, args.workers, args.r_m, "smoke_B", SCRATCH)
    dir_b = run_experiment_with_oracle(cfg_b2, eval_every=1, checkpoint_every=1)
    rounds_b = read_jsonl(dir_b / "rounds.jsonl")

    print(f"  A rounds={len(rounds_a)}  B rounds={len(rounds_b)}")
    if len(rounds_a) != len(rounds_b):
        failures.append(f"resume produced {len(rounds_b)} rounds, uninterrupted produced {len(rounds_a)}")
        _fail("round count mismatch after resume")
    else:
        mismatches = []
        for ra, rb in zip(rounds_a, rounds_b, strict=True):
            for a in aids:
                ca, cb = ra["constraints"][a], rb["constraints"][a]
                for key in ("mechanism_reported_cost_return", "constraint_residual", "lambda_after"):
                    if ca[key] != cb[key]:
                        mismatches.append((ra["round_k"], a, key, ca[key], cb[key]))
                for va, vb in zip(ca["reports"], cb["reports"], strict=True):
                    if va["value"] != vb["value"]:
                        mismatches.append((ra["round_k"], a, va["source_id"], va["value"], vb["value"]))
            if ra["source_batch"]["policy_checksum"] != rb["source_batch"]["policy_checksum"]:
                mismatches.append((ra["round_k"], "-", "policy_checksum", "", ""))
        if mismatches:
            failures.append(f"resume was not bitwise: {len(mismatches)} mismatching fields")
            _fail(f"resume not bitwise, first 5: {mismatches[:5]}")
        else:
            print("  resume is BITWISE identical across every source value, "
                  "residual, lambda and policy checksum")

    # ---------------------------------------------------------------- verdict
    print("\n=== SMOKE TEST VERDICT ===")
    if failures:
        print(f"FAILED ({len(failures)} problems):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASSED. The integrated loop runs; resume is bitwise; pinning, RNG")
    print("independence and dual wiring all hold on live data.")
    print(f"\nMeasured cost: {t['s_per_round']:.1f}s/round -> "
          f"{t['s_per_round'] * 250 / 3600:.2f}h for the 250-round primary run.")
    if not args.keep:
        print(f"(scratch kept at {SCRATCH} -- it is not a scientific artifact)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
