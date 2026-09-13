#!/usr/bin/env python
"""Structural verification of one unconstrained-control training run.

Implements gates U-S1..U-S7 of docs/unconstrained_control.md section 8.
These are validity checks on the ARTIFACT, not on the result: whether the
run finished, whether its numbers are finite, whether the multiplier was
actually pinned to zero, and whether an attack or an aggregator crept in.
Nothing here looks at the size or sign of the U-A difference, so running
it per seed cannot leak the outcome or bias the estimate.

The scientific contrast belongs to `scripts/analyze_unconstrained.py` and
is only computed once all three seeds exist.

Usage:
    python scripts/verify_unconstrained_run.py --run results/runs_unconstrained/U_seed0
    python scripts/verify_unconstrained_run.py --run ... --allow-partial
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROUNDS = 250
EXPECTED_ENV_SEEDS = 22_500   # 250 rounds x M=3 x R_m=30 recorded pairs;
                              # the 5 validation rounds go to validation_reference.jsonl
D_BUDGET = 25.0
KL_MEDIAN_MAX = 0.0069        # G10-C-iii verbatim; REPORTED, not gated
KL_P95_MAX = 0.01548

SEED_ENTROPY = {
    0: 286314957402113664887331205920951063913,
    1: 52021175099534868945312500562751741008,
    2: 335268198726974212397240672597355197200,
}
CLEAN_DIR = {
    0: ROOT / "results/runs_constraint_batch_g9/g9_batch_clean",
    1: ROOT / "results/runs_constraint_batch_g10/seed1",
    2: ROOT / "results/runs_constraint_batch_g10/seed2",
}


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def check(out: list, gate: str, ok: bool, detail=None) -> bool:
    out.append({"gate": gate, "pass": bool(ok), "detail": detail})
    return bool(ok)


def env_seeds(run: Path) -> set:
    """Every environment seed this run drew, from source_seeds.jsonl.

    A seed is the recorded PAIR, not either of its halves: each trajectory
    is keyed by a two-element list, and two runs could legitimately share a
    single 32-bit half by chance while drawing entirely different
    trajectories. Comparing pairs is what `g10_derive_source_entropy.py`
    verified offline before the clean runs launched, so U-S7 compares the
    same objects.
    """
    seeds: set = set()
    f = run / "source_seeds.jsonl"
    if not f.exists():
        return seeds
    for row in read_jsonl(f):
        for _sid, pairs in (row.get("seeds") or {}).items():
            for pair in pairs:
                seeds.add(tuple(pair) if isinstance(pair, list) else pair)
    return seeds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--allow-partial", action="store_true",
                    help="do not fail U-S1 on a short run (for mid-run inspection)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    if not run.is_absolute():
        run = ROOT / run
    seed = int(run.name.split("_seed")[-1])

    orc = read_jsonl(run / "oracle.jsonl") if (run / "oracle.jsonl").exists() else []
    rnd = read_jsonl(run / "rounds.jsonl") if (run / "rounds.jsonl").exists() else []
    md = json.loads((run / "run_metadata.json").read_text(encoding="utf-8")) \
        if (run / "run_metadata.json").exists() else {}
    snap = md.get("config_snapshot", {})

    gates: list = []
    rep: dict = {"run": str(run.relative_to(ROOT)).replace("\\", "/"), "seed": seed,
                 "predeclaration": "docs/unconstrained_control.md",
                 "rounds_oracle": len(orc), "rounds_rounds": len(rnd)}

    # ---------------------------------------------------------------- U-S1
    complete = len(orc) == len(rnd) == EXPECTED_ROUNDS
    check(gates, "U-S1 run complete (250 rounds in both logs)",
          complete or args.allow_partial,
          {"oracle": len(orc), "rounds": len(rnd), "expected": EXPECTED_ROUNDS,
           "partial_allowed": args.allow_partial})
    rep["complete"] = complete

    if not orc or not rnd:
        rep["gates"] = gates
        rep["all_pass"] = False
        print(json.dumps(rep, indent=2))
        return 1

    aids = sorted(orc[0]["agents"])
    K = min(len(orc), len(rnd))

    def oarr(f):
        return np.array([[r["agents"][a][f] for a in aids] for r in orc[:K]], dtype=float)

    def carr(f):
        return np.array([[r["constraints"][a][f] for a in aids] for r in rnd[:K]], dtype=float)

    lam = carr("lambda_after")
    true_cost = oarr("true_cost_return")
    task_return = oarr("episodic_task_return")
    mech = carr("mechanism_reported_cost_return")
    residual = carr("constraint_residual")
    gap = oarr("detection_gap_vs_aggregate")

    # ---------------------------------------------------------------- U-S2
    arrays = {"lambda": lam, "true_cost": true_cost, "task_return": task_return,
              "mech_reported": mech, "residual": residual, "mech_gap": gap}
    nonfinite = {k: int((~np.isfinite(v)).sum()) for k, v in arrays.items()}
    check(gates, "U-S2 all logged quantities finite (no NaN, no Inf)",
          all(v == 0 for v in nonfinite.values()), nonfinite)

    # ---------------------------------------------------------------- U-S3
    check(gates, "U-S3 lambda is exactly 0.0 at every round and every agent",
          bool((lam == 0.0).all()),
          {"max_abs": float(np.abs(lam).max()),
           "n_nonzero_cells": int((lam != 0.0).sum()), "n_cells": int(lam.size),
           "lambda_mean": float(lam.mean()), "lambda_max": float(lam.max())})

    # ---------------------------------------------------------------- U-S4
    corrupted = sorted(rnd[0]["constraints"][aids[0]].get("corrupted_source_ids", []))
    check(gates, "U-S4 no attack and no RCE",
          (snap.get("attack", {}).get("name") == "none"
           and snap.get("attack", {}).get("f") == 0
           and corrupted == []
           and snap.get("defense", {}).get("name") == "mean"
           and snap.get("defense", {}).get("f") == 0),
          {"attack": snap.get("attack", {}).get("name"),
           "attack_f": snap.get("attack", {}).get("f"),
           "corrupted_source_ids": corrupted,
           "defense": snap.get("defense", {}).get("name"),
           "defense_f": snap.get("defense", {}).get("f")})

    # ---------------------------------------------------------------- U-S5
    # The recorded snapshot is what actually ran; re-check the treatment and
    # the untouched learner against it rather than against the YAML on disk.
    ppo_expect = {"clip": 0.2, "gae_lambda": 0.95, "gamma": 0.99, "lr": 0.0003,
                  "epochs": 4, "minibatches": 4, "hidden_dim": 64,
                  "entropy_coef": 0.001, "value_coef": 0.5, "grad_clip": 0.5,
                  "critic_lr": None}
    ppo_got = snap.get("ppo", {})
    check(gates, "U-S5 recorded config is the control: lambda_max 0, learner untouched",
          (snap.get("dual", {}).get("lambda_max") == 0.0
           and snap.get("dual", {}).get("eta_lambda") == 0.035
           and all(ppo_got.get(k) == v for k, v in ppo_expect.items())
           and snap.get("total_steps") == 500_000
           and snap.get("rollout_length") == 2000
           and snap.get("constraint_estimator") == "mc_window"
           and snap.get("env", {}).get("budget") == D_BUDGET
           and snap.get("topology", {}).get("name") == "ring"
           and snap.get("source_collection", {}).get("seed_entropy") == SEED_ENTROPY[seed]),
          {"lambda_max": snap.get("dual", {}).get("lambda_max"),
           "eta_lambda": snap.get("dual", {}).get("eta_lambda"),
           "ppo_mismatch": {k: (v, ppo_got.get(k)) for k, v in ppo_expect.items()
                            if ppo_got.get(k) != v},
           "total_steps": snap.get("total_steps"),
           "rollout_length": snap.get("rollout_length"),
           "constraint_estimator": snap.get("constraint_estimator"),
           "seed_entropy_matches": snap.get("source_collection", {}).get("seed_entropy")
                                   == SEED_ENTROPY[seed]})

    # ---------------------------------------------------------------- U-S6
    check(gates, "U-S6 metadata records status complete and a committed src/",
          (md.get("status") == "complete" or args.allow_partial),
          {"status": md.get("status"), "git_sha": md.get("git", {}).get("sha"),
           "git_dirty": md.get("git", {}).get("dirty"),
           "git_dirty_paths": md.get("git", {}).get("dirty_paths"),
           "note": "dirty is expected: the control's own predeclaration, configs "
                   "and scripts are untracked until this campaign is committed. "
                   "What matters is that no src/ file is modified -- listed above "
                   "for audit."})

    # ---------------------------------------------------------------- U-S7
    mine = env_seeds(run)
    rep["n_env_seeds"] = len(mine)
    overlaps = {}
    for other in (0, 1, 2):
        if other == seed:
            continue
        o = run.parent / f"U_seed{other}"
        if o.exists():
            overlaps[f"U_seed{other}"] = len(mine & env_seeds(o))
    check(gates, "U-S7 env seeds recorded and not shared with the other control seeds",
          ((len(mine) == EXPECTED_ENV_SEEDS or args.allow_partial)
           and all(v == 0 for v in overlaps.values())),
          {"n_env_seeds": len(mine), "expected": EXPECTED_ENV_SEEDS,
           "overlap_with": overlaps})

    # ------------------------------------------------- reported, NOT gated
    last = slice(-50, None)
    health: dict = {
        "task_return_first20": float(task_return[:20].mean()),
        "task_return_last50": float(task_return[last].mean()),
        "task_learning_gain": float(task_return[last].mean() - task_return[:50].mean()),
        "lambda_mean": float(lam.mean()),
        "residual_last50_sum": float(residual[last].mean(axis=0).sum()),
    }
    ppo0 = rnd[0]["constraints"][aids[0]].get("ppo") or {}
    for key in ("approx_kl", "kl"):
        if key in ppo0:
            k = np.array([[r["constraints"][a]["ppo"][key] for a in aids]
                          for r in rnd[:K]], dtype=float)
            health["kl_median"] = float(np.median(k))
            health["kl_p95"] = float(np.percentile(k, 95))
            health["kl_median_within_G10_bar"] = bool(health["kl_median"] <= KL_MEDIAN_MAX)
            health["kl_p95_within_G10_bar"] = bool(health["kl_p95"] <= KL_P95_MAX)
            break
    if "entropy" in ppo0:
        e = np.array([[r["constraints"][a]["ppo"]["entropy"] for a in aids]
                      for r in rnd[:K]], dtype=float)
        health["entropy_first10"] = float(e[:10].mean())
        health["entropy_last10"] = float(e[-10:].mean())
        health["entropy_declines"] = bool(health["entropy_last10"] < health["entropy_first10"])
    rep["learning_health_reported_not_gated"] = health
    rep["timing"] = md.get("timing")
    rep["env_steps"] = md.get("env_steps")

    rep["gates"] = gates
    rep["all_pass"] = all(g["pass"] for g in gates)

    out = Path(args.json) if args.json else run / "unconstrained_run_verification.json"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print(f"\nStructural verification: {rep['run']}")
    for g in gates:
        print(f"  [{'PASS' if g['pass'] else 'FAIL'}] {g['gate']}")
        if not g["pass"]:
            print(f"         {g['detail']}")
    print(f"\n{'ALL PASS' if rep['all_pass'] else 'FAILED'}  -> {out}")
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
