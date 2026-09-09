#!/usr/bin/env python
"""A3 gates G1-ii, G2-ii/iii, G3, G7: verify the M=5 RCE mechanism on a run.

Same method as A2's validator -- re-derive the whole aggregation from the
run's own logged pre-attack source reports and require it to reproduce every
logged field to 1e-9 -- but the gates it enforces are the opposite ones.

A2 had to prove the margin was DEAD: |T|=1, MAD structurally zero, spread
floored, margin a constant. A3 has to prove it is ALIVE: |T|=3, MAD a real
measurement, nothing floored, and the margin responding to corruption. If
A3-G3-i or G3-ii fails then A3 has not tested the margin at all and no
conclusion about Theorem 2's condition may be drawn from it -- the same
discipline that made A2 report its own degeneracy rather than bury it.

Usage:
    python scripts/a3_rce_mechanism_validation.py \
        --run results/runs_a3/C_seed0 --condition C \
        --paired results/runs_a3/B_seed0 \
        --attacked-source batch_1 --out <run>/a3_mechanism_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safelie.defenses.rce import rce_aggregate  # noqa: E402

logging.getLogger("safelie.defenses.rce").setLevel(logging.CRITICAL)

TOL = 1e-9
B_MAGNITUDE = 12.5
D_BUDGET = 25.0
BETA = 1.5
SIGMA_MIN = 1e-3
MIN_RETAINED = 3
F = 1
M = 5

# All pre-registered in docs/a3_gates.md sections 2 and 6.
G3_II_MIN_FRAC = 0.99                 # spread > sigma_min on >= 99% of cells
G3_III_BAND_CLEAN = (0.25, 0.60)      # mean applied_margin in E'  (predicted 0.404)
G3_III_BAND_ATTACK = (0.35, 0.80)     # mean applied_margin in C'  (predicted 0.544)
G7_I_MIN_FRAC = 0.99                  # attacked source trimmed
G7_II_BAND = (-0.70, -0.02)           # within-run counterfactual (predicted -0.222)
PRED_SHIFT = -0.222


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def load(run_dir: Path) -> dict:
    rounds = read_jsonl(run_dir / "rounds.jsonl")
    aids = sorted(rounds[0]["constraints"].keys())
    sids = sorted(r["source_id"] for r in rounds[0]["constraints"][aids[0]]["reports"])
    pre = np.array(
        [[[next(x["value"] for x in r["constraints"][a]["reports"] if x["source_id"] == sid)
           for sid in sids] for a in aids] for r in rounds], dtype=float)
    logged = {
        k: np.array([[r["constraints"][a]["aggregate"][k] for a in aids] for r in rounds],
                    dtype=float)
        for k in ("point_estimate", "spread", "retained_n", "applied_margin",
                  "pessimistic_estimate")
    }
    logged["mechanism_reported_cost_return"] = np.array(
        [[r["constraints"][a]["mechanism_reported_cost_return"] for a in aids] for r in rounds],
        dtype=float)
    logged["constraint_residual"] = np.array(
        [[r["constraints"][a]["constraint_residual"] for a in aids] for r in rounds], dtype=float)
    return {
        "dir": str(run_dir), "rounds": len(rounds), "agent_ids": aids, "source_ids": sids,
        "pre": pre, "logged": logged,
        "degenerate": np.array([[r["constraints"][a]["aggregate"]["degenerate"] for a in aids]
                                for r in rounds], dtype=bool),
        "corrupted": sorted(rounds[0]["constraints"][aids[0]]["corrupted_source_ids"]),
        "epsilon_offline": [[r["constraints"][a].get("epsilon_offline") for a in aids]
                            for r in rounds],
        "guarantee_in_force": [[r["constraints"][a].get("guarantee_in_force") for a in aids]
                               for r in rounds],
    }


def rce_cells(values: np.ndarray) -> dict:
    K, N, _ = values.shape
    out = {k: np.zeros((K, N)) for k in
           ("point_estimate", "spread", "retained_n", "applied_margin", "pessimistic_estimate")}
    retained = np.zeros((K, N, M - 2 * F))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in range(K):
            for n in range(N):
                r = rce_aggregate(values[k, n], F, BETA, SIGMA_MIN, MIN_RETAINED)
                for key in out:
                    out[key][k, n] = getattr(r, key)
                retained[k, n] = np.sort(r.retained_values)
    out["retained_values"] = retained
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--condition", required=True, choices=["C", "E"])
    ap.add_argument("--paired", required=True,
                    help="the same-seed no-RCE run: B' for condition C', A' for E'")
    ap.add_argument("--attacked-source", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run, paired = load(Path(args.run)), load(Path(args.paired))
    rep: dict = {
        "run": run["dir"], "condition": args.condition, "paired_reference": paired["dir"],
        "rounds": run["rounds"], "M": M, "f": F, "retained_set_size": M - 2 * F,
        "source_ids": run["source_ids"], "attacked_source": args.attacked_source,
        "corrupted_source_ids_in_log": run["corrupted"],
    }
    if len(run["source_ids"]) != M:
        rep["fatal"] = f"expected {M} sources, log has {len(run['source_ids'])}"
        Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print("FATAL:", rep["fatal"])
        return 1

    values = run["pre"].copy()
    if args.condition == "C":
        assert args.attacked_source, "condition C' needs --attacked-source"
        j = run["source_ids"].index(args.attacked_source)
        values[:, :, j] -= B_MAGNITUDE

    got = rce_cells(values)

    # ---------------------------------------------- G2: exact map identity
    errs = {k: float(np.abs(got[k] - run["logged"][k]).max())
            for k in ("point_estimate", "spread", "retained_n", "applied_margin",
                      "pessimistic_estimate")}
    errs["mechanism_reported_cost_return"] = float(
        np.abs(got["pessimistic_estimate"] - run["logged"]["mechanism_reported_cost_return"]).max())
    errs["constraint_residual"] = float(
        np.abs((got["pessimistic_estimate"] - D_BUDGET)
               - run["logged"]["constraint_residual"]).max())
    rep["G2_map_identity"] = {
        "n_cells": int(values.shape[0] * values.shape[1]),
        "max_abs_error": errs,
        "shift_applied": -B_MAGNITUDE if args.condition == "C" else 0.0,
        "pass": bool(max(errs.values()) <= TOL and
                     (run["corrupted"] == [args.attacked_source] if args.condition == "C"
                      else run["corrupted"] == [])),
    }

    # -------------------------------- G3: THE MARGIN IS LIVE (A3's reason to exist)
    rn = run["logged"]["retained_n"]
    sp = run["logged"]["spread"]
    mg = run["logged"]["applied_margin"]
    rep["G3_i_retained_structure"] = {
        "frac_retained_n_eq_3": float((rn == 3).mean()),
        "frac_degenerate_true": float(run["degenerate"].mean()),
        "retained_n_unique": sorted({int(x) for x in np.unique(rn)}),
        "pass": bool((rn == 3).all() and not run["degenerate"].any()),
    }
    frac_live = float((sp > SIGMA_MIN + TOL).mean())
    rep["G3_ii_mad_is_a_measurement"] = {
        "frac_spread_above_sigma_min": frac_live,
        "spread_min": float(sp.min()), "spread_max": float(sp.max()),
        "spread_mean": float(sp.mean()), "spread_median": float(np.median(sp)),
        "frac_spread_at_floor": float((np.abs(sp - SIGMA_MIN) <= TOL).mean()),
        "bar": G3_II_MIN_FRAC,
        "pass": bool(frac_live >= G3_II_MIN_FRAC),
    }
    band = G3_III_BAND_ATTACK if args.condition == "C" else G3_III_BAND_CLEAN
    rep["G3_iii_margin_magnitude"] = {
        "mean_applied_margin": float(mg.mean()),
        "median_applied_margin": float(np.median(mg)),
        "sd_applied_margin": float(mg.std(ddof=1)),
        "predicted": 0.544 if args.condition == "C" else 0.404,
        "band": list(band),
        "a2_constant_for_comparison": BETA * SIGMA_MIN,
        "ratio_vs_a2_constant": float(mg.mean() / (BETA * SIGMA_MIN)),
        "pass": bool(band[0] <= mg.mean() <= band[1]),
    }
    eps = [v for row in run["epsilon_offline"] for v in row if v is not None]
    uniq = sorted({round(float(v), 12) for v in eps}) if eps else []
    rep["G1_v_calibration_not_vacuous"] = {
        "epsilon_offline_unique": uniq,
        "sigma_min": SIGMA_MIN,
        "strictly_above_sigma_min": bool(uniq and min(uniq) > SIGMA_MIN + TOL),
        "guarantee_in_force_frac_true": float(np.mean(
            [[bool(v) for v in row] for row in run["guarantee_in_force"]])),
        "note": ("At M=5 the MAD is live, so epsilon_offline is a real quantile of a "
                 "real dispersion distribution rather than A2's constant sigma_min. "
                 "guarantee_in_force is still NOT a verification of Theorem 2's "
                 "precondition -- see safelie.eval.margin."),
        "pass": bool(uniq and min(uniq) > SIGMA_MIN + TOL),
    }

    # ------------------------------------------------ G7: what happens to the sources
    if args.condition == "C":
        j = run["source_ids"].index(args.attacked_source)
        attacked_post = values[:, :, j]
        honest = np.delete(values, j, axis=2)
        ret = got["retained_values"]
        is_attacked = (np.abs(ret - attacked_post[..., None]) <= TOL).any(axis=2)
        frac_trimmed = float((~is_attacked).mean())
        honest_sorted = np.sort(honest, axis=2)
        three_lowest = np.abs(ret - honest_sorted[:, :, :3]).max(axis=2) <= TOL
        rep["G7_i_source_set"] = {
            "frac_attacked_source_trimmed": frac_trimmed,
            "frac_attacked_source_retained": float(is_attacked.mean()),
            "frac_retained_equals_three_lowest_honest": float(three_lowest.mean()),
            "bar": G7_I_MIN_FRAC,
            "interpretation": ("With B = 12.5 the corrupted report is far below the "
                               "honest ones, so trimming discards it and retains the "
                               "three lowest of the four honest sources."),
            "pass": bool(frac_trimmed >= G7_I_MIN_FRAC),
        }
        clean_side = rce_cells(run["pre"])
        shift = got["pessimistic_estimate"] - clean_side["pessimistic_estimate"]
        ms = float(shift.mean())
        rep["G7_ii_within_run_counterfactual"] = {
            "mean_rce_shift": ms, "median_rce_shift": float(np.median(shift)),
            "sd_rce_shift": float(shift.std(ddof=1)),
            "predicted": PRED_SHIFT, "band": list(G7_II_BAND),
            "mean_aggregator_shift_for_comparison": -B_MAGNITUDE / M,
            "suppression_factor_vs_mean": float((B_MAGNITUDE / M) / abs(ms)) if ms else None,
            "a2_measured_shift_at_M3": -0.695,
            "pass": bool(G7_II_BAND[0] <= ms <= G7_II_BAND[1]),
        }
        rep["G7_iii_margin_under_attack"] = {
            "mean_applied_margin": float(mg.mean()),
            "note": "compared against the same seed's E' by scripts/analyze_a3.py (G3-iv)",
        }
    else:
        med3 = np.sort(values, axis=2)[:, :, 1:4]
        rep["G7_i_source_set"] = {
            "note": "clean condition; nothing to trim adversarially",
            "frac_retained_equals_middle_three": float(
                (np.abs(got["retained_values"] - med3).max(axis=2) <= TOL).mean()),
            "pass": True,
        }
        rep["G7_ii_within_run_counterfactual"] = {
            "note": "not applicable to a clean condition", "pass": True}

    # --------------------------------------------- G1-ii: within-machine CRN
    d0 = run["pre"][0] - paired["pre"][0]
    rep["G1_ii_crn_round0"] = {
        "paired_run": paired["dir"],
        "max_abs_pre_attack_difference": float(np.abs(d0).max()),
        "per_source_max_abs": {sid: float(np.abs(d0[:, i]).max())
                               for i, sid in enumerate(run["source_ids"])},
        "pass": bool(np.abs(d0).max() <= TOL),
        "note": ("Both runs are produced on the SAME machine (BRANCH-B', gates doc "
                 "section 12.4), so this is a within-machine check as declared."),
    }

    finite = all(np.isfinite(run["logged"][k]).all() for k in run["logged"])
    rep["finite"] = bool(finite)
    gates = {k: v["pass"] for k, v in rep.items() if isinstance(v, dict) and "pass" in v}
    rep["gates"] = gates
    rep["all_pass"] = bool(all(gates.values()) and finite)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print(f"=== A3 M=5 mechanism validation: {run['dir']} (condition {args.condition}') ===")
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  finite: {finite}")
    if "G3_ii_mad_is_a_measurement" in rep:
        g = rep["G3_ii_mad_is_a_measurement"]
        print(f"  MAD live on {g['frac_spread_above_sigma_min']:.4f} of cells "
              f"(mean spread {g['spread_mean']:.4f})")
        m3 = rep["G3_iii_margin_magnitude"]
        print(f"  mean margin {m3['mean_applied_margin']:.4f} "
              f"(predicted {m3['predicted']}, band {m3['band']}, "
              f"{m3['ratio_vs_a2_constant']:.0f}x A2's constant)")
    print(f"-> {out}")
    print("ALL PASS" if rep["all_pass"] else "FAILURES PRESENT")
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
