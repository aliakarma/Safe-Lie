#!/usr/bin/env python
"""A2 gates G1-ii, G2-ii/iii, G7: verify the RCE mechanism on a finished run.

Everything here is checked by RE-DERIVING the aggregation from the run's own
logged pre-attack source reports and comparing against what the run recorded.
That is a stronger check than A1's constant-shift identity: it reproduces the
whole map

    values  = pre-attack reports, with -B applied to the mapped source
    agg     = rce_aggregate(values, f, beta, sigma_min, min_retained)

and requires `point_estimate`, `spread`, `retained_n`, `applied_margin` and
`pessimistic_estimate` to match the log to <= 1e-9 on every (round, owner)
cell. If aggregation ever ran on something other than the logged reports, or
the attack landed anywhere other than the mapped source, this fails.

It also answers the two mechanistic questions the gates document asks about
what RCE does to the SOURCE SET at M=3, f=1 (section 16 of the brief):
which source is retained, whether the attacked source is removed or merely
bounded, and whether the margin is ever load-bearing.

Usage:
    python scripts/a2_rce_mechanism_validation.py \
        --run results/runs_a2/C_seed0 --condition C \
        --paired results/runs_a1/B_seed0 \
        --attacked-source batch_1 --out <run>/a2_mechanism_report.json
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

# The degenerate-retained-set warning fires on every cell by construction at
# M=3, f=1 (that is the audited finding, not a surprise); silence it so the
# validator's own output stays readable.
logging.getLogger("safelie.defenses.rce").setLevel(logging.CRITICAL)

TOL = 1e-9
B_MAGNITUDE = 12.5
D_BUDGET = 25.0
BETA = 1.5
SIGMA_MIN = 1e-3
MIN_RETAINED = 3
F = 1

# Pre-registered in docs/a2_rce_gates.md section 5 / gate G7-iv.
PRED_RCE_SHIFT = -0.59
G7_IV_BAND = (-1.20, -0.20)
G7_III_MIN_FRAC = 0.99


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def load(run_dir: Path) -> dict:
    rounds = read_jsonl(run_dir / "rounds.jsonl")
    aids = sorted(rounds[0]["constraints"].keys())
    sids = sorted(r["source_id"] for r in rounds[0]["constraints"][aids[0]]["reports"])
    pre = np.array(
        [[[next(x["value"] for x in r["constraints"][a]["reports"] if x["source_id"] == sid)
           for sid in sids] for a in aids] for r in rounds],
        dtype=float)                                            # K x N x S
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
        "guarantee_in_force": [[r["constraints"][a].get("guarantee_in_force") for a in aids]
                               for r in rounds],
        "epsilon_offline": [[r["constraints"][a].get("epsilon_offline") for a in aids]
                            for r in rounds],
    }


def rce_cells(values: np.ndarray) -> dict:
    """rce_aggregate over a K x N x S array, vectorised into flat arrays."""
    K, N, _ = values.shape
    out = {k: np.zeros((K, N)) for k in
           ("point_estimate", "spread", "retained_n", "applied_margin", "pessimistic_estimate")}
    retained_val = np.zeros((K, N))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in range(K):
            for n in range(N):
                r = rce_aggregate(values[k, n], F, BETA, SIGMA_MIN, MIN_RETAINED)
                out["point_estimate"][k, n] = r.point_estimate
                out["spread"][k, n] = r.spread
                out["retained_n"][k, n] = r.retained_n
                out["applied_margin"][k, n] = r.applied_margin
                out["pessimistic_estimate"][k, n] = r.pessimistic_estimate
                retained_val[k, n] = r.retained_values[0]
    out["retained_value"] = retained_val
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--condition", required=True, choices=["C", "E"])
    ap.add_argument("--paired", required=True,
                    help="the same-seed no-RCE run: B for condition C, A for condition E")
    ap.add_argument("--attacked-source", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run = load(Path(args.run))
    paired = load(Path(args.paired))
    rep: dict = {
        "run": run["dir"], "condition": args.condition, "paired_reference": paired["dir"],
        "rounds": run["rounds"], "agent_ids": run["agent_ids"], "source_ids": run["source_ids"],
        "attacked_source": args.attacked_source,
        "corrupted_source_ids_in_log": run["corrupted"],
    }

    # ---------------------------------------------------- post-attack values
    values = run["pre"].copy()
    if args.condition == "C":
        assert args.attacked_source, "condition C needs --attacked-source"
        j = run["source_ids"].index(args.attacked_source)
        values[:, :, j] -= B_MAGNITUDE

    # --------------------------------------- G2-ii / G2-iii: exact map identity
    got = rce_cells(values)
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
        "attack_applied": args.condition == "C",
        "shift_applied": -B_MAGNITUDE if args.condition == "C" else 0.0,
        "pass": bool(max(errs.values()) <= TOL and
                     (run["corrupted"] == [args.attacked_source] if args.condition == "C"
                      else run["corrupted"] == [])),
    }

    # ------------------------------------------------------- G7-i / G7-ii
    rep["G7_i_degenerate_structure"] = {
        "frac_retained_n_eq_1": float((run["logged"]["retained_n"] == 1).mean()),
        "frac_degenerate_true": float(run["degenerate"].mean()),
        "pass": bool((run["logged"]["retained_n"] == 1).all() and run["degenerate"].all()),
    }
    eps = [v for row in run["epsilon_offline"] for v in row if v is not None]
    rep["G7_ii_margin_is_constant"] = {
        "frac_spread_eq_sigma_min": float((np.abs(run["logged"]["spread"] - SIGMA_MIN)
                                           <= TOL).mean()),
        "frac_margin_eq_beta_sigma_min": float(
            (np.abs(run["logged"]["applied_margin"] - BETA * SIGMA_MIN) <= TOL).mean()),
        "applied_margin_value": float(run["logged"]["applied_margin"].mean()),
        "mad_ever_nonzero": bool((run["logged"]["spread"] > SIGMA_MIN + TOL).any()),
        "epsilon_offline_unique": sorted({round(float(v), 12) for v in eps}) if eps else [],
        "guarantee_in_force_frac_true": float(np.mean(
            [[bool(v) for v in row] for row in run["guarantee_in_force"]])),
        "theoretical_condition_ever_active": False,
        "note": ("MAD is structurally 0 at |T|=1, so the spread is always floored to "
                 "sigma_min and the margin is the constant beta*sigma_min. The MAD "
                 "condition is never active and guarantee_in_force is vacuous "
                 "(docs/a2_rce_gates.md section 4)."),
        "pass": bool((np.abs(run["logged"]["spread"] - SIGMA_MIN) <= TOL).all()
                     and (np.abs(run["logged"]["applied_margin"] - BETA * SIGMA_MIN) <= TOL).all()),
    }

    # --------------------------------------------- G7-iii: what gets trimmed
    if args.condition == "C":
        j = run["source_ids"].index(args.attacked_source)
        attacked_post = values[:, :, j]
        honest = np.delete(values, j, axis=2)
        retained_is_attacked = np.abs(got["retained_value"] - attacked_post) <= TOL
        retained_is_min_honest = np.abs(got["retained_value"] - honest.min(axis=2)) <= TOL
        attacked_is_global_min = (attacked_post <= values.min(axis=2) + TOL)
        frac_trimmed = float((~retained_is_attacked).mean())
        frac_min_honest = float(retained_is_min_honest.mean())
        rep["G7_iii_source_set"] = {
            "frac_attacked_source_trimmed": frac_trimmed,
            "frac_attacked_source_retained": float(retained_is_attacked.mean()),
            "frac_attacked_source_is_global_min": float(attacked_is_global_min.mean()),
            "frac_retained_equals_min_of_two_honest": frac_min_honest,
            "interpretation": ("RCE REMOVES the corrupted source rather than bounding its "
                               "influence: with B = 12.5 = 11.9 sigma_src the corrupted "
                               "report is the minimum, so trimming the extremes discards it "
                               "and retains the SMALLER of the two honest sources."),
            "pass": bool(frac_trimmed >= G7_III_MIN_FRAC and frac_min_honest >= G7_III_MIN_FRAC),
        }

        # ---- G7-iv: within-run counterfactual, zero sampling noise
        clean_side = rce_cells(run["pre"])
        shift = got["pessimistic_estimate"] - clean_side["pessimistic_estimate"]
        mean_shift = float(shift.mean())
        rep["G7_iv_within_run_counterfactual"] = {
            "mean_rce_shift": mean_shift,
            "median_rce_shift": float(np.median(shift)),
            "sd_rce_shift": float(shift.std(ddof=1)),
            "predicted": PRED_RCE_SHIFT,
            "band": list(G7_IV_BAND),
            "mean_aggregator_shift_for_comparison": -B_MAGNITUDE / len(run["source_ids"]),
            "suppression_factor_vs_mean": float(
                (B_MAGNITUDE / len(run["source_ids"])) / abs(mean_shift)) if mean_shift else None,
            "pass": bool(G7_IV_BAND[0] <= mean_shift <= G7_IV_BAND[1]),
        }
    else:
        rep["G7_iii_source_set"] = {
            "note": "condition E carries no attack; nothing to trim adversarially",
            "frac_retained_equals_median": float(
                (np.abs(got["retained_value"] - np.median(values, axis=2)) <= TOL).mean()),
            "pass": True,
        }
        rep["G7_iv_within_run_counterfactual"] = {
            "note": "not applicable to a clean condition", "pass": True}

    # ------------------------------------------------- G7-v: ordering stability
    order = np.argsort(values, axis=2)
    same = (order[1:] == order[:-1]).all(axis=2)
    rep["G7_v_source_ordering_stability"] = {
        "frac_consecutive_rounds_same_order": float(same.mean()),
        "descriptive_only": True,
    }

    # ------------------------------- G1-ii: CRN against the paired no-RCE run
    d0 = run["pre"][0] - paired["pre"][0]
    rep["G1_ii_crn_round0"] = {
        "paired_run": paired["dir"],
        "max_abs_pre_attack_difference": float(np.abs(d0).max()),
        "per_source_max_abs": {sid: float(np.abs(d0[:, i]).max())
                               for i, sid in enumerate(run["source_ids"])},
        "underlying_draws_identical": bool(np.abs(d0).max() <= TOL),
        "pass": bool(np.abs(d0).max() <= TOL),
    }

    # ------------------------------------------------------------ finiteness
    finite = all(np.isfinite(run["logged"][k]).all() for k in run["logged"])
    rep["finite"] = bool(finite)

    gates = {k: v["pass"] for k, v in rep.items()
             if isinstance(v, dict) and "pass" in v}
    rep["gates"] = gates
    rep["all_pass"] = bool(all(gates.values()) and finite)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print(f"=== A2 RCE mechanism validation: {run['dir']} (condition {args.condition}) ===")
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  finite: {finite}")
    print(f"-> {out}")
    print(f"{'ALL PASS' if rep['all_pass'] else 'FAILURES PRESENT'}")
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
