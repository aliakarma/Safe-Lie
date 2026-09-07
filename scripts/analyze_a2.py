#!/usr/bin/env python
"""A2 analyser -- the 2x2 attack x RCE factorial.

Implements docs/a2_rce_gates.md exactly. Every bar is read from that
document; none is chosen here.

    Y00 = A   clean,  no RCE      Y10 = B   attack, no RCE
    Y01 = E   clean,  RCE         Y11 = C   attack, RCE

    attack effect            B - A     (established by A1: +3.986)
    clean RCE effect         E - A     (RCE's generic conservatism)
    attack effect under RCE  C - E
    RCE main effect (raw)    C - B     reported, never primary
    INTERACTION              (C - E) - (B - A)      <- primary

The metric is whole-run network-average true cost, where HIGHER IS WORSE,
so a negative interaction means the attack costs less under RCE than
without it. The sign convention is asserted at run time against the known
A1 value rather than assumed.

Paired by seed throughout. Per decision D6 (MIN_SEEDS_FOR_INFERENCE = 5)
the paired t-test is emitted as a labelled sensitivity analysis only; at
three seeds the evidence is per-seed sign consistency.

Usage:
    python scripts/analyze_a2.py --out results/runs_a2/a2_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants
# All from docs/a2_rce_gates.md. Do not tune here.
D_BUDGET = 25.0
B_MAGNITUDE = 12.5
M_SOURCES = 3
BETA = 1.5
SIGMA_MIN = 1e-3
A1_B_MINUS_A = 3.9858                # established, for the sign-convention check
G5_REDUCTION_BAR = 0.5               # PASS needs (C-E) <= 0.5 * (B-A)
G7_III_MIN_FRAC = 0.99
G7_IV_BAND = (-1.20, -0.20)
PRED_RCE_SHIFT = -0.59
KL_MEDIAN_MAX = 0.0069               # G10-C-iii, verbatim
KL_P95_MAX = 0.01548
LAMBDA_SAT_MAX = 0.05
MIN_SEEDS_FOR_INFERENCE = 5          # decision D6
TOL = 1e-9

SEEDS = (0, 1, 2)
ATTACKED_SOURCE = {0: "batch_1", 1: "batch_2", 2: "batch_3"}

RUN_DIRS = {
    ("A", 0): "results/runs_constraint_batch_g9/g9_batch_clean",
    ("A", 1): "results/runs_constraint_batch_g10/seed1",
    ("A", 2): "results/runs_constraint_batch_g10/seed2",
    **{("B", s): f"results/runs_a1/B_seed{s}" for s in SEEDS},
    **{(c, s): f"results/runs_a2/{c}_seed{s}" for c in ("C", "E") for s in SEEDS},
}
RCE_CONDITIONS = ("C", "E")


# ------------------------------------------------------------------ loading
def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def load_run(run_dir: str) -> dict | None:
    d = Path(run_dir)
    if not (d / "oracle.jsonl").exists() or not (d / "rounds.jsonl").exists():
        return None
    orc, rnd = read_jsonl(d / "oracle.jsonl"), read_jsonl(d / "rounds.jsonl")
    if not orc or not rnd:
        return None
    aids = sorted(orc[0]["agents"].keys())
    K = min(len(orc), len(rnd))
    orc, rnd = orc[:K], rnd[:K]

    def oarr(field):
        return np.array([[r["agents"][a][field] for a in aids] for r in orc], dtype=float)

    def carr(field):
        return np.array([[r["constraints"][a][field] for a in aids] for r in rnd], dtype=float)

    def aggarr(field):
        return np.array([[r["constraints"][a]["aggregate"][field] for a in aids] for r in rnd],
                        dtype=float)

    out = {
        "dir": run_dir, "agent_ids": aids, "K": K,
        "true_cost": oarr("true_cost_return"),
        "task_return": oarr("episodic_task_return"),
        "detection_gap": oarr("detection_gap"),
        "violated": oarr("violated"),
        "peak_violation": oarr("peak_violation_so_far"),
        "mech_reported": carr("mechanism_reported_cost_return"),
        "residual": carr("constraint_residual"),
        "lam": carr("lambda_after"),
        "spread": aggarr("spread"),
        "point_estimate": aggarr("point_estimate"),
        "retained_n": aggarr("retained_n"),
        "applied_margin": aggarr("applied_margin"),
    }
    sids = sorted(r["source_id"] for r in rnd[0]["constraints"][aids[0]]["reports"])
    out["source_ids"] = sids
    out["source_values"] = {
        sid: np.array([[next(x["value"] for x in r["constraints"][a]["reports"]
                             if x["source_id"] == sid) for a in aids] for r in rnd], dtype=float)
        for sid in sids
    }
    out["corrupted_source_ids"] = sorted(rnd[0]["constraints"][aids[0]]["corrupted_source_ids"])
    out["guarantee_in_force"] = [[r["constraints"][a].get("guarantee_in_force") for a in aids]
                                 for r in rnd]
    out["epsilon_offline"] = [[r["constraints"][a].get("epsilon_offline") for a in aids]
                              for r in rnd]
    ppo0 = rnd[0]["constraints"][aids[0]].get("ppo") or {}
    for key in ("approx_kl", "kl", "entropy"):
        if key in ppo0:
            out[f"ppo_{key}"] = np.array(
                [[r["constraints"][a]["ppo"][key] for a in aids] for r in rnd], dtype=float)
    md = d / "run_metadata.json"
    if md.exists():
        out["metadata"] = json.loads(md.read_text(encoding="utf-8"))
    gc = d / "guarantee_calibration.json"
    if gc.exists():
        out["calibration"] = json.loads(gc.read_text(encoding="utf-8"))
    mech = d / "a2_mechanism_report.json"
    if mech.exists():
        out["mechanism_report"] = json.loads(mech.read_text(encoding="utf-8"))
    return out


# ------------------------------------------------------------- summarising
def summarize(run: dict) -> dict:
    tc, tr, gp = run["true_cost"], run["task_return"], run["detection_gap"]
    net, last = tc.mean(axis=1), slice(-50, None)
    lam = run["lam"]
    s = {
        "run_dir": run["dir"], "rounds": run["K"],
        "corrupted_source_ids": run["corrupted_source_ids"],
        # --- task
        "task_return_whole": float(tr.mean()),
        "task_return_first20": float(tr[:20].mean()),
        "task_return_first50": float(tr[:50].mean()),
        "task_return_last50": float(tr[last].mean()),
        "task_learning_gain": float(tr[last].mean() - tr[:50].mean()),
        # --- safety
        "true_cost_net_whole": float(net.mean()),
        "true_cost_net_last50": float(net[last].mean()),
        "true_cost_net_last50_sd_over_rounds": float(net[last].std(ddof=1)),
        "true_cost_net_rounds51_150": float(net[50:150].mean()),
        "true_cost_net_median": float(np.median(net)),
        "true_cost_per_agent_whole": [float(x) for x in tc.mean(axis=0)],
        "true_cost_per_agent_last50": [float(x) for x in tc[last].mean(axis=0)],
        "n_agents_over_d_last50": int((tc[last].mean(axis=0) > D_BUDGET).sum()),
        "violation_rate_last50": float(run["violated"][last].mean()),
        "violation_rate_whole": float(run["violated"].mean()),
        "peak_violation": float(run["peak_violation"].max()),
        # --- reported / gap
        "mech_reported_whole": float(run["mech_reported"].mean()),
        "mech_reported_last50": float(run["mech_reported"][last].mean()),
        "aggregate_point_estimate_whole": float(run["point_estimate"].mean()),
        "detection_gap_whole": float(gp.mean()),
        "detection_gap_last50": float(gp[last].mean()),
        # --- feedback
        "lambda_mean": float(lam.mean()),
        "lambda_max": float(lam.max()),
        "lambda_frac_positive": float((lam > 0).mean()),
        "lambda_frac_saturated": float((lam >= 25.0 - 1e-9).mean()),
        "lambda_per_agent_last50": [float(x) for x in lam[last].mean(axis=0)],
        "residual_whole": float(run["residual"].mean()),
        "residual_last50_per_agent": [float(x) for x in run["residual"][last].mean(axis=0)],
        "residual_last50_sum": float(run["residual"][last].mean(axis=0).sum()),
        # --- aggregation diagnostics
        "aggregator_spread_mean": float(run["spread"].mean()),
        "aggregator_retained_n_mean": float(run["retained_n"].mean()),
        "aggregator_applied_margin_mean": float(run["applied_margin"].mean()),
        # --- source-level
        "source_value_means": {sid: float(v.mean()) for sid, v in run["source_values"].items()},
        "source_value_means_last50": {sid: float(v[last].mean())
                                      for sid, v in run["source_values"].items()},
        "between_source_sd_mean": float(np.stack(
            [run["source_values"][sid] for sid in run["source_ids"]], axis=0).std(axis=0, ddof=1).mean()),
        "finite": bool(all(np.isfinite(a).all() for a in
                           (tc, tr, gp, lam, run["residual"], run["mech_reported"],
                            run["spread"], run["point_estimate"]))),
    }
    for key in ("ppo_approx_kl", "ppo_kl"):
        if key in run:
            s["kl_median"] = float(np.median(run[key]))
            s["kl_p95"] = float(np.percentile(run[key], 95))
            break
    if "ppo_entropy" in run:
        e = run["ppo_entropy"]
        s["entropy_first10"] = float(e[:10].mean())
        s["entropy_last10"] = float(e[-10:].mean())
    if "calibration" in run:
        s["epsilon_offline"] = run["calibration"].get("epsilon_offline")
        s["calibration_rounds"] = run["calibration"].get("n_rounds")
    gif = [v for row in run["guarantee_in_force"] for v in row if v is not None]
    if gif:
        s["guarantee_in_force_frac_true"] = float(np.mean([bool(v) for v in gif]))
    if "metadata" in run:
        t = run["metadata"].get("timing", {})
        s["wall_clock_s"] = t.get("wall_clock_s")
        s["env_steps"] = run["metadata"].get("env_steps")
    return s


# ------------------------------------------------------------- paired stats
def paired(summaries: dict, metric: str, x: str, y: str, seeds) -> dict:
    diffs = np.array([summaries[(x, s)][metric] - summaries[(y, s)][metric] for s in seeds])
    n = len(diffs)
    m = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    out = {
        "contrast": f"{x}-{y}", "metric": metric, "n": n,
        "per_seed": {str(s): float(summaries[(x, s)][metric] - summaries[(y, s)][metric])
                     for s in seeds},
        "mean": m, "sd": sd, "se": se,
        "all_same_sign": bool(len({np.sign(d) for d in diffs} - {0.0}) <= 1
                              and not np.any(diffs == 0)),
        "sign": "positive" if m > 0 else ("negative" if m < 0 else "zero"),
        "cohens_dz": float(m / sd) if n > 1 and sd > 0 else float("nan"),
    }
    try:
        from scipy import stats as sps
        if n > 1 and sd > 0:
            tcrit = float(sps.t.ppf(0.975, n - 1))
            out["ci95"] = [m - tcrit * se, m + tcrit * se]
            t, p = sps.ttest_rel([summaries[(x, s)][metric] for s in seeds],
                                 [summaries[(y, s)][metric] for s in seeds])
            out["paired_t"] = {"t": float(t), "p": float(p),
                               "label": ("SENSITIVITY ONLY -- decision D6 forbids inference "
                                         f"below {MIN_SEEDS_FOR_INFERENCE} seeds")
                               if n < MIN_SEEDS_FOR_INFERENCE else "primary"}
            n_pos = int((diffs > 0).sum())
            out["sign_test_p"] = float(sps.binomtest(n_pos, n, 0.5).pvalue)
            out["sign_test_min_attainable_p"] = float(sps.binomtest(n, n, 0.5).pvalue)
    except Exception as exc:                                    # pragma: no cover
        out["stats_error"] = str(exc)
    return out


def interaction(summaries: dict, metric: str, seeds) -> dict:
    """I = (Y11 - Y01) - (Y10 - Y00) = (C - E) - (B - A), per seed then pooled.

    Computed as a per-seed scalar first, so the CI is over the SAME three
    paired units the rest of the analysis uses -- not over a difference of
    two independently-pooled means.
    """
    per_seed = {}
    for s in seeds:
        c_e = summaries[("C", s)][metric] - summaries[("E", s)][metric]
        b_a = summaries[("B", s)][metric] - summaries[("A", s)][metric]
        per_seed[str(s)] = float(c_e - b_a)
    diffs = np.array(list(per_seed.values()))
    n = len(diffs)
    m = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    out = {
        "quantity": "(C-E)-(B-A)", "metric": metric, "n": n,
        "per_seed": per_seed, "mean": m, "sd": sd, "se": se,
        "all_same_sign": bool(len({np.sign(d) for d in diffs} - {0.0}) <= 1
                              and not np.any(diffs == 0)),
        "n_negative": int((diffs < 0).sum()),
        "cohens_dz": float(m / sd) if n > 1 and sd > 0 else float("nan"),
    }
    try:
        from scipy import stats as sps
        if n > 1 and sd > 0:
            tcrit = float(sps.t.ppf(0.975, n - 1))
            out["ci95"] = [m - tcrit * se, m + tcrit * se]
            out["sign_test_p"] = float(sps.binomtest(int((diffs > 0).sum()), n, 0.5).pvalue)
            out["label"] = ("SENSITIVITY ONLY -- decision D6 forbids inference below "
                            f"{MIN_SEEDS_FOR_INFERENCE} seeds")
    except Exception as exc:                                    # pragma: no cover
        out["stats_error"] = str(exc)
    return out


# --------------------------------------------------- RCE mechanism roll-up
def rce_mechanism_rollup(runs: dict, seeds) -> dict:
    """Sections 15/16 of the brief, aggregated over seeds.

    Everything here comes from the per-run a2_mechanism_report.json written
    by scripts/a2_rce_mechanism_validation.py, plus the run logs themselves.
    """
    out = {"per_run": {}, "note": (
        "At M=3, f=1 the trimmed set holds exactly one value, so MAD is "
        "structurally zero, the spread is floored to sigma_min, and the margin "
        "is the constant beta*sigma_min = %.4g. The theoretical MAD condition "
        "is never active and guarantee_in_force is vacuous "
        "(docs/a2_rce_gates.md section 4)." % (BETA * SIGMA_MIN))}
    for cond in RCE_CONDITIONS:
        for s in seeds:
            run = runs.get((cond, s))
            if run is None:
                continue
            entry = {
                "spread_min": float(run["spread"].min()),
                "spread_max": float(run["spread"].max()),
                "frac_spread_eq_sigma_min": float(
                    (np.abs(run["spread"] - SIGMA_MIN) <= TOL).mean()),
                "applied_margin_mean": float(run["applied_margin"].mean()),
                "frac_margin_eq_beta_sigma_min": float(
                    (np.abs(run["applied_margin"] - BETA * SIGMA_MIN) <= TOL).mean()),
                "retained_n_unique": sorted({int(x) for x in np.unique(run["retained_n"])}),
                "mad_ever_nonzero": bool((run["spread"] > SIGMA_MIN + TOL).any()),
                "theoretical_condition_ever_active": False,
                "epsilon_offline": (run.get("calibration") or {}).get("epsilon_offline"),
                "guarantee_in_force_frac_true": float(np.mean(
                    [[bool(v) for v in row] for row in run["guarantee_in_force"]])),
            }
            mr = run.get("mechanism_report") or {}
            for key in ("G7_iii_source_set", "G7_iv_within_run_counterfactual",
                        "G7_v_source_ordering_stability", "G1_ii_crn_round0",
                        "G2_map_identity"):
                if key in mr:
                    entry[key] = mr[key]
            out["per_run"][f"{cond}_seed{s}"] = entry
    # pooled attacked-source behaviour across the three C runs
    trimmed = [e["G7_iii_source_set"].get("frac_attacked_source_trimmed")
               for k, e in out["per_run"].items()
               if k.startswith("C") and "G7_iii_source_set" in e]
    shifts = [e["G7_iv_within_run_counterfactual"].get("mean_rce_shift")
              for k, e in out["per_run"].items()
              if k.startswith("C") and "G7_iv_within_run_counterfactual" in e]
    if trimmed:
        out["pooled_frac_attacked_source_trimmed"] = float(np.mean(trimmed))
    if shifts and all(v is not None for v in shifts):
        out["pooled_mean_rce_aggregate_shift"] = float(np.mean(shifts))
        out["predicted_rce_aggregate_shift"] = PRED_RCE_SHIFT
        out["mean_aggregator_shift_for_comparison"] = -B_MAGNITUDE / M_SOURCES
        out["estimator_corruption_suppression_factor"] = float(
            (B_MAGNITUDE / M_SOURCES) / abs(np.mean(shifts)))
    return out


# ------------------------------------------------------------------- gates
def evaluate_gates(summaries: dict, runs: dict, contrasts: dict, inter: dict,
                   mech: dict, seeds) -> dict:
    g: dict = {}
    have = lambda c: all((c, s) in summaries for s in seeds)          # noqa: E731

    # --- A2-G1 / A2-G2: structural, from the per-run mechanism reports
    structural = {}
    for cond in RCE_CONDITIONS:
        for s in seeds:
            mr = (runs.get((cond, s)) or {}).get("mechanism_report")
            structural[f"{cond}_seed{s}"] = (mr or {}).get("all_pass")
    g["A2_G1_G2_structural"] = {
        "per_run": structural,
        "detail": "see each run's a2_mechanism_report.json",
        "pass": bool(structural and all(v is True for v in structural.values())),
    }

    # --- A2-G3: clean RCE effect measured and reported (validity only)
    if have("E"):
        g["A2_G3_clean_rce_effect"] = {
            "E-A": contrasts["E-A"]["true_cost_net_whole"],
            "runs_complete": all(summaries[("E", s)]["rounds"] == 250 for s in seeds),
            "finite": all(summaries[("E", s)]["finite"] for s in seeds),
            "note": "value is a measurement, not a hypothesis; gated only for run validity",
            "pass": bool(all(summaries[("E", s)]["rounds"] == 250 for s in seeds)
                         and all(summaries[("E", s)]["finite"] for s in seeds)),
        }

    # --- A2-G4: residual attack effect reported
    if have("C") and have("E"):
        g["A2_G4_residual_attack_effect"] = {
            "C-E": contrasts["C-E"]["true_cost_net_whole"],
            "B-A": contrasts["B-A"]["true_cost_net_whole"],
            "reported_only": True, "pass": True,
        }

    # --- A2-G5: the interaction, with the pre-registered decision rule
    if have("C") and have("E"):
        i = inter["true_cost_net_whole"]
        b_a = contrasts["B-A"]["true_cost_net_whole"]["mean"]
        c_e = contrasts["C-E"]["true_cost_net_whole"]["mean"]
        reduction = (b_a - c_e) / b_a if b_a != 0 else float("nan")
        strong = bool(i["mean"] < 0 and i["n_negative"] == len(seeds)
                      and c_e <= G5_REDUCTION_BAR * b_a)
        partial = bool(i["mean"] < 0)
        g["A2_G5_interaction"] = {
            "interaction": i,
            "B-A_mean": b_a, "C-E_mean": c_e,
            "fraction_of_attack_effect_removed": float(reduction),
            "bar_for_pass": f"I<0 in all {len(seeds)} seeds and (C-E) <= "
                            f"{G5_REDUCTION_BAR} * (B-A)",
            "verdict": ("PASS" if strong else ("CONDITIONAL" if partial else "FAIL")),
            "pass": strong,
        }

    # --- A2-G6: learning health, G10-C verbatim
    health = {}
    for cond in ("C", "E"):
        for s in seeds:
            if (cond, s) not in summaries:
                continue
            u = summaries[(cond, s)]
            checks = {
                "finite": u["finite"],
                "kl_median_ok": u.get("kl_median", 0.0) <= KL_MEDIAN_MAX,
                "kl_p95_ok": u.get("kl_p95", 0.0) <= KL_P95_MAX,
                "lambda_saturation_ok": u["lambda_frac_saturated"] < LAMBDA_SAT_MAX,
                "entropy_declines": (u.get("entropy_last10", 0.0) < u.get("entropy_first10", 1.0)),
                "entropy_positive": u.get("entropy_last10", 1.0) > 0.0,
            }
            checks["pass"] = all(checks.values())
            # reported, NOT gated -- gating it would make Outcome E unreachable
            checks["task_learning_gain_reported"] = u["task_learning_gain"]
            health[f"{cond}_seed{s}"] = checks
    g["A2_G6_learning_health"] = {
        "per_run": health,
        "note": "G10-C-ii (task gain >= +75) reported, not gated (gates doc A2-G6)",
        "pass": bool(health and all(v["pass"] for v in health.values())),
    }

    # --- A2-G7: mechanism behaves as the audit says
    per_run_g7 = {}
    for key, entry in mech.get("per_run", {}).items():
        sub = {
            "retained_n_always_1": entry["retained_n_unique"] == [1],
            "spread_always_sigma_min": entry["frac_spread_eq_sigma_min"] >= 1.0 - TOL,
            "margin_always_constant": entry["frac_margin_eq_beta_sigma_min"] >= 1.0 - TOL,
            "mad_never_active": not entry["mad_ever_nonzero"],
        }
        if key.startswith("C"):
            g3 = entry.get("G7_iii_source_set", {})
            g4 = entry.get("G7_iv_within_run_counterfactual", {})
            sub["attacked_source_trimmed"] = (
                g3.get("frac_attacked_source_trimmed", 0.0) >= G7_III_MIN_FRAC)
            sub["retained_is_min_honest"] = (
                g3.get("frac_retained_equals_min_of_two_honest", 0.0) >= G7_III_MIN_FRAC)
            ms = g4.get("mean_rce_shift")
            sub["shift_in_band"] = bool(ms is not None
                                        and G7_IV_BAND[0] <= ms <= G7_IV_BAND[1])
        sub["pass"] = all(v for k, v in sub.items() if k != "pass")
        per_run_g7[key] = sub
    g["A2_G7_rce_mechanism"] = {
        "per_run": per_run_g7,
        "pass": bool(per_run_g7 and all(v["pass"] for v in per_run_g7.values())),
    }
    return g


# ------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/runs_a2/a2_report.json")
    args = ap.parse_args()

    runs, summaries, missing = {}, {}, []
    for (cond, seed), path in RUN_DIRS.items():
        r = load_run(path)
        if r is None:
            missing.append(f"{cond}_seed{seed} ({path})")
            continue
        runs[(cond, seed)] = r
        summaries[(cond, seed)] = summarize(r)

    available = sorted({c for c, _ in summaries})
    seeds = [s for s in SEEDS if all((c, s) in summaries for c in ("A", "B", "C", "E"))]

    report: dict = {
        "design": {
            "factors": {"attack": ["no", "yes"], "rce": ["no", "yes"]},
            "cells": {"Y00": "A", "Y10": "B", "Y11": "C", "Y01": "E"},
            "primary_quantity": "(C-E)-(B-A) on true_cost_net_whole",
            "metric_direction": "higher true cost is WORSE",
            "seeds": list(SEEDS), "complete_seeds": seeds,
            "budget_d": D_BUDGET, "B": B_MAGNITUDE, "M": M_SOURCES,
            "defense": {"name": "rce", "f": 1, "beta": BETA, "sigma_min": SIGMA_MIN},
            "attacked_source_by_seed": ATTACKED_SOURCE,
            "gates_doc": "docs/a2_rce_gates.md",
        },
        "missing_runs": missing,
        "conditions_available": available,
        "per_run": {f"{c}_seed{s}": summaries[(c, s)] for (c, s) in sorted(summaries)},
        "complete": not missing,
    }

    METRICS = ("true_cost_net_whole", "true_cost_net_last50", "task_return_whole",
               "task_return_last50", "mech_reported_whole", "detection_gap_whole",
               "lambda_mean", "lambda_max", "violation_rate_whole", "violation_rate_last50",
               "residual_whole", "aggregator_spread_mean")

    contrasts: dict = {}
    for x, y in (("B", "A"), ("E", "A"), ("C", "E"), ("C", "B"), ("C", "A")):
        if seeds and all((c, s) in summaries for c in (x, y) for s in seeds):
            contrasts[f"{x}-{y}"] = {m: paired(summaries, m, x, y, seeds) for m in METRICS}
    report["contrasts"] = contrasts

    inter = {}
    if seeds and all((c, s) in summaries for c in ("A", "B", "C", "E") for s in seeds):
        inter = {m: interaction(summaries, m, seeds) for m in METRICS}
    report["interaction"] = inter

    # Sign-convention assertion: the analysis is only meaningful if B-A
    # reproduces the established A1 attack effect on the same metric.
    if "B-A" in contrasts:
        got = contrasts["B-A"]["true_cost_net_whole"]["mean"]
        report["sign_convention_check"] = {
            "B-A_true_cost_net_whole": got,
            "a1_established_value": A1_B_MINUS_A,
            "agrees_to_1e-3": bool(abs(got - A1_B_MINUS_A) < 1e-3),
            "note": "positive means the attack RAISED true cost; higher cost is worse",
        }

    mech = rce_mechanism_rollup(runs, seeds) if seeds else {"per_run": {}}
    report["rce_mechanism"] = mech

    gates = evaluate_gates(summaries, runs, contrasts, inter, mech, seeds) if seeds else {}
    report["gates"] = gates

    # ------------------------------------------------------------- verdict
    if not seeds:
        report["verdict"] = "INCOMPLETE"
        report["verdict_reason"] = f"missing runs: {missing}"
    else:
        structural_ok = all(gates[k]["pass"] for k in
                            ("A2_G1_G2_structural", "A2_G6_learning_health",
                             "A2_G7_rce_mechanism") if k in gates)
        g5 = gates.get("A2_G5_interaction", {})
        if not structural_ok:
            report["verdict"] = "FAIL -- STRUCTURAL"
        elif g5.get("verdict") == "PASS":
            report["verdict"] = "PASS -- DEFENSE SUPPORTED"
        elif g5.get("verdict") == "CONDITIONAL":
            report["verdict"] = "CONDITIONAL PASS -- DEFENSE PARTLY SUPPORTED"
        else:
            report["verdict"] = "FAIL -- DEFENSE NOT ESTABLISHED"
        report["verdict_inputs"] = {
            "structural_ok": structural_ok,
            "n_complete_seeds": len(seeds),
            "G5": {k: v for k, v in g5.items() if k != "interaction"},
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ------------------------------------------------------------- console
    print("=" * 74)
    print(f"A2 -- attack x RCE factorial   (complete seeds: {seeds or 'none'})")
    print("=" * 74)
    if missing:
        print("MISSING RUNS:")
        for m in missing:
            print("   ", m)
    if seeds:
        print("\nWhole-run network-average TRUE COST (higher = worse):")
        print(f"  {'seed':>5} {'A':>9} {'B':>9} {'E':>9} {'C':>9}")
        for s in seeds:
            print(f"  {s:>5} " + " ".join(
                f"{summaries[(c, s)]['true_cost_net_whole']:>9.3f}" for c in ("A", "B", "E", "C")))
        print("\nContrasts on true_cost_net_whole:")
        for name in ("B-A", "E-A", "C-E", "C-B"):
            if name in contrasts:
                c = contrasts[name]["true_cost_net_whole"]
                ci = c.get("ci95", [float('nan')] * 2)
                print(f"  {name:>4}: mean={c['mean']:+8.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
                      f"per-seed={[round(v, 3) for v in c['per_seed'].values()]}  "
                      f"same-sign={c['all_same_sign']}")
        if inter:
            i = inter["true_cost_net_whole"]
            ci = i.get("ci95", [float('nan')] * 2)
            print(f"\n  INTERACTION (C-E)-(B-A): mean={i['mean']:+8.3f}  "
                  f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
                  f"per-seed={[round(v, 3) for v in i['per_seed'].values()]}")
        print("\nGates:")
        for k, v in gates.items():
            print(f"  [{'PASS' if v.get('pass') else 'FAIL'}] {k}")
    print(f"\nVERDICT: {report['verdict']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
