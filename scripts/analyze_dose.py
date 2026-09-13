#!/usr/bin/env python
"""DR dose-sweep analyser -- B/d in {0, 0.25, 0.5, 1.0}.

Implements docs/dose_response_gates.md exactly. Every bar is read from
that document; none is chosen here.

The design is paired by seed at every dose (gates doc 4.1), so every
contrast is computed as a per-seed difference and then summarised over
seeds. Per decision D6 (`safelie.analysis.stats.MIN_SEEDS_FOR_INFERENCE
= 5`) and gates doc 9.1, at three seeds the primary evidence is per-seed
sign and ordering consistency; the paired t-test is emitted only as a
labelled sensitivity analysis, alongside the exact sign test.

**No functional form is fitted.** Gates doc 9.2 forbids a regression,
slope, intercept or R^2, and none is computed here. The mechanistic
expectation `8.3333 * (B/d)` is evaluated as a per-dose per-seed RATIO
(DR-G3), which is a descriptive diagnostic, not a model.

The gap reported is the MECHANISM gap, `true_cost_return -
mechanism_reported_cost_return`, not `oracle.jsonl:detection_gap`, which
is defined against the own critic and would swamp any dose signal
(gates doc 6).

Usage:
    python scripts/analyze_dose.py --out results/runs_dose/dose_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- constants
# All from docs/dose_response_gates.md. Do not tune here.
D_BUDGET = 25.0
M_SOURCES = 3
SEEDS = (0, 1, 2)
DOSES = (0.0, 0.25, 0.5, 1.0)               # gates doc 3
G1_III_MIN_EFFECT = 1.0                     # gates doc 8, DR-G1-iii, cost units
G5_II_MIN_AGENTS = 4                        # gates doc 8, DR-G5-ii, of 6
DR_G3_RATIO_BAND = (0.7, 1.3)               # gates doc 8, DR-G3 "consistent" band
DELTA_EQUIV = 15.3                          # gates doc 8, DR-G7, task-return units
DELTA_EQUIV_PAPER = 1.36                    # DR-G7 secondary, one clean between-seed sd
KL_MEDIAN_MAX = 0.0069                      # DR-G6 (A1-G6 / G10-C-iii verbatim)
KL_P95_MAX = 0.01548
LAMBDA_SAT_MAX = 0.05
MIN_SEEDS_FOR_INFERENCE = 5                 # decision D6
TOL = 1e-9

ATTACKED_SOURCE = {0: "batch_1", 1: "batch_2", 2: "batch_3"}   # gates doc 3.2

# (dose, seed) -> run directory. The 0.0 and 0.5 arms are the already
# committed reference runs; they are READ, never rewritten.
RUN_DIRS: dict[tuple[float, int], str] = {
    (0.0, 0): "results/runs_constraint_batch_g9/g9_batch_clean",
    (0.0, 1): "results/runs_constraint_batch_g10/seed1",
    (0.0, 2): "results/runs_constraint_batch_g10/seed2",
    **{(0.5, s): f"results/runs_a1/B_seed{s}" for s in SEEDS},
    **{(0.25, s): f"results/runs_dose/B025_seed{s}" for s in SEEDS},
    **{(1.0, s): f"results/runs_dose/B100_seed{s}" for s in SEEDS},
}


def dose_label(dose: float) -> str:
    return f"B/d={dose:g}"


# ------------------------------------------------------------------ loading
def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def load_run(run_dir: str) -> dict | None:
    d = ROOT / run_dir
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
        return np.array([[r["constraints"][a][field] for a in aids] for r in rnd],
                        dtype=float)

    out = {
        "dir": run_dir, "agent_ids": aids, "K": K,
        "true_cost": oarr("true_cost_return"),
        "task_return": oarr("episodic_task_return"),
        "own_critic_gap": oarr("detection_gap"),
        "violated": oarr("violated"),
        "peak_violation": oarr("peak_violation_so_far"),
        "mech_reported": carr("mechanism_reported_cost_return"),
        "residual": carr("constraint_residual"),
        "lam": carr("lambda_after"),
        "spread": np.array([[r["constraints"][a]["aggregate"]["spread"] for a in aids]
                            for r in rnd], dtype=float),
        "corrupted_source_ids": sorted(
            rnd[0]["constraints"][aids[0]].get("corrupted_source_ids") or []),
        "rounds_raw": rnd,
    }
    # The mechanism gap -- gates doc 6.
    out["mech_gap"] = out["true_cost"] - out["mech_reported"]

    ppo0 = rnd[0]["constraints"][aids[0]].get("ppo") or {}
    for key in ("approx_kl", "kl"):
        if key in ppo0:
            out["ppo_kl"] = np.array(
                [[r["constraints"][a]["ppo"][key] for a in aids] for r in rnd], dtype=float)
            break
    if "entropy" in ppo0:
        out["ppo_entropy"] = np.array(
            [[r["constraints"][a]["ppo"]["entropy"] for a in aids] for r in rnd], dtype=float)

    md = d / "run_metadata.json"
    if md.exists():
        out["metadata"] = json.loads(md.read_text(encoding="utf-8"))
    return out


# ------------------------------------------------------------- summarising
def summarize(run: dict) -> dict:
    tc, tr = run["true_cost"], run["task_return"]
    mg, lam = run["mech_gap"], run["lam"]
    net, last = tc.mean(axis=1), slice(-50, None)
    mg_net = mg.mean(axis=1)
    s = {
        "run_dir": run["dir"], "rounds": run["K"],
        "corrupted_source_ids": run["corrupted_source_ids"],
        # --- task
        "task_return_whole": float(tr.mean()),
        "task_return_first20": float(tr[:20].mean()),
        "task_return_first50": float(tr[:50].mean()),
        "task_return_last50": float(tr[last].mean()),
        "task_learning_gain": float(tr[last].mean() - tr[:50].mean()),
        # --- safety (PRIMARY: network average of per-agent expected cost)
        "true_cost_net_whole": float(net.mean()),
        "true_cost_net_last50": float(net[last].mean()),
        "true_cost_net_last50_sd_over_rounds": float(net[last].std(ddof=1)),
        "true_cost_per_agent_whole": [float(x) for x in tc.mean(axis=0)],
        "true_cost_per_agent_last50": [float(x) for x in tc[last].mean(axis=0)],
        "n_agents_over_d_last50": int((tc[last].mean(axis=0) > D_BUDGET).sum()),
        "violation_rate_whole": float(run["violated"].mean()),
        "violation_rate_last50": float(run["violated"][last].mean()),
        "peak_violation": float(run["peak_violation"].max()),
        # --- reported / MECHANISM gap (gates doc 6)
        "mech_reported_whole": float(run["mech_reported"].mean()),
        "mech_reported_last50": float(run["mech_reported"][last].mean()),
        "mech_gap_whole": float(mg_net.mean()),
        "mech_gap_last50": float(mg_net[last].mean()),
        "mech_gap_per_agent_whole": [float(x) for x in mg.mean(axis=0)],
        # reported for continuity with A1's tables, never gated here
        "own_critic_gap_whole": float(run["own_critic_gap"].mean()),
        "own_critic_gap_last50": float(run["own_critic_gap"][last].mean()),
        # --- feedback
        "lambda_mean": float(lam.mean()),
        "lambda_max": float(lam.max()),
        "lambda_frac_positive": float((lam > 0).mean()),
        "lambda_frac_saturated": float((lam >= 25.0 - 1e-9).mean()),
        "lambda_per_agent_last50": [float(x) for x in lam[last].mean(axis=0)],
        "residual_last50_sum": float(run["residual"][last].mean(axis=0).sum()),
        "source_spread_mean": float(run["spread"].mean()),
        "finite": bool(all(np.isfinite(a).all() for a in
                           (tc, tr, mg, lam, run["residual"], run["mech_reported"]))),
    }
    if "ppo_kl" in run:
        s["kl_median"] = float(np.median(run["ppo_kl"]))
        s["kl_p95"] = float(np.percentile(run["ppo_kl"], 95))
    if "ppo_entropy" in run:
        e = run["ppo_entropy"]
        s["entropy_first10"] = float(e[:10].mean())
        s["entropy_last10"] = float(e[-10:].mean())
    if "metadata" in run:
        t = run["metadata"].get("timing", {})
        s["wall_clock_s"] = t.get("wall_clock_s")
        s["git_sha"] = (run["metadata"].get("git") or {}).get("sha")
        snap = run["metadata"].get("config_snapshot", {})
        s["config_budget_ratio"] = (snap.get("attack") or {}).get("budget_ratio", 0.0)
        s["config_defense"] = (snap.get("defense") or {}).get("name")
        prov = run["metadata"].get("provenance") or {}
        s["provenance_cpu"] = prov.get("cpu_model")
        s["provenance_arch"] = prov.get("architecture")
    return s


# ------------------------------------------------------------- paired stats
def paired_vals(summaries, metric, dose_x, dose_y, seeds):
    return np.array([summaries[(dose_x, s)][metric] - summaries[(dose_y, s)][metric]
                     for s in seeds], dtype=float)


def paired(summaries, metric, dose_x, dose_y, seeds) -> dict:
    diffs = paired_vals(summaries, metric, dose_x, dose_y, seeds)
    n = len(diffs)
    m = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    out = {
        "contrast": f"{dose_label(dose_x)} - {dose_label(dose_y)}",
        "metric": metric, "n": n,
        "per_seed": {str(s): float(d) for s, d in zip(seeds, diffs, strict=True)},
        "mean": m, "sd": sd, "se": se,
        "all_positive": bool(np.all(diffs > 0)),
        "all_same_sign": bool(np.all(diffs > 0) or np.all(diffs < 0)),
        "sign": "positive" if m > 0 else ("negative" if m < 0 else "zero"),
        "cohens_dz": float(m / sd) if n > 1 and sd > 0 else float("nan"),
    }
    try:
        from scipy import stats as sps
        if n > 1 and sd > 0:
            tcrit = float(sps.t.ppf(0.975, n - 1))
            out["ci95"] = [m - tcrit * se, m + tcrit * se]
            t, p = sps.ttest_rel(
                [summaries[(dose_x, s)][metric] for s in seeds],
                [summaries[(dose_y, s)][metric] for s in seeds])
            out["paired_t"] = {
                "t": float(t), "p": float(p),
                "label": ("SENSITIVITY ONLY -- decision D6 forbids inference below "
                          f"{MIN_SEEDS_FOR_INFERENCE} seeds"
                          if n < MIN_SEEDS_FOR_INFERENCE else "primary")}
            out["sign_test_p"] = float(sps.binomtest(int((diffs > 0).sum()), n, 0.5).pvalue)
            out["sign_test_min_attainable_p"] = float(sps.binomtest(n, n, 0.5).pvalue)
    except Exception as exc:                                   # pragma: no cover
        out["stats_error"] = str(exc)
    return out


def summarize_array(diffs: np.ndarray, label: str, seeds) -> dict:
    n = len(diffs)
    m = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    out = {"quantity": label, "n": n,
           "per_seed": {str(s): float(d) for s, d in zip(seeds, diffs, strict=True)},
           "mean": m, "sd": sd, "se": se,
           "all_positive": bool(np.all(diffs > 0))}
    try:
        from scipy import stats as sps
        if n > 1 and sd > 0:
            tcrit = float(sps.t.ppf(0.975, n - 1))
            out["ci95"] = [m - tcrit * se, m + tcrit * se]
    except Exception as exc:                                   # pragma: no cover
        out["stats_error"] = str(exc)
    return out


def tost(summaries, metric, dose_x, dose_y, seeds, margin) -> dict:
    diffs = paired_vals(summaries, metric, dose_x, dose_y, seeds)
    n = len(diffs)
    res = {"contrast": f"{dose_label(dose_x)} - {dose_label(dose_y)}",
           "metric": metric, "margin": margin, "n": n, "mean": float(diffs.mean())}
    if n < 2:
        res["equivalent"] = None
        return res
    sd = float(diffs.std(ddof=1))
    se = sd / math.sqrt(n)
    res.update(sd=sd, se=se)
    try:
        from scipy import stats as sps
        p_lo = float(sps.t.sf((diffs.mean() + margin) / se, n - 1))
        p_hi = float(sps.t.cdf((diffs.mean() - margin) / se, n - 1))
        res["p_lower"], res["p_upper"] = p_lo, p_hi
        res["p_tost"] = max(p_lo, p_hi)
        res["equivalent"] = bool(max(p_lo, p_hi) < 0.05)
        tcrit = float(sps.t.ppf(0.95, n - 1))
        res["ci90"] = [diffs.mean() - tcrit * se, diffs.mean() + tcrit * se]
    except Exception as exc:                                   # pragma: no cover
        res["stats_error"] = str(exc)
    return res


# ------------------------------------------------------------------- gates
def evaluate_gates(summaries, seeds, doses) -> dict:
    g: dict = {}
    nonzero = [d for d in doses if d > 0]
    ordered = sorted(doses)

    # ---- DR-G1 ----------------------------------------------------------
    g1_i = {dose_label(d): paired(summaries, "true_cost_net_whole", d, 0.0, seeds)
            for d in nonzero}
    g1_i_pass = all(v["all_positive"] for v in g1_i.values())

    # G1-ii: all adjacent increments positive, every seed. 9 cells at 4 doses.
    increments = {}
    for lo, hi in zip(ordered, ordered[1:], strict=False):
        increments[f"{dose_label(lo)} -> {dose_label(hi)}"] = paired(
            summaries, "true_cost_net_whole", hi, lo, seeds)
    n_inc_cells = sum(len(v["per_seed"]) for v in increments.values())
    n_inc_pos = sum(1 for v in increments.values() for x in v["per_seed"].values() if x > 0)
    g1_ii_pass = n_inc_pos == n_inc_cells

    # G1-iii: the dose-range contrast, on paired DELTAS vs clean.
    hi_d, lo_d = max(nonzero), min(nonzero)
    range_diffs = (paired_vals(summaries, "true_cost_net_whole", hi_d, 0.0, seeds)
                   - paired_vals(summaries, "true_cost_net_whole", lo_d, 0.0, seeds))
    g1_iii = summarize_array(
        range_diffs, f"Delta J_true({dose_label(hi_d)}) - Delta J_true({dose_label(lo_d)})",
        seeds)
    g1_iii_pass = bool(g1_iii["mean"] > G1_III_MIN_EFFECT)

    g["DR_G1"] = {
        "G1_i_delta_positive_all_seeds_every_dose": {
            "pass": g1_i_pass, "per_dose": g1_i},
        "G1_ii_monotone_ordering_all_seeds": {
            "pass": g1_ii_pass,
            "criterion": "all adjacent-dose increments positive in every seed",
            "cells_positive": n_inc_pos, "cells_total": n_inc_cells,
            "increments": increments},
        "G1_iii_dose_range_effect": {
            "pass": g1_iii_pass, "bar": G1_III_MIN_EFFECT, **g1_iii},
        "pass": bool(g1_i_pass and g1_ii_pass and g1_iii_pass),
    }
    # Secondary window, reported only (gates doc 9.3).
    g["DR_G1_last50_reported_not_gated"] = {
        dose_label(d): paired(summaries, "true_cost_net_last50", d, 0.0, seeds)
        for d in nonzero}

    # ---- DR-G2: the mechanism gap --------------------------------------
    g2_i = {dose_label(d): paired(summaries, "mech_gap_whole", d, 0.0, seeds)
            for d in nonzero}
    g2_inc = {f"{dose_label(lo)} -> {dose_label(hi)}":
              paired(summaries, "mech_gap_whole", hi, lo, seeds)
              for lo, hi in zip(ordered, ordered[1:], strict=False)}
    g2_cells = sum(len(v["per_seed"]) for v in g2_inc.values())
    g2_pos = sum(1 for v in g2_inc.values() for x in v["per_seed"].values() if x > 0)
    g["DR_G2"] = {
        "G2_i_positive_all_seeds_every_dose": {
            "pass": all(v["all_positive"] for v in g2_i.values()), "per_dose": g2_i},
        "G2_ii_monotone_ordering_all_seeds": {
            "pass": g2_pos == g2_cells, "cells_positive": g2_pos,
            "cells_total": g2_cells, "increments": g2_inc},
        "G2_iii_absolute_reported_not_gated": {
            dose_label(d): {str(s): summaries[(d, s)]["mech_gap_whole"] for s in seeds}
            for d in ordered},
        "pass": bool(all(v["all_positive"] for v in g2_i.values())
                     and g2_pos == g2_cells),
    }

    # ---- DR-G3: the mechanistic expectation. REPORTED, NOT GATED -------
    # ratio = observed Delta J_true / (B/M).  NOT a fit (gates doc 9.2).
    g3: dict = {"expectation": "Delta J_true(dose) ~= (B/d) * d / M = 8.3333 * (B/d)",
                "not_a_fit": "no regression, slope, intercept or R^2 is computed "
                             "(gates doc 9.2)",
                "band": list(DR_G3_RATIO_BAND), "per_dose": {}}
    all_in_band = True
    for d in nonzero:
        pred = d * D_BUDGET / M_SOURCES
        obs = paired_vals(summaries, "true_cost_net_whole", d, 0.0, seeds)
        ratios = obs / pred
        in_band = bool(np.all((ratios >= DR_G3_RATIO_BAND[0])
                              & (ratios <= DR_G3_RATIO_BAND[1])))
        all_in_band &= in_band
        g3["per_dose"][dose_label(d)] = {
            "B": d * D_BUDGET, "predicted_delta": pred,
            "observed_delta_per_seed": {str(s): float(x) for s, x in zip(seeds, obs, strict=True)},
            "observed_delta_mean": float(obs.mean()),
            "ratio_per_seed": {str(s): float(r) for s, r in zip(seeds, ratios, strict=True)},
            "ratio_mean": float(ratios.mean()),
            "within_band": in_band,
        }
    g3["verdict"] = "consistent" if all_in_band else "flagged"
    g3["note"] = ("a flag is a finding to describe (attenuation / saturation, "
                  "gates doc 10.1 Outcome C), not a failure; B is not re-tuned")
    g["DR_G3_reported_not_gated"] = g3

    # ---- DR-G4: the corruption reached the dual ------------------------
    lam_vs_clean = {dose_label(d): paired(summaries, "lambda_mean", d, 0.0, seeds)
                    for d in nonzero}
    g4_ii_pass = all(all(x < 0 for x in v["per_seed"].values())
                     for v in lam_vs_clean.values())
    lam_inc = {f"{dose_label(lo)} -> {dose_label(hi)}":
               paired(summaries, "lambda_mean", hi, lo, seeds)
               for lo, hi in zip(ordered, ordered[1:], strict=False)}
    g["DR_G4"] = {
        "G4_i_mechanism_identity": {
            "note": "gated per run by scripts/dose_verify_run.py (DR-S2/S3) and "
                    "scripts/a1_mechanism_validation.py at this dose's --B; "
                    "see each run's dose_run_verification.json and "
                    "dose_mechanism_report.json",
            "pass": None},
        "G4_ii_attack_lowers_lambda": {"pass": g4_ii_pass, "per_dose": lam_vs_clean},
        "G4_iii_reported_not_gated": {
            "lambda_mean_by_dose": {dose_label(d):
                                    {str(s): summaries[(d, s)]["lambda_mean"] for s in seeds}
                                    for d in ordered},
            "lambda_frac_positive_by_dose": {
                dose_label(d): {str(s): summaries[(d, s)]["lambda_frac_positive"]
                                for s in seeds} for d in ordered},
            "lambda_frac_saturated_by_dose": {
                dose_label(d): {str(s): summaries[(d, s)]["lambda_frac_saturated"]
                                for s in seeds} for d in ordered},
            "lambda_adjacent_increments": lam_inc,
            "lambda_monotone_decreasing_all_seeds": bool(all(
                all(x < 0 for x in v["per_seed"].values()) for v in lam_inc.values())),
            "cumulative_injected_bias_K_times_B": {
                dose_label(d): 250 * d * D_BUDGET for d in ordered},
            "mech_reported_by_dose": {
                dose_label(d): {str(s): summaries[(d, s)]["mech_reported_whole"]
                                for s in seeds} for d in ordered}},
        "pass": bool(g4_ii_pass),
    }

    # ---- DR-G5: per-agent ----------------------------------------------
    per_agent: dict = {}
    g5_ii_pass = True
    for d in nonzero:
        by_seed = {}
        for s in seeds:
            inc = [summaries[(d, s)]["true_cost_per_agent_whole"][i]
                   - summaries[(0.0, s)]["true_cost_per_agent_whole"][i]
                   for i in range(len(summaries[(d, s)]["true_cost_per_agent_whole"]))]
            net_sign = np.sign(summaries[(d, s)]["true_cost_net_whole"]
                               - summaries[(0.0, s)]["true_cost_net_whole"])
            agree = int(sum(1 for x in inc if np.sign(x) == net_sign))
            by_seed[str(s)] = {"per_agent_increment": [float(x) for x in inc],
                               "n_agreeing_with_network": agree,
                               "required": G5_II_MIN_AGENTS,
                               "pass": agree >= G5_II_MIN_AGENTS}
            g5_ii_pass &= agree >= G5_II_MIN_AGENTS
        per_agent[dose_label(d)] = by_seed
    g["DR_G5"] = {
        "G5_i_reported_not_gated": {
            "n_agents_over_d_last50_by_dose": {
                dose_label(d): {str(s): summaries[(d, s)]["n_agents_over_d_last50"]
                                for s in seeds} for d in ordered}},
        "G5_ii_sign_agreement": {"pass": bool(g5_ii_pass), "per_dose": per_agent},
        "pass": bool(g5_ii_pass),
    }

    # ---- DR-G6: learning health ----------------------------------------
    health: dict = {}
    g6_pass = True
    for d in ordered:
        for s in seeds:
            v = summaries[(d, s)]
            ok = bool(v["finite"] and v["lambda_frac_saturated"] < LAMBDA_SAT_MAX)
            detail = {"finite": v["finite"],
                      "lambda_frac_saturated": v["lambda_frac_saturated"],
                      "task_learning_gain_reported_not_gated": v["task_learning_gain"]}
            if "kl_median" in v:
                ok = ok and v["kl_median"] <= KL_MEDIAN_MAX and v["kl_p95"] <= KL_P95_MAX
                detail.update(kl_median=v["kl_median"], kl_p95=v["kl_p95"])
            if "entropy_first10" in v:
                ok = ok and v["entropy_last10"] < v["entropy_first10"] \
                    and v["entropy_last10"] > 0
                detail.update(entropy_first10=v["entropy_first10"],
                              entropy_last10=v["entropy_last10"])
            detail["pass"] = ok
            health[f"{dose_label(d)}|seed{s}"] = detail
            g6_pass &= ok
    g["DR_G6"] = {"pass": bool(g6_pass), "per_run": health}

    # ---- DR-G7: task performance and stealth, per dose -----------------
    g["DR_G7"] = {
        "delta_equiv": DELTA_EQUIV,
        "tost_primary": {dose_label(d):
                         tost(summaries, "task_return_last50", d, 0.0, seeds, DELTA_EQUIV)
                         for d in nonzero},
        "tost_paper_criterion": {
            dose_label(d): tost(summaries, "task_return_last50", d, 0.0, seeds,
                                DELTA_EQUIV_PAPER) for d in nonzero},
        "task_return_last50_difference": {
            dose_label(d): paired(summaries, "task_return_last50", d, 0.0, seeds)
            for d in nonzero},
        "task_learning_gain_by_dose": {
            dose_label(d): {str(s): summaries[(d, s)]["task_learning_gain"] for s in seeds}
            for d in ordered},
        "note": "p>0.05 on a difference test is NEVER reported as stealth "
                "(gates doc 8, DR-G7)",
    }
    return g


def verdict(gates: dict, structural_ok: bool | None, g1_iii_ci_contains_zero: bool) -> str:
    """gates doc 10.2, applied literally."""
    if structural_ok is False:
        return "FAIL"
    if gates["DR_G4"]["pass"] is False:
        return "FAIL"
    if (gates["DR_G1"]["pass"] and gates["DR_G2"]["pass"]
            and gates["DR_G4"]["pass"] and gates["DR_G5"]["pass"]
            and structural_ok):
        return "PASS"
    if not gates["DR_G1"]["G1_ii_monotone_ordering_all_seeds"]["pass"] \
            and g1_iii_ci_contains_zero:
        return "INCONCLUSIVE"
    return "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/runs_dose/dose_report.json")
    args = ap.parse_args()

    runs, summaries, missing = {}, {}, []
    for (dose, seed), rel in sorted(RUN_DIRS.items()):
        r = load_run(rel)
        if r is None:
            missing.append({"dose": dose, "seed": seed, "dir": rel})
            continue
        runs[(dose, seed)] = r
        summaries[(dose, seed)] = summarize(r)

    doses_available = sorted({d for (d, _s) in summaries})
    complete = not missing

    report: dict = {
        "predeclaration": "docs/dose_response_gates.md",
        "design": {
            "independent_variable": "attack.budget_ratio (B/d)",
            "doses": list(DOSES),
            "seeds": list(SEEDS),
            "budget_d": D_BUDGET, "M": M_SOURCES,
            "corrupted_source_by_seed": {str(k): v for k, v in ATTACKED_SOURCE.items()},
            "reference_arms_committed": {"B/d=0": "G9/G10 clean", "B/d=0.5": "A1 condition B"},
            "primary_outcome": "whole-run network-average true cost",
            "primary_contrast": "Delta J_true(dose) = J_true(dose) - J_true(clean), paired by seed",
        },
        "missing_runs": missing,
        "complete": complete,
        "doses_available": doses_available,
        "per_run": {f"{dose_label(d)}|seed{s}": summaries[(d, s)]
                    for (d, s) in sorted(summaries)},
    }

    # Treatment integrity, read from each run's own snapshot.
    integrity = {}
    for (d, s), v in sorted(summaries.items()):
        integrity[f"{dose_label(d)}|seed{s}"] = {
            "config_budget_ratio": v.get("config_budget_ratio"),
            "expected_budget_ratio": d,
            "match": v.get("config_budget_ratio") == d,
            "defense": v.get("config_defense"),
            "defense_is_mean": v.get("config_defense") == "mean",
            "corrupted_source_ids": v["corrupted_source_ids"],
            "expected_corrupted": ([ATTACKED_SOURCE[s]] if d > 0 else []),
            "git_sha": v.get("git_sha"),
            "cpu": v.get("provenance_cpu"), "arch": v.get("provenance_arch"),
        }
    report["treatment_integrity"] = integrity
    structural_ok = all(e["match"] and e["defense_is_mean"]
                        and e["corrupted_source_ids"] == e["expected_corrupted"]
                        for e in integrity.values())
    report["treatment_integrity_pass"] = structural_ok

    if complete:
        gates = evaluate_gates(summaries, SEEDS, DOSES)
        report["gates"] = gates
        ci = gates["DR_G1"]["G1_iii_dose_range_effect"].get("ci95")
        ci_contains_zero = bool(ci and ci[0] <= 0 <= ci[1])
        report["verdict"] = verdict(gates, structural_ok, ci_contains_zero)
        report["verdict_inputs"] = {
            "treatment_integrity": structural_ok,
            "DR_G1": gates["DR_G1"]["pass"],
            "DR_G2": gates["DR_G2"]["pass"],
            "DR_G4": gates["DR_G4"]["pass"],
            "DR_G5": gates["DR_G5"]["pass"],
            "DR_G6": gates["DR_G6"]["pass"],
            "G1_iii_ci95_contains_zero": ci_contains_zero,
            "note": "per-run structural gates DR-S0..S12 live in each run's "
                    "dose_run_verification.json and are not re-evaluated here",
        }
    else:
        report["verdict"] = "INCOMPLETE"
        report["note"] = (f"{len(missing)} of {len(RUN_DIRS)} cells missing; "
                          "gates are not evaluated on a partial grid")

    report["statistical_policy"] = {
        "min_seeds_for_inference": MIN_SEEDS_FOR_INFERENCE,
        "n": len(SEEDS),
        "primary_evidence": "per-seed sign and ordering consistency",
        "paired_t": "sensitivity analysis only, never a significance claim",
        "sign_test_min_attainable_p_at_n3": 0.25,
        "multiplicity": "not applied at n=3; no inferential family exists "
                        "(gates doc 9.1)",
        "functional_form": "NOT estimated -- no regression, slope, intercept or R^2 "
                           "(gates doc 9.2)",
    }

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"=== DR dose sweep: {len(summaries)}/{len(RUN_DIRS)} cells present ===")
    if missing:
        for m in missing:
            print(f"  MISSING  B/d={m['dose']:g} seed{m['seed']}  {m['dir']}")
    print(f"\n{'dose':10s} {'seed':5s} {'J_true':>9s} {'J_last50':>9s} "
          f"{'mech_rep':>9s} {'mech_gap':>9s} {'lam_mean':>9s} {'task_l50':>9s}")
    for (d, s) in sorted(summaries):
        v = summaries[(d, s)]
        print(f"{dose_label(d):10s} {s:<5d} {v['true_cost_net_whole']:9.4f} "
              f"{v['true_cost_net_last50']:9.4f} {v['mech_reported_whole']:9.4f} "
              f"{v['mech_gap_whole']:9.4f} {v['lambda_mean']:9.4f} "
              f"{v['task_return_last50']:9.4f}")
    print(f"\nverdict: {report['verdict']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
