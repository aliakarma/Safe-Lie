#!/usr/bin/env python
"""A3 analyser -- the 2x2 attack x RCE factorial at M=5, where the margin is live.

Implements docs/a3_gates.md exactly. Every bar is read from that document;
none is chosen here. Structurally identical to scripts/analyze_a2.py, because
A3 must be comparable to A2 on the same decision rule -- what changes is M,
the run directories, and the mechanism gates, which A3 inverts: A2 had to
prove the margin was dead, A3 has to prove it is alive.

All twelve runs are produced on ONE machine (BRANCH-B', gates doc section
12.4), so every contrast is a within-machine difference and no machine factor
enters.

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
    python scripts/analyze_a3.py --out results/runs_a3/a3_report.json
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
M_SOURCES = 5
BETA = 1.5
SIGMA_MIN = 1e-3
# A3 collects its own undefended references, so there is no external value
# to check the sign convention against. What IS checkable is the direction:
# the attack must RAISE true cost, as it did in A1 and A2.
A2_B_MINUS_A_FOR_REFERENCE = 3.9858   # A2's M=3 value, reported for contrast
G5_REDUCTION_BAR = 0.5               # PASS needs (C-E) <= 0.5 * (B-A)
G7_I_MIN_FRAC = 0.99
G7_II_BAND = (-0.70, -0.02)
PRED_RCE_SHIFT = -0.222
G3_II_MIN_FRAC = 0.99
G3_III_BAND_CLEAN = (0.25, 0.60)
G3_III_BAND_ATTACK = (0.35, 0.80)
KL_MEDIAN_MAX = 0.0069               # G10-C-iii, verbatim
KL_P95_MAX = 0.01548
LAMBDA_SAT_MAX = 0.05
MIN_SEEDS_FOR_INFERENCE = 5          # decision D6
TOL = 1e-9

SEEDS = (0, 1, 2)
ATTACKED_SOURCE = {0: "batch_1", 1: "batch_3", 2: "batch_5"}

# A3 reuses NOTHING: changing M changes the source architecture, so it
# collects its own undefended references too.
RUN_DIRS = {(c, s): f"results/runs_a3/{c}_seed{s}"
            for c in ("A", "B", "C", "E") for s in SEEDS}
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
    mech = d / "a3_mechanism_report.json"
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
    """The margin diagnostics, aggregated over seeds.

    A2's version of this function existed to document a DEAD margin. A3's
    exists to characterise a LIVE one: at M=5, f=1 the retained set holds 3
    values, min_retained is satisfied, nothing is floored, and beta*MAD is a
    real data-dependent quantity that should respond to corruption.
    """
    out = {"per_run": {}, "note": (
        "At M=5, f=1 the trimmed set holds |T| = M - 2f = 3 values, so "
        "min_retained=3 is satisfied, the spread is a real MAD and is NOT "
        "floored, and the margin beta*MAD is data-dependent. Contrast A2 at "
        "M=3, where |T|=1 forced MAD to 0 and the margin to the constant "
        "%.4g (docs/a3_gates.md sections 2 and 12)." % (BETA * SIGMA_MIN))}
    for cond in RCE_CONDITIONS:
        for s in seeds:
            run = runs.get((cond, s))
            if run is None:
                continue
            sp, mg = run["spread"], run["applied_margin"]
            entry = {
                "retained_n_unique": sorted({int(x) for x in np.unique(run["retained_n"])}),
                "frac_retained_n_eq_3": float((run["retained_n"] == 3).mean()),
                "spread_min": float(sp.min()), "spread_max": float(sp.max()),
                "spread_mean": float(sp.mean()), "spread_median": float(np.median(sp)),
                "frac_spread_above_sigma_min": float((sp > SIGMA_MIN + TOL).mean()),
                "frac_spread_at_floor": float((np.abs(sp - SIGMA_MIN) <= TOL).mean()),
                "applied_margin_mean": float(mg.mean()),
                "applied_margin_median": float(np.median(mg)),
                "applied_margin_sd": float(mg.std(ddof=1)),
                "margin_ratio_vs_a2_constant": float(mg.mean() / (BETA * SIGMA_MIN)),
                "mad_is_live": bool((sp > SIGMA_MIN + TOL).mean() >= G3_II_MIN_FRAC),
                "epsilon_offline": (run.get("calibration") or {}).get("epsilon_offline"),
                "guarantee_in_force_frac_true": float(np.mean(
                    [[bool(v) for v in row] for row in run["guarantee_in_force"]])),
            }
            mr = run.get("mechanism_report") or {}
            for key in ("G3_i_retained_structure", "G3_ii_mad_is_a_measurement",
                        "G3_iii_margin_magnitude", "G1_v_calibration_not_vacuous",
                        "G7_i_source_set", "G7_ii_within_run_counterfactual",
                        "G1_ii_crn_round0", "G2_map_identity"):
                if key in mr:
                    entry[key] = mr[key]
            out["per_run"][f"{cond}_seed{s}"] = entry

    # A3-G3-iv: does the margin RESPOND to corruption? Per seed, C' vs E'.
    responds = {}
    for s in seeds:
        c = out["per_run"].get(f"C_seed{s}")
        e = out["per_run"].get(f"E_seed{s}")
        if c and e:
            responds[str(s)] = {
                "margin_C": c["applied_margin_mean"], "margin_E": e["applied_margin_mean"],
                "ratio": float(c["applied_margin_mean"] / e["applied_margin_mean"])
                if e["applied_margin_mean"] else None,
                "responds": bool(c["applied_margin_mean"] > e["applied_margin_mean"]),
            }
    if responds:
        out["G3_iv_margin_responds_to_corruption"] = {
            "per_seed": responds,
            "predicted_ratio": 1.34,
            "all_seeds_respond": bool(all(v["responds"] for v in responds.values())),
            "mean_ratio": float(np.mean([v["ratio"] for v in responds.values()
                                         if v["ratio"] is not None])),
        }

    trimmed = [e["G7_i_source_set"].get("frac_attacked_source_trimmed")
               for k, e in out["per_run"].items()
               if k.startswith("C") and "G7_i_source_set" in e]
    shifts = [e["G7_ii_within_run_counterfactual"].get("mean_rce_shift")
              for k, e in out["per_run"].items()
              if k.startswith("C") and "G7_ii_within_run_counterfactual" in e]
    if trimmed and all(v is not None for v in trimmed):
        out["pooled_frac_attacked_source_trimmed"] = float(np.mean(trimmed))
    if shifts and all(v is not None for v in shifts):
        out["pooled_mean_rce_aggregate_shift"] = float(np.mean(shifts))
        out["predicted_rce_aggregate_shift"] = PRED_RCE_SHIFT
        out["mean_aggregator_shift_for_comparison"] = -B_MAGNITUDE / M_SOURCES
        out["estimator_corruption_suppression_factor"] = float(
            (B_MAGNITUDE / M_SOURCES) / abs(np.mean(shifts)))
        out["a2_suppression_at_M3_for_comparison"] = 6.0
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
    g["A3_G1_G2_structural"] = {
        "per_run": structural,
        "detail": "see each run's a2_mechanism_report.json",
        "pass": bool(structural and all(v is True for v in structural.values())),
    }

    # --- A2-G3: clean RCE effect measured and reported (validity only)
    if have("E"):
        g["A3_G4_clean_rce_effect"] = {
            "E-A": contrasts["E-A"]["true_cost_net_whole"],
            "runs_complete": all(summaries[("E", s)]["rounds"] == 250 for s in seeds),
            "finite": all(summaries[("E", s)]["finite"] for s in seeds),
            "note": "value is a measurement, not a hypothesis; gated only for run validity",
            "pass": bool(all(summaries[("E", s)]["rounds"] == 250 for s in seeds)
                         and all(summaries[("E", s)]["finite"] for s in seeds)),
        }

    # --- A2-G4: residual attack effect reported
    if have("C") and have("E"):
        g["A3_residual_attack_effect"] = {
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
        g["A3_G5_interaction"] = {
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
    g["A3_G6_learning_health"] = {
        "per_run": health,
        "note": "G10-C-ii (task gain >= +75) reported, not gated (gates doc A3-G6)",
        "pass": bool(health and all(v["pass"] for v in health.values())),
    }

    # --- A3-G3 / A3-G7: the margin is LIVE and behaves as derived.
    # This inverts A2's mechanism gate. A2 required the margin to be a
    # constant; A3 requires it to be a real, responsive measurement. If
    # G3-i or G3-ii fails, A3 has NOT tested the margin and no conclusion
    # about Theorem 2's condition may be drawn from it.
    per_run_g7 = {}
    for key, entry in mech.get("per_run", {}).items():
        is_attacked = key.startswith("C")
        band = G3_III_BAND_ATTACK if is_attacked else G3_III_BAND_CLEAN
        sub = {
            "retained_n_always_3": entry["retained_n_unique"] == [3],
            "mad_is_live": entry["frac_spread_above_sigma_min"] >= G3_II_MIN_FRAC,
            "margin_in_band": bool(band[0] <= entry["applied_margin_mean"] <= band[1]),
            "calibration_not_vacuous": bool(
                (entry.get("epsilon_offline") or 0.0) > SIGMA_MIN + TOL),
        }
        if is_attacked:
            g1 = entry.get("G7_i_source_set", {})
            g2 = entry.get("G7_ii_within_run_counterfactual", {})
            sub["attacked_source_trimmed"] = (
                g1.get("frac_attacked_source_trimmed", 0.0) >= G7_I_MIN_FRAC)
            ms = g2.get("mean_rce_shift")
            sub["shift_in_band"] = bool(ms is not None and G7_II_BAND[0] <= ms <= G7_II_BAND[1])
        sub["pass"] = all(v for k, v in sub.items() if k != "pass")
        per_run_g7[key] = sub
    resp = mech.get("G3_iv_margin_responds_to_corruption")
    g["A3_G3_G7_rce_mechanism"] = {
        "per_run": per_run_g7,
        "G3_iv_margin_responds_to_corruption": resp,
        "pass": bool(per_run_g7 and all(v["pass"] for v in per_run_g7.values())
                     and (resp is None or resp["all_seeds_respond"])),
    }
    return g


# ------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/runs_a3/a3_report.json")
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
            "primary_quantity": "(C'-E')-(B'-A') on true_cost_net_whole",
            "metric_direction": "higher true cost is WORSE",
            "seeds": list(SEEDS), "complete_seeds": seeds,
            "budget_d": D_BUDGET, "B": B_MAGNITUDE, "M": M_SOURCES,
            "defense": {"name": "rce", "f": 1, "beta": BETA, "sigma_min": SIGMA_MIN},
            "attacked_source_by_seed": ATTACKED_SOURCE,
            "gates_doc": "docs/a3_gates.md",
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
                            ("A3_G1_G2_structural", "A3_G6_learning_health",
                             "A3_G3_G7_rce_mechanism") if k in gates)
        g5 = gates.get("A3_G5_interaction", {})
        if len(seeds) < len(SEEDS):
            # The A2-G5 rule is stated over ALL THREE seeds ("I<0 in all 3
            # seeds"). Evaluated on a subset it passes vacuously -- one
            # negative seed satisfies "all of them" when the set has one
            # element. Emitting a final verdict from partial data would put a
            # PASS into an artifact that later reads as the A2 result, which
            # is exactly the goalpost drift the pre-declaration exists to
            # prevent. Report the provisional numbers, withhold the verdict.
            report["verdict"] = f"PROVISIONAL -- {len(seeds)} of {len(SEEDS)} seeds complete"
            report["verdict_note"] = (
                "Not an A2 verdict. Contrasts below are per-seed differences with no "
                "cross-seed CI; the A2-G5 decision rule requires all three seeds and is "
                "not evaluated here.")
        elif not structural_ok:
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
    print(f"A3 -- attack x RCE factorial at M=5   (complete seeds: {seeds or 'none'})")
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
