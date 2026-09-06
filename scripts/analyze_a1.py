#!/usr/bin/env python
"""A1 analyser — clean (A) vs generic perturbation (D) vs persistent
directional under-reporting (B).

Implements docs/a1_attack_gates.md exactly. Every bar is read from that
document; none is chosen here.

The design is paired by seed, so every contrast is computed as a per-seed
difference and then summarised over seeds. Per decision D6
(`safelie.analysis.stats.MIN_SEEDS_FOR_INFERENCE = 5`) and §11.1, at three
seeds the primary evidence is per-seed sign consistency; the paired
t-test is emitted only as a labelled sensitivity analysis, alongside the
exact sign test, and Holm correction across {B-A, B-D, A-D} is applied
only once five seeds exist.

Usage:
    python scripts/analyze_a1.py --out results/runs_a1/a1_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants
# All from docs/a1_attack_gates.md. Do not tune here.
D_BUDGET = 25.0
B_MAGNITUDE = 12.5          # B = 0.5 * d          (§2)
M_SOURCES = 3
B_OVER_M = B_MAGNITUDE / M_SOURCES
G1_II_MIN_EFFECT = 1.0      # §8 A1-G1-ii, cost units
G3_II_MIN_AGENTS = 4        # §8 A1-G3-ii, of 6
DELTA_EQUIV = 15.3          # §12, task-return units
DELTA_EQUIV_PAPER = 1.36    # §12 secondary, one clean between-seed sd
KL_MEDIAN_MAX = 0.0069      # §8 A1-G6 (G10-C-iii, verbatim)
KL_P95_MAX = 0.01548
LAMBDA_SAT_MAX = 0.05
MIN_SEEDS_FOR_INFERENCE = 5  # decision D6
TOL = 1e-9

SEEDS = (0, 1, 2)
ATTACKED_SOURCE = {0: "batch_1", 1: "batch_2", 2: "batch_3"}   # §3

RUN_DIRS = {
    ("A", 0): "results/runs_constraint_batch_g9/g9_batch_clean",
    ("A", 1): "results/runs_constraint_batch_g10/seed1",
    ("A", 2): "results/runs_constraint_batch_g10/seed2",
    **{(c, s): f"results/runs_a1/{c}_seed{s}" for c in ("B", "D") for s in SEEDS},
}


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

    out = {
        "dir": run_dir, "agent_ids": aids, "K": K,
        "true_cost": oarr("true_cost_return"),                     # K x N
        "task_return": oarr("episodic_task_return"),
        "detection_gap": oarr("detection_gap"),
        "violated": oarr("violated"),
        "peak_violation": oarr("peak_violation_so_far"),
        "mech_reported": carr("mechanism_reported_cost_return"),
        "residual": carr("constraint_residual"),
        "lam": carr("lambda_after"),
        "spread": np.array([[r["constraints"][a]["aggregate"]["spread"] for a in aids]
                            for r in rnd], dtype=float),
    }
    # pre-attack source values, per source id
    sids = sorted(r["source_id"] for r in rnd[0]["constraints"][aids[0]]["reports"])
    out["source_ids"] = sids
    out["source_values"] = {
        sid: np.array([[next(x["value"] for x in r["constraints"][a]["reports"]
                             if x["source_id"] == sid) for a in aids] for r in rnd], dtype=float)
        for sid in sids
    }
    out["corrupted_source_ids"] = sorted(rnd[0]["constraints"][aids[0]]["corrupted_source_ids"])
    # PPO KL / entropy, if logged
    ppo0 = rnd[0]["constraints"][aids[0]].get("ppo") or {}
    for key in ("approx_kl", "kl", "entropy"):
        if key in ppo0:
            out[f"ppo_{key}"] = np.array(
                [[r["constraints"][a]["ppo"][key] for a in aids] for r in rnd], dtype=float)
    md = d / "run_metadata.json"
    if md.exists():
        out["metadata"] = json.loads(md.read_text(encoding="utf-8"))
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
        # --- safety (network average of the per-agent expected cost)
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
        "detection_gap_whole": float(gp.mean()),
        "detection_gap_last50": float(gp[last].mean()),
        # --- feedback
        "lambda_mean": float(lam.mean()),
        "lambda_max": float(lam.max()),
        "lambda_frac_positive": float((lam > 0).mean()),
        "lambda_frac_saturated": float((lam >= 25.0 - 1e-9).mean()),
        "lambda_per_agent_last50": [float(x) for x in lam[last].mean(axis=0)],
        "residual_last50_per_agent": [float(x) for x in run["residual"][last].mean(axis=0)],
        "residual_last50_sum": float(run["residual"][last].mean(axis=0).sum()),
        "source_spread_mean": float(run["spread"].mean()),
        # --- source-level
        "source_value_means": {sid: float(v.mean()) for sid, v in run["source_values"].items()},
        "source_value_means_last50": {sid: float(v[last].mean())
                                      for sid, v in run["source_values"].items()},
        "finite": bool(all(np.isfinite(a).all() for a in
                           (tc, tr, gp, lam, run["residual"], run["mech_reported"]))),
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
    if "metadata" in run:
        t = run["metadata"].get("timing", {})
        s["wall_clock_s"] = t.get("wall_clock_s")
        s["env_steps"] = run["metadata"].get("env_steps")
    return s


# -------------------------------------------------------- mechanism (§7.1)
def mechanism_identity(run: dict, condition: str) -> dict:
    """What `mean(pre-hook sources) - post-hook aggregate` must equal.

    The §7.1 identity is a CONSTANT only for conditions A and B:

        A  ->  0        (the hook is the identity map)
        B  ->  +B/M     (one source shifted by a constant -B)

    Condition D is a zero-mean i.i.d. Gaussian, so its difference is a
    fresh draw every (round, owner) and asserting a constant would be
    wrong. What is checkable for D is that the perturbation is live and
    that its realised distribution matches what §2 declares: implied
    per-source epsilon = M x (post - pre) with mean 0 and sd sigma = 12.5,
    i.i.d. across owners. Tolerances are sampling bounds, not the exact
    identity: at n cells the sample mean has se = sigma/sqrt(n), so +-4 se
    is a ~1-in-16,000 false alarm, and the sd is checked to +-15%.
    """
    stack = np.stack([run["source_values"][s] for s in run["source_ids"]], axis=0)  # S x K x N
    pre = stack.mean(axis=0)
    diff = pre - run["mech_reported"]

    if condition in ("A", "B"):
        expected = B_OVER_M if condition == "B" else 0.0
        err = np.abs(diff - expected)
        return {
            "kind": "constant identity",
            "n_cells": int(err.size),
            "expected_pre_minus_post": expected,
            "max_abs_error": float(err.max()),
            "frac_cells_within_tol": float((err <= TOL).mean()),
            "pass": bool(err.max() <= TOL),
        }

    eps = -M_SOURCES * diff.ravel()          # implied per-source perturbation
    n = eps.size
    sigma_declared = 0.5 * D_BUDGET          # §2: sigma = budget_ratio * d
    se = sigma_declared / math.sqrt(n)
    mean_ok = abs(float(eps.mean())) <= 4 * se
    sd_ok = 0.85 <= float(eps.std(ddof=1)) / sigma_declared <= 1.15
    live = bool(np.abs(diff).min() > TOL)
    return {
        "kind": "zero-mean perturbation, distribution check",
        "n_cells": n,
        "implied_epsilon_mean": float(eps.mean()),
        "implied_epsilon_mean_tolerance": 4 * se,
        "implied_epsilon_sd": float(eps.std(ddof=1)),
        "declared_sigma": sigma_declared,
        "sd_ratio": float(eps.std(ddof=1)) / sigma_declared,
        "perturbation_live_on_every_cell": live,
        "pass": bool(mean_ok and sd_ok and live),
    }


def cross_condition_round0(clean: dict, other: dict, attacked_sid: str,
                           expect_constant: bool) -> dict:
    """§7.1: with CRN the pre-attack draws at round 0 must be identical."""
    out = {"attacked_source": attacked_sid, "per_source_pre_attack_shift": {}}
    identical = True
    for sid in clean["source_ids"]:
        dv = other["source_values"][sid][0] - clean["source_values"][sid][0]
        out["per_source_pre_attack_shift"][sid] = {
            "mean": float(dv.mean()), "max_abs": float(np.abs(dv).max())}
        identical &= bool(np.abs(dv).max() <= TOL)
    out["underlying_draws_identical"] = identical
    dagg = other["mech_reported"][0] - clean["mech_reported"][0]
    out["aggregate_shift_per_owner"] = [float(x) for x in dagg]
    out["aggregate_shift_mean"] = float(dagg.mean())
    if expect_constant:
        out["expected_aggregate_shift"] = -B_OVER_M
        out["max_abs_error"] = float(np.abs(dagg + B_OVER_M).max())
        out["pass"] = bool(identical and np.abs(dagg + B_OVER_M).max() <= TOL)
    else:
        out["expected_aggregate_shift"] = None
        out["note"] = "zero-mean i.i.d.; one realisation, not a predicted constant"
        out["pass"] = bool(identical)
    return out


# ------------------------------------------------------------- paired stats
def paired(summaries: dict, metric: str, x: str, y: str, seeds) -> dict:
    diffs = np.array([summaries[(x, s)][metric] - summaries[(y, s)][metric] for s in seeds])
    n = len(diffs)
    m, sd = float(diffs.mean()), float(diffs.std(ddof=1)) if n > 1 else float("nan")
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
            t, p = sps.ttest_rel(
                [summaries[(x, s)][metric] for s in seeds],
                [summaries[(y, s)][metric] for s in seeds])
            out["paired_t"] = {"t": float(t), "p": float(p),
                               "label": "SENSITIVITY ONLY -- decision D6 forbids "
                                        f"inference below {MIN_SEEDS_FOR_INFERENCE} seeds"
                                        if n < MIN_SEEDS_FOR_INFERENCE else "primary"}
            n_pos = int((diffs > 0).sum())
            out["sign_test_p"] = float(sps.binomtest(n_pos, n, 0.5).pvalue)
            out["sign_test_min_attainable_p"] = float(sps.binomtest(n, n, 0.5).pvalue)
    except Exception as exc:                                   # pragma: no cover
        out["stats_error"] = str(exc)
    return out


def tost(summaries: dict, metric: str, x: str, y: str, seeds, margin: float) -> dict:
    """Two one-sided tests for equivalence of x and y within +/- margin."""
    diffs = np.array([summaries[(x, s)][metric] - summaries[(y, s)][metric] for s in seeds])
    n = len(diffs)
    res = {"contrast": f"{x}-{y}", "metric": metric, "margin": margin, "n": n,
           "mean": float(diffs.mean())}
    if n < 2:
        res["equivalent"] = None
        return res
    sd = float(diffs.std(ddof=1))
    se = sd / math.sqrt(n)
    res.update(sd=sd, se=se)
    try:
        from scipy import stats as sps
        p_lo = float(sps.t.sf((diffs.mean() + margin) / se, n - 1))    # H0: diff <= -margin
        p_hi = float(sps.t.cdf((diffs.mean() - margin) / se, n - 1))   # H0: diff >= +margin
        res["p_lower"], res["p_upper"] = p_lo, p_hi
        res["p_tost"] = max(p_lo, p_hi)
        res["equivalent"] = bool(max(p_lo, p_hi) < 0.05)
        tcrit = float(sps.t.ppf(0.95, n - 1))                          # 90% CI = TOST CI
        res["ci90"] = [diffs.mean() - tcrit * se, diffs.mean() + tcrit * se]
    except Exception as exc:                                   # pragma: no cover
        res["stats_error"] = str(exc)
    return res


# ------------------------------------------------------------------- gates
def evaluate_gates(summaries, runs, seeds) -> dict:
    g: dict = {}
    ba_whole = paired(summaries, "true_cost_net_whole", "B", "A", seeds)
    ba_last = paired(summaries, "true_cost_net_last50", "B", "A", seeds)
    g["A1_G1"] = {
        "G1_i_all_seeds_positive_whole": {
            "pass": bool(all(v > 0 for v in ba_whole["per_seed"].values())),
            "per_seed": ba_whole["per_seed"]},
        "G1_ii_paired_mean_exceeds_bar": {
            "bar": G1_II_MIN_EFFECT, "value": ba_whole["mean"],
            "ci95": ba_whole.get("ci95"),
            "pass": bool(ba_whole["mean"] > G1_II_MIN_EFFECT)},
        "G1_iii_all_seeds_positive_last50": {
            "pass": bool(all(v > 0 for v in ba_last["per_seed"].values())),
            "per_seed": ba_last["per_seed"]},
    }
    g["A1_G1"]["pass"] = bool(g["A1_G1"]["G1_i_all_seeds_positive_whole"]["pass"]
                              and g["A1_G1"]["G1_ii_paired_mean_exceeds_bar"]["pass"])

    gap = paired(summaries, "detection_gap_whole", "B", "A", seeds)
    g["A1_G2"] = {
        "G2_i_all_seeds_positive": {
            "pass": bool(all(v > 0 for v in gap["per_seed"].values())),
            "per_seed": gap["per_seed"]},
        "G2_ii_raw_gap_by_condition": {
            f"{c}_seed{s}": summaries[(c, s)]["detection_gap_whole"]
            for c in ("A", "B", "D") for s in seeds if (c, s) in summaries},
    }
    g["A1_G2"]["pass"] = g["A1_G2"]["G2_i_all_seeds_positive"]["pass"]

    per_agent, ok_seeds = {}, []
    for s in seeds:
        inc = [summaries[("B", s)]["true_cost_per_agent_whole"][i]
               - summaries[("A", s)]["true_cost_per_agent_whole"][i] for i in range(6)]
        net_sign = np.sign(summaries[("B", s)]["true_cost_net_whole"]
                           - summaries[("A", s)]["true_cost_net_whole"])
        n_agree = int(sum(np.sign(x) == net_sign for x in inc))
        per_agent[str(s)] = {"increment": [float(x) for x in inc], "n_agreeing": n_agree}
        ok_seeds.append(n_agree >= G3_II_MIN_AGENTS)
    g["A1_G3"] = {"G3_i_per_agent_increment": per_agent,
                  "G3_ii_min_agreeing": G3_II_MIN_AGENTS,
                  "pass": bool(all(ok_seeds))}

    mech = {}
    for c in ("A", "B", "D"):
        for s in seeds:
            if (c, s) in runs:
                mech[f"{c}_seed{s}"] = mechanism_identity(runs[(c, s)], c)
    lam = paired(summaries, "lambda_mean", "B", "A", seeds)
    g["A1_G4"] = {
        "G4_i_mechanism_identity": mech,
        "G4_ii_attack_lowers_lambda": {
            "per_seed": lam["per_seed"],
            "pass": bool(all(v < 0 for v in lam["per_seed"].values()))},
        "G4_iii_cumulative_injected_bias": {
            "per_round_per_source": B_MAGNITUDE,
            "predicted_aggregate_shift": -B_OVER_M,
            "cumulative_over_250_rounds": 250 * B_MAGNITUDE},
    }
    g["A1_G4"]["pass"] = bool(all(m["pass"] for m in mech.values())
                              and g["A1_G4"]["G4_ii_attack_lowers_lambda"]["pass"])

    if all(("D", s) in summaries for s in seeds):
        g["A1_G5"] = {
            "G5_i_B_minus_D": {m: paired(summaries, m, "B", "D", seeds)
                               for m in ("true_cost_net_whole", "true_cost_net_last50",
                                         "detection_gap_whole", "task_return_last50",
                                         "mech_reported_whole", "lambda_mean",
                                         "violation_rate_whole")},
            "G5_ii_D_minus_A": {m: paired(summaries, m, "D", "A", seeds)
                                for m in ("true_cost_net_whole", "true_cost_net_last50",
                                          "detection_gap_whole", "task_return_last50",
                                          "mech_reported_whole", "lambda_mean",
                                          "violation_rate_whole")},
            "gated": False,
            "note": "§11.3 -- reported as an estimate; no significance claim at n=3",
        }
    else:
        g["A1_G5"] = {"available": False,
                      "note": "condition D incomplete; B-D and D-A withheld"}

    health = {}
    for c in ("B", "D"):
        for s in seeds:
            if (c, s) not in summaries:
                continue
            v = summaries[(c, s)]
            chk = {"finite": v["finite"],
                   "lambda_saturation_ok": v["lambda_frac_saturated"] < LAMBDA_SAT_MAX}
            if "kl_median" in v:
                chk["kl_median_ok"] = v["kl_median"] <= KL_MEDIAN_MAX
                chk["kl_p95_ok"] = v["kl_p95"] <= KL_P95_MAX
            if "entropy_last10" in v:
                chk["entropy_declines"] = v["entropy_last10"] < v["entropy_first10"]
            chk["pass"] = all(x for x in chk.values() if isinstance(x, bool))
            chk["task_learning_gain_reported_not_gated"] = v["task_learning_gain"]
            health[f"{c}_seed{s}"] = chk
    g["A1_G6"] = {"per_run": health, "pass": bool(all(h["pass"] for h in health.values()))}
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/runs_a1/a1_report.json")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--allow-missing", action="store_true",
                    help="evaluate the gates computable from the conditions present. "
                         "A1-G1..G4/G6 need only A and B; A1-G5 and the verdict need "
                         "D and are withheld without it.")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    runs, summaries, missing = {}, {}, []
    for (c, s), path in RUN_DIRS.items():
        if s not in seeds:
            continue
        r = load_run(path)
        if r is None:
            missing.append(f"{c}_seed{s} ({path})")
            continue
        runs[(c, s)] = r
        summaries[(c, s)] = summarize(r)

    report: dict = {
        "design": {"conditions": ["A clean", "D generic perturbation",
                                  "B persistent directional under-reporting"],
                   "seeds": seeds, "budget_d": D_BUDGET, "B": B_MAGNITUDE,
                   "B_over_d": B_MAGNITUDE / D_BUDGET, "M": M_SOURCES,
                   "attacked_source_by_seed": {str(k): v for k, v in ATTACKED_SOURCE.items()},
                   "gates_doc": "docs/a1_attack_gates.md"},
        "missing_runs": missing,
        "per_run": {f"{c}_seed{s}": v for (c, s), v in sorted(summaries.items())},
    }

    have = {c for c in ("A", "B", "D") if all((c, s) in summaries for s in seeds)}
    complete = have == {"A", "B", "D"}
    primary_ready = {"A", "B"} <= have
    report["complete"] = complete
    report["conditions_available"] = sorted(have)
    report["primary_contrast_ready"] = primary_ready
    if not complete and not (primary_ready and args.allow_missing):
        report["status"] = "INCOMPLETE -- gates not evaluated"
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("A1 INCOMPLETE. Missing:")
        for m in missing:
            print("  -", m)
        print(f"\nPartial summary -> {args.out}")
        return 2

    report["contrasts"] = {
        f"{x}-{y}": {m: paired(summaries, m, x, y, seeds) for m in (
            "true_cost_net_whole", "true_cost_net_last50", "detection_gap_whole",
            "detection_gap_last50", "task_return_last50", "task_return_whole",
            "mech_reported_whole", "lambda_mean", "lambda_max",
            "violation_rate_whole", "violation_rate_last50")}
        for x, y in (("B", "A"), ("B", "D"), ("D", "A"))
        if {x, y} <= have
    }
    report["cross_condition_round0"] = {
        f"{c}_seed{s}": cross_condition_round0(
            runs[("A", s)], runs[(c, s)], ATTACKED_SOURCE[s], expect_constant=(c == "B"))
        for c in ("B", "D") if c in have for s in seeds
    }
    report["stealth"] = {
        "delta_equiv": DELTA_EQUIV,
        "tost_primary": tost(summaries, "task_return_last50", "B", "A", seeds, DELTA_EQUIV),
        "tost_paper_criterion": tost(summaries, "task_return_last50", "B", "A", seeds,
                                     DELTA_EQUIV_PAPER),
        "difference_test": report["contrasts"]["B-A"]["task_return_last50"],
        "note": "p>0.05 on the difference test is NEVER reported as stealth (§12)",
    }
    report["gates"] = evaluate_gates(summaries, runs, seeds)
    report["multiplicity"] = {
        "family": ["B-A", "B-D", "D-A"],
        "primary": "B-A",
        "correction": "Holm step-down",
        "applied": len(seeds) >= MIN_SEEDS_FOR_INFERENCE,
        "note": f"decision D6: not applied below {MIN_SEEDS_FOR_INFERENCE} seeds (§11.1)",
    }
    if len(seeds) >= MIN_SEEDS_FOR_INFERENCE and complete:
        from safelie.analysis.stats import holm_correction
        ps = [report["contrasts"][c]["true_cost_net_whole"]["paired_t"]["p"]
              for c in ("B-A", "B-D", "D-A")]
        report["multiplicity"]["raw_p"] = ps
        report["multiplicity"]["holm_p"] = holm_correction(ps)

    gates = report["gates"]
    structural_ok = gates["A1_G4"]["pass"]
    g1 = gates["A1_G1"]["pass"]
    all_sci = all(gates[k]["pass"] for k in ("A1_G1", "A1_G2", "A1_G3", "A1_G4", "A1_G6"))
    if complete:
        bd_positive = all(
            v > 0 for v in
            gates["A1_G5"]["G5_i_B_minus_D"]["true_cost_net_whole"]["per_seed"].values())
        if not structural_ok or not g1:
            verdict = "FAIL"
        elif all_sci and bd_positive:
            verdict = "PASS"
        else:
            verdict = "CONDITIONAL PASS"
    else:
        # The verdict rule needs B-D. Withholding it is not a formality: a
        # PASS requires the generic-perturbation control, and announcing one
        # before condition D exists would be claiming that control passed.
        bd_positive = None
        verdict = "WITHHELD -- condition D incomplete"
    report["verdict"] = verdict
    report["verdict_inputs"] = {
        "structural_mechanism_ok": structural_ok, "A1_G1": g1,
        "all_scientific_gates": all_sci, "B_minus_D_positive_all_seeds": bd_positive}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nA1 gates ({len(seeds)} seeds)")
    for k in ("A1_G1", "A1_G2", "A1_G3", "A1_G4", "A1_G6"):
        print(f"  [{'PASS' if gates[k]['pass'] else 'FAIL'}] {k}")
    ba = report["contrasts"]["B-A"]["true_cost_net_whole"]
    print(f"\n  B-A true cost (whole-run): {ba['mean']:+.3f}  per-seed {ba['per_seed']}")
    if gates["A1_G5"].get("available") is False:
        print("  [held] A1_G5 (B-D): condition D incomplete")
    else:
        print(f"  [ est] A1_G5 (B-D, not gated at n={len(seeds)})")
        bd = gates["A1_G5"]["G5_i_B_minus_D"]["true_cost_net_whole"]
        print(f"  B-D true cost (whole-run): {bd['mean']:+.3f}  per-seed {bd['per_seed']}")
    print(f"\nVERDICT: {verdict}   -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
