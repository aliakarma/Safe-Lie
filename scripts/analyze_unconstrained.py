#!/usr/bin/env python
"""Unconstrained-control analyser — condition U vs condition A.

Implements docs/unconstrained_control.md exactly. Written and committed
BEFORE seed 0 finished, per section 11 of that document, and not modified
after any control number became visible.

    A = clean, constrained    (lambda_max = 25.0)   -- G9/G10, already frozen
    U = clean, unconstrained  (lambda_max =  0.0)   -- this control

    primary:  J_C,unconstrained - J_C,constrained  =  U - A
              on whole-run network-average true cost, HIGHER IS WORSE

There is **no effect-size gate** and none is invented here: section 8 of
the pre-declaration records that no prior declaration fixes a bar for
U-A, and that writing one with the constrained side already in hand would
be choosing a bar against half-known data. The primary quantity is an
estimate with per-seed values and a paired 95% CI, interpreted in words.

Paired by seed throughout. Per decision D6 (MIN_SEEDS_FOR_INFERENCE = 5)
the paired t is emitted as a labelled sensitivity analysis only; at three
seeds the evidence is the effect size and per-seed sign consistency.

Usage:
    python scripts/analyze_unconstrained.py
    python scripts/analyze_unconstrained.py --out results/runs_unconstrained/unconstrained_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D_BUDGET = 25.0
MIN_SEEDS_FOR_INFERENCE = 5      # decision D6
SEEDS = (0, 1, 2)

RUN_DIRS = {
    ("A", 0): "results/runs_constraint_batch_g9/g9_batch_clean",
    ("A", 1): "results/runs_constraint_batch_g10/seed1",
    ("A", 2): "results/runs_constraint_batch_g10/seed2",
    **{("U", s): f"results/runs_unconstrained/U_seed{s}" for s in SEEDS},
}

# Section 6's list, in the order that document fixes it.
PRIMARY = "true_cost_net_whole"
REPORTED_METRICS = (
    "true_cost_net_whole", "true_cost_net_last50",
    "task_return_first20", "task_return_last50", "task_learning_gain",
    "task_return_whole",
    "violation_rate_whole", "violation_rate_last50",
    "mech_reported_whole", "mech_reported_last50",
    "mech_gap_whole", "mech_gap_last50",
    "lambda_mean", "lambda_max", "lambda_frac_positive",
    "residual_last50_sum",
    "n_agents_over_d_whole", "n_agents_over_d_last50",
)


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def load(run_dir: str) -> dict | None:
    d = ROOT / run_dir
    if not (d / "oracle.jsonl").exists() or not (d / "rounds.jsonl").exists():
        return None
    orc, rnd = read_jsonl(d / "oracle.jsonl"), read_jsonl(d / "rounds.jsonl")
    if not orc or not rnd:
        return None
    aids = sorted(orc[0]["agents"])
    K = min(len(orc), len(rnd))
    orc, rnd = orc[:K], rnd[:K]
    last = slice(-50, None)

    def oarr(f):
        return np.array([[r["agents"][a][f] for a in aids] for r in orc], dtype=float)

    def carr(f):
        return np.array([[r["constraints"][a][f] for a in aids] for r in rnd], dtype=float)

    tc, tr = oarr("true_cost_return"), oarr("episodic_task_return")
    lam, res = carr("lambda_after"), carr("constraint_residual")
    mech, gap = carr("mechanism_reported_cost_return"), oarr("detection_gap_vs_aggregate")
    viol = oarr("violated")
    net = tc.mean(axis=1)

    s = {
        "run_dir": run_dir, "rounds": K, "n_agents": len(aids),
        # --- safety
        "true_cost_net_whole": float(net.mean()),
        "true_cost_net_last50": float(net[last].mean()),
        "true_cost_net_last50_sd_over_rounds": float(net[last].std(ddof=1)),
        "true_cost_per_agent_whole": [float(x) for x in tc.mean(axis=0)],
        "true_cost_per_agent_last50": [float(x) for x in tc[last].mean(axis=0)],
        "n_agents_over_d_whole": int((tc.mean(axis=0) > D_BUDGET).sum()),
        "n_agents_over_d_last50": int((tc[last].mean(axis=0) > D_BUDGET).sum()),
        "violation_rate_whole": float(viol.mean()),
        "violation_rate_last50": float(viol[last].mean()),
        # --- task
        "task_return_whole": float(tr.mean()),
        "task_return_first20": float(tr[:20].mean()),
        "task_return_first50": float(tr[:50].mean()),
        "task_return_last50": float(tr[last].mean()),
        "task_learning_gain": float(tr[last].mean() - tr[:50].mean()),
        # --- dual
        "lambda_mean": float(lam.mean()),
        "lambda_max": float(lam.max()),
        "lambda_frac_positive": float((lam > 0).mean()),
        "lambda_exactly_zero_everywhere": bool((lam == 0.0).all()),
        "lambda_per_agent_last50": [float(x) for x in lam[last].mean(axis=0)],
        "residual_last50_per_agent": [float(x) for x in res[last].mean(axis=0)],
        "residual_last50_sum": float(res[last].mean(axis=0).sum()),
        "residual_whole_mean": float(res.mean()),
        # --- mechanism
        "mech_reported_whole": float(mech.mean()),
        "mech_reported_last50": float(mech[last].mean()),
        "mech_gap_whole": float(gap.mean()),
        "mech_gap_last50": float(gap[last].mean()),
        # --- learning trajectory (section 6), decimated to 25 points
        "trajectory": {
            "round": list(range(0, K, max(1, K // 25))),
            "true_cost_net": [float(net[i]) for i in range(0, K, max(1, K // 25))],
            "task_return": [float(tr[i].mean()) for i in range(0, K, max(1, K // 25))],
            "lambda_mean": [float(lam[i].mean()) for i in range(0, K, max(1, K // 25))],
        },
        "finite": bool(all(np.isfinite(a).all() for a in (tc, tr, lam, res, mech, gap))),
    }
    ppo0 = rnd[0]["constraints"][aids[0]].get("ppo") or {}
    for key in ("approx_kl", "kl"):
        if key in ppo0:
            k = np.array([[r["constraints"][a]["ppo"][key] for a in aids] for r in rnd],
                         dtype=float)
            s["kl_median"], s["kl_p95"] = float(np.median(k)), float(np.percentile(k, 95))
            break
    if "entropy" in ppo0:
        e = np.array([[r["constraints"][a]["ppo"]["entropy"] for a in aids] for r in rnd],
                     dtype=float)
        s["entropy_first10"], s["entropy_last10"] = float(e[:10].mean()), float(e[-10:].mean())
    md = ROOT / run_dir / "run_metadata.json"
    if md.exists():
        m = json.loads(md.read_text(encoding="utf-8"))
        s["wall_clock_s"] = (m.get("timing") or {}).get("wall_clock_s")
        s["lambda_max_config"] = (m.get("config_snapshot") or {}).get("dual", {}).get("lambda_max")
    return s


def paired(summ: dict, metric: str, x: str, y: str, seeds) -> dict:
    """Verbatim from scripts/analyze_a1.py."""
    diffs = np.array([summ[(x, s)][metric] - summ[(y, s)][metric] for s in seeds], dtype=float)
    n = len(diffs)
    m = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    out = {
        "contrast": f"{x}-{y}", "metric": metric, "n": n,
        "per_seed": {str(s): float(summ[(x, s)][metric] - summ[(y, s)][metric]) for s in seeds},
        f"{x}_per_seed": {str(s): summ[(x, s)][metric] for s in seeds},
        f"{y}_per_seed": {str(s): summ[(y, s)][metric] for s in seeds},
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
            t, p = sps.ttest_rel([summ[(x, s)][metric] for s in seeds],
                                 [summ[(y, s)][metric] for s in seeds])
            out["paired_t"] = {"t": float(t), "p": float(p),
                               "label": "SENSITIVITY ONLY -- decision D6 forbids inference "
                                        f"below {MIN_SEEDS_FOR_INFERENCE} seeds"}
            n_pos = int((diffs > 0).sum())
            out["sign_test_p"] = float(sps.binomtest(n_pos, n, 0.5).pvalue)
            out["sign_test_min_attainable_p"] = float(sps.binomtest(n, n, 0.5).pvalue)
    except Exception as exc:                                     # pragma: no cover
        out["stats_error"] = str(exc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/runs_unconstrained/unconstrained_report.json")
    ap.add_argument("--allow-partial", action="store_true",
                    help="summarise whatever seeds exist; the U-A contrast is "
                         "WITHHELD unless all three are present")
    args = ap.parse_args()
    seeds = list(SEEDS)

    summ, missing = {}, []
    for (c, s), path in sorted(RUN_DIRS.items()):
        v = load(path)
        if v is None:
            missing.append(f"{c}_seed{s} ({path})")
            continue
        summ[(c, s)] = v

    report: dict = {
        "predeclaration": "docs/unconstrained_control.md",
        "design": {
            "conditions": ["A clean constrained (lambda_max=25.0)",
                           "U clean unconstrained (lambda_max=0.0)"],
            "seeds": seeds, "budget_d": D_BUDGET, "N": 6, "M": 3,
            "primary": "U-A on true_cost_net_whole (network-average, higher is worse)",
            "gate": "none -- no predeclared effect-size bar exists (section 8)",
        },
        "missing_runs": missing,
        "per_run": {f"{c}_seed{s}": v for (c, s), v in sorted(summ.items())},
    }

    have = {c for c in ("A", "U") if all((c, s) in summ for s in seeds)}
    report["conditions_available"] = sorted(have)
    report["complete"] = have == {"A", "U"}

    if not report["complete"]:
        report["status"] = "INCOMPLETE -- U-A contrast withheld"
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Unconstrained control INCOMPLETE. Missing:")
        for m in missing:
            print("  -", m)
        print(f"\nPartial summary -> {args.out}")
        return 0 if args.allow_partial else 2

    report["contrasts"] = {m: paired(summ, m, "U", "A", seeds) for m in REPORTED_METRICS}

    # Per-agent, per-seed increments -- section 6 requires the per-agent values,
    # and the network average alone would hide a redistribution.
    report["per_agent"] = {
        str(s): {
            "U_whole": summ[("U", s)]["true_cost_per_agent_whole"],
            "A_whole": summ[("A", s)]["true_cost_per_agent_whole"],
            "increment_whole": [u - a for u, a in zip(
                summ[("U", s)]["true_cost_per_agent_whole"],
                summ[("A", s)]["true_cost_per_agent_whole"])],
            "U_last50": summ[("U", s)]["true_cost_per_agent_last50"],
            "A_last50": summ[("A", s)]["true_cost_per_agent_last50"],
            "increment_last50": [u - a for u, a in zip(
                summ[("U", s)]["true_cost_per_agent_last50"],
                summ[("A", s)]["true_cost_per_agent_last50"])],
            "n_agents_agreeing_with_net_sign": int(sum(
                np.sign(u - a) == np.sign(summ[("U", s)]["true_cost_net_whole"]
                                          - summ[("A", s)]["true_cost_net_whole"])
                for u, a in zip(summ[("U", s)]["true_cost_per_agent_whole"],
                                summ[("A", s)]["true_cost_per_agent_whole"]))),
        } for s in seeds
    }

    report["treatment_integrity"] = {
        "U_lambda_exactly_zero_all_seeds": all(
            summ[("U", s)]["lambda_exactly_zero_everywhere"] for s in seeds),
        "U_lambda_max_config": {str(s): summ[("U", s)].get("lambda_max_config") for s in seeds},
        "A_lambda_mean": {str(s): summ[("A", s)]["lambda_mean"] for s in seeds},
        "all_finite": all(v["finite"] for v in summ.values()),
    }

    report["statistical_policy"] = {
        "n": 3,
        "pairing": "by seed, common random numbers on the source streams",
        "inference": "effect size + paired 95% CI + per-seed signs",
        "paired_t": f"SENSITIVITY ONLY (decision D6, MIN_SEEDS_FOR_INFERENCE="
                    f"{MIN_SEEDS_FOR_INFERENCE})",
        "seeds_added_after_seeing_results": False,
        "power_analysis_retrofitted": False,
    }

    prim = report["contrasts"][PRIMARY]
    report["interpretation"] = {
        "primary_quantity": "J_C,unconstrained - J_C,constrained",
        "value": prim["mean"],
        "ci95": prim.get("ci95"),
        "per_seed": prim["per_seed"],
        "all_same_sign": prim["all_same_sign"],
        "cohens_dz": prim["cohens_dz"],
        "reading": (
            "Removing the dual constraint RAISES the learned true safety cost; "
            "the constraint is behaviourally active."
            if prim["mean"] > 0 and prim["all_same_sign"] else
            "Removing the dual constraint LOWERS the learned true safety cost -- "
            "an unexpected sign that must be reported and explained, not suppressed."
            if prim["mean"] < 0 and prim["all_same_sign"] else
            "The contrast is small and/or sign-inconsistent across seeds. On this "
            "evidence the dual is not demonstrably load-bearing at this operating "
            "point, and A1/A2 must be re-read in that light (section 7)."),
        "does_not_establish": [
            "that the constraint is SUFFICIENT for safety -- the constrained runs "
            "violate on ~47% of rounds and 2-3 of 6 agents sit above d",
            "any transfer to another environment, topology, budget, M or f",
            "anything about aggregators other than the mean",
        ],
    }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nUnconstrained control (n=3, paired by seed)")
    print(f"  treatment integrity: lambda exactly 0 in all U seeds = "
          f"{report['treatment_integrity']['U_lambda_exactly_zero_all_seeds']}")
    print(f"\n  {'metric':<26} {'U mean':>10} {'A mean':>10} {'U-A':>10} {'95% CI':>22}  signs")
    for m in REPORTED_METRICS:
        c = report["contrasts"][m]
        um = np.mean([summ[("U", s)][m] for s in seeds])
        am = np.mean([summ[("A", s)][m] for s in seeds])
        ci = c.get("ci95")
        cis = f"[{ci[0]:+8.3f},{ci[1]:+8.3f}]" if ci else " " * 22
        print(f"  {m:<26} {um:>10.4f} {am:>10.4f} {c['mean']:>+10.4f} {cis}  "
              f"{'same' if c['all_same_sign'] else 'mixed'}")
    print(f"\n  PRIMARY  J_C,unconstrained - J_C,constrained = {prim['mean']:+.4f}")
    if prim.get("ci95"):
        print(f"           95% CI [{prim['ci95'][0]:+.4f}, {prim['ci95'][1]:+.4f}]  "
              f"dz = {prim['cohens_dz']:+.2f}")
    print(f"           per seed: {prim['per_seed']}")
    print(f"\n  {report['interpretation']['reading']}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
