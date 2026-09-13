#!/usr/bin/env python
"""Zero-compute reanalysis: the MECHANISM-level detection gap.

`scripts/analyze_a{1,2}.py` summarise `oracle.jsonl:detection_gap`, which
`experiment.py` defines as

    detection_gap = true_cost_return - own_critic_reported_cost_return

and whose own comment says it "measures critic estimation error, not the
attack". The quantity the paper actually studies -- how far the value the
*constraint mechanism* consumes sits below the truth -- is

    detection_gap_vs_aggregate = true_cost_return - mechanism_reported_cost_return

It is logged on every oracle row of every committed run and was never
analysed. This script recomputes the A1/A2 contrasts on it.

Nothing is re-run and no log is modified. The paired-difference machinery,
the seed set and the summarisation windows are copied verbatim from
`scripts/analyze_a1.py`; only the gap field changes. Per decision D6 the
paired t is emitted as a labelled sensitivity analysis only.

Usage:
    python scripts/reanalyze_mechanism_gap.py --out results/mechanism_gap_reanalysis.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

D_BUDGET = 25.0
MIN_SEEDS_FOR_INFERENCE = 5
SEEDS = (0, 1, 2)

RUN_DIRS = {
    ("A", 0): "results/runs_constraint_batch_g9/g9_batch_clean",
    ("A", 1): "results/runs_constraint_batch_g10/seed1",
    ("A", 2): "results/runs_constraint_batch_g10/seed2",
    **{("B", s): f"results/runs_a1/B_seed{s}" for s in SEEDS},
    **{("D", s): f"results/runs_a1/D_seed{s}" for s in SEEDS},
    **{("C", s): f"results/runs_a2/C_seed{s}" for s in SEEDS},
    **{("E", s): f"results/runs_a2/E_seed{s}" for s in SEEDS},
}


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def summarize(run_dir: str) -> dict | None:
    d = Path(run_dir)
    if not (d / "oracle.jsonl").exists() or not (d / "rounds.jsonl").exists():
        return None
    orc, rnd = read_jsonl(d / "oracle.jsonl"), read_jsonl(d / "rounds.jsonl")
    aids = sorted(orc[0]["agents"].keys())
    K = min(len(orc), len(rnd))
    orc, rnd = orc[:K], rnd[:K]
    last = slice(-50, None)

    def oarr(field):
        return np.array([[r["agents"][a][field] for a in aids] for r in orc], dtype=float)

    def carr(field):
        return np.array([[r["constraints"][a][field] for a in aids] for r in rnd], dtype=float)

    mech_gap = oarr("detection_gap_vs_aggregate")        # K x N, the corrected metric
    own_gap = oarr("detection_gap")                      # K x N, what was reported before
    true_cost = oarr("true_cost_return")
    mech_rep_oracle = oarr("mechanism_reported_cost_return")
    mech_rep_rounds = carr("mechanism_reported_cost_return")

    # independent reconstruction: the field must equal truth - mechanism value
    recon = true_cost - mech_rep_oracle
    # and the oracle's mechanism value must be the one rounds.jsonl fed the dual
    cross = float(np.abs(mech_rep_oracle - mech_rep_rounds).max())

    return {
        "run_dir": run_dir,
        "rounds": K,
        "mech_gap_whole": float(mech_gap.mean()),
        "mech_gap_last50": float(mech_gap[last].mean()),
        "mech_gap_per_agent_whole": [float(x) for x in mech_gap.mean(axis=0)],
        "own_critic_gap_whole": float(own_gap.mean()),
        "own_critic_gap_last50": float(own_gap[last].mean()),
        "true_cost_net_whole": float(true_cost.mean(axis=1).mean()),
        "mech_reported_whole": float(mech_rep_rounds.mean()),
        # verification
        "recon_max_abs_error": float(np.abs(recon - mech_gap).max()),
        "oracle_vs_rounds_mech_max_abs_error": cross,
        "identity_check_whole": float(
            (true_cost.mean(axis=1).mean() - mech_rep_rounds.mean()) - mech_gap.mean()),
        "finite": bool(np.isfinite(mech_gap).all()),
    }


def paired(summaries: dict, metric: str, x: str, y: str, seeds) -> dict:
    """Verbatim from scripts/analyze_a1.py."""
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
    from scipy import stats as sps
    if n > 1 and sd > 0:
        tcrit = float(sps.t.ppf(0.975, n - 1))
        out["ci95"] = [m - tcrit * se, m + tcrit * se]
        t, p = sps.ttest_rel([summaries[(x, s)][metric] for s in seeds],
                             [summaries[(y, s)][metric] for s in seeds])
        out["paired_t"] = {"t": float(t), "p": float(p),
                           "label": "SENSITIVITY ONLY -- decision D6 forbids inference "
                                    f"below {MIN_SEEDS_FOR_INFERENCE} seeds"}
    return out


def paired_of_diffs(per_seed_diffs: dict) -> dict:
    """Summarise an already-formed per-seed difference (used for the interaction)."""
    seeds = sorted(per_seed_diffs)
    diffs = np.array([per_seed_diffs[s] for s in seeds], dtype=float)
    n = len(diffs)
    m, sd = float(diffs.mean()), float(diffs.std(ddof=1))
    se = sd / math.sqrt(n)
    from scipy import stats as sps
    tcrit = float(sps.t.ppf(0.975, n - 1))
    return {
        "n": n, "per_seed": {str(s): float(per_seed_diffs[s]) for s in seeds},
        "mean": m, "sd": sd, "se": se,
        "ci95": [m - tcrit * se, m + tcrit * se],
        "all_same_sign": bool(len({np.sign(d) for d in diffs} - {0.0}) <= 1
                              and not np.any(diffs == 0)),
        "cohens_dz": float(m / sd) if sd > 0 else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/mechanism_gap_reanalysis.json")
    args = ap.parse_args()
    seeds = list(SEEDS)

    summaries, missing = {}, []
    for (c, s), path in sorted(RUN_DIRS.items()):
        v = summarize(path)
        if v is None:
            missing.append(f"{c}_seed{s} ({path})")
            continue
        summaries[(c, s)] = v

    report = {
        "metric": "detection_gap_vs_aggregate = true_cost_return - mechanism_reported_cost_return",
        "superseded_metric": "detection_gap = true_cost_return - own_critic_reported_cost_return",
        "seeds": seeds,
        "missing_runs": missing,
        "per_run": {f"{c}_seed{s}": v for (c, s), v in sorted(summaries.items())},
        "verification": {
            "max_recon_error_over_runs": max(
                v["recon_max_abs_error"] for v in summaries.values()),
            "max_oracle_vs_rounds_mech_error": max(
                v["oracle_vs_rounds_mech_max_abs_error"] for v in summaries.values()),
            "max_identity_check_whole": max(
                abs(v["identity_check_whole"]) for v in summaries.values()),
            "all_finite": all(v["finite"] for v in summaries.values()),
        },
    }

    have = {c for c in ("A", "B", "C", "D", "E") if all((c, s) in summaries for s in seeds)}
    contrasts = {}
    for x, y in (("B", "A"), ("D", "A"), ("B", "D"), ("C", "E"), ("E", "A"), ("C", "B")):
        if {x, y} <= have:
            contrasts[f"{x}-{y}"] = {
                m: paired(summaries, m, x, y, seeds)
                for m in ("mech_gap_whole", "mech_gap_last50", "own_critic_gap_whole")
            }
    report["contrasts"] = contrasts

    if {"A", "B", "C", "E"} <= have:
        for m in ("mech_gap_whole", "mech_gap_last50"):
            inter = {s: ((summaries[("C", s)][m] - summaries[("E", s)][m])
                         - (summaries[("B", s)][m] - summaries[("A", s)][m]))
                     for s in seeds}
            ce = contrasts["C-E"][m]["mean"]
            ba = contrasts["B-A"][m]["mean"]
            report.setdefault("interaction", {})[m] = {
                "definition": "(C-E) - (B-A)",
                **paired_of_diffs(inter),
                "C_minus_E_mean": ce,
                "B_minus_A_mean": ba,
                "reduction_fraction_of_BA": (1.0 - ce / ba) if ba != 0 else None,
            }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Mechanism-gap reanalysis (n=3, paired by seed)")
    print("verification:", json.dumps(report["verification"], indent=2))
    print("\nPer-run whole-run gaps (mechanism | own-critic):")
    for k, v in report["per_run"].items():
        print(f"  {k:10s}  mech {v['mech_gap_whole']:+8.4f}   own {v['own_critic_gap_whole']:+8.4f}")
    print("\nContrasts on mech_gap_whole:")
    for name, d in contrasts.items():
        c = d["mech_gap_whole"]
        ci = c.get("ci95")
        cis = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "n/a"
        print(f"  {name:5s} {c['mean']:+8.4f}  95% CI {cis}  same-sign={c['all_same_sign']}  "
              f"dz={c['cohens_dz']:+.2f}")
    for m, d in report.get("interaction", {}).items():
        print(f"\nInteraction (C-E)-(B-A) on {m}: {d['mean']:+.4f} "
              f"95% CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
              f"same-sign={d['all_same_sign']} reduction={d['reduction_fraction_of_BA']:.1%}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
