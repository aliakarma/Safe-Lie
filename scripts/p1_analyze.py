#!/usr/bin/env python
"""P1 analysis: localization vs consensus spreading.

docs/p1_concentrated_attack_gates.md Sections 7-8. Computes every primary
measurement from committed artifacts and writes
`results/runs_p1/p1_report.json` plus the primary figure.

THREE DISPLACEMENT OBJECTS ARE KEPT APART, deliberately:

  observed   lambda(attacked run) - lambda(clean run, same seed, same
             topology). The thing an operator could actually compare.
             Confounded by primal divergence: the two runs trained
             different policies. Available for Condition R only -- there is
             no clean W=I run (gates doc Section 6).

  linear     the UNPROJECTED recursion e_{k+1} = W e_k + eta * delta_k
             driven by the ACTUALLY INJECTED delta_k. This is Proposition
             `cor:spread`'s object under its own interior assumption. It is
             a theoretical prediction, not a measurement, and is labelled as
             such everywhere.

  realized   lambda itself, per agent. No counterfactual, no subtraction,
             so nothing to saturate or confound. Reported for both
             conditions and carrying the monitor.

The projected within-run counterfactual is NOT reported as a primary
quantity: under W=I every coordinate saturates against a rail and the
statistic measures the projection rather than the mechanism (gates doc
Section 6).

Usage:
    python scripts/p1_analyze.py
    python scripts/p1_analyze.py --out results/runs_p1/p1_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safelie.consensus.mixing import second_largest_singular_value  # noqa: E402
from safelie.consensus.topologies import build_topology  # noqa: E402
from safelie.eval.monitor import (  # noqa: E402
    calibrate_threshold,
    concentration_ratio,
    detection_frequency,
    median_deviation_trace,
    per_agent_detection_frequency,
)

RUNS = ROOT / "results" / "runs_p1"
CLEAN_RING = {
    0: ROOT / "results/runs_constraint_batch_g9/g9_batch_clean",
    1: ROOT / "results/runs_constraint_batch_g10/seed1",
    2: ROOT / "results/runs_constraint_batch_g10/seed2",
}
TAU_GRID = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]  # pre-declared, gates doc Section 8
FPR = 0.05


def load_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_arrays(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    cfg = meta["config_snapshot"]
    rounds = load_jsonl(run_dir / "rounds.jsonl")
    aids = sorted(rounds[0]["constraints"])
    get = lambda key: np.array([[r["constraints"][a][key] for a in aids] for r in rounds])  # noqa: E731
    out = {
        "cfg": cfg, "meta": meta, "aids": aids, "K": len(rounds),
        "lam": get("lambda_after"),
        "residual": get("constraint_residual"),
        "mech_report": get("mechanism_reported_cost_return"),
    }
    out["delta"] = (get("injected_delta") if "injected_delta" in rounds[0]["constraints"][aids[0]]
                    else np.zeros_like(out["lam"]))
    orc = run_dir / "oracle.jsonl"
    out["oracle"] = load_jsonl(orc) if orc.exists() else []
    return out


def linear_prediction(W: np.ndarray, delta: np.ndarray, eta: float) -> np.ndarray:
    """e_{k+1} = W e_k + eta * delta_k, unprojected. Proposition cor:spread."""
    e = np.zeros(delta.shape[1])
    out = []
    for k in range(delta.shape[0]):
        e = W @ e + eta * delta[k]
        out.append(e.copy())
    return np.array(out)


def summarize(vec_t: np.ndarray) -> dict:
    """Dispersion summary of a (K, N) per-agent quantity."""
    return {
        "mean": vec_t.mean(axis=1).tolist(),
        "max": vec_t.max(axis=1).tolist(),
        "min": vec_t.min(axis=1).tolist(),
        "sd": vec_t.std(axis=1, ddof=1).tolist(),
        "concentration": concentration_ratio(vec_t).tolist(),
        "l1": np.abs(vec_t).sum(axis=1).tolist(),
    }


def tail(x, n=50):
    a = np.asarray(x, dtype=float)[-n:]
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RUNS / "p1_report.json"))
    ap.add_argument("--figure", default=str(RUNS / "p1_displacement.png"))
    args = ap.parse_args()

    report: dict = {"campaign": "P1", "tau_grid": TAU_GRID, "runs": {}, "notes": []}
    found = []
    for cond in ("R", "I"):
        for seed in (0, 1, 2):
            rd = RUNS / f"{cond}_seed{seed}"
            if (rd / "rounds.jsonl").exists():
                found.append((cond, seed, rd))
    if not found:
        print("No P1 runs found yet.")
        return 0
    print(f"found {len(found)} P1 run(s): {[f'{c}_seed{s}' for c, s, _ in found]}\n")

    for cond, seed, rd in found:
        A = run_arrays(rd)
        cfg = A["cfg"]
        n = cfg["env"]["n_agents"]
        eta = cfg["dual"]["eta_lambda"]
        W = build_topology(cfg["topology"]["name"], n,
                           p=cfg["topology"].get("p"), graph_seed=cfg["topology"].get("graph_seed", 0))
        owners = cfg["attack"].get("corrupted_owner_ids") or []
        j = A["aids"].index(owners[0]) if owners else None
        key = f"{cond}_seed{seed}"
        R: dict = {
            "run_id": cfg["run_id"], "seed": seed, "condition": cond,
            "topology": cfg["topology"]["name"],
            "sigma2": second_largest_singular_value(W),
            "K": A["K"], "attacked_owner": owners, "attacked_owner_index": j,
            "corrupted_source": cfg["attack"].get("corrupted_source_ids"),
            "eta_lambda": eta, "lambda_max": cfg["dual"]["lambda_max"],
        }

        # ---- 1/11 injected perturbation ---------------------------------
        delta = A["delta"]
        R["injected"] = {
            "per_round_vector_round0": delta[0].tolist(),
            "concentration_mean": float(np.nanmean(concentration_ratio(delta))),
            "concentration_all_one": bool(np.allclose(np.nan_to_num(concentration_ratio(delta), nan=1.0), 1.0)),
            "total_mass_1T_sum": float(delta.sum()),
            "per_round_mass": float(delta.sum(axis=1).mean()),
        }

        # ---- 2/4 realized multipliers -----------------------------------
        lam = A["lam"]
        R["realized_lambda"] = {
            "per_agent_mean": lam.mean(axis=0).tolist(),
            "per_agent_final": lam[-1].tolist(),
            "dispersion_sd_mean": float(lam.std(axis=1, ddof=1).mean()),
            "dispersion_sd_last50": float(lam[-50:].std(axis=1, ddof=1).mean()),
            "frac_at_floor": float((lam <= 1e-12).mean()),
            "frac_at_ceiling": float((lam >= cfg["dual"]["lambda_max"] - 1e-12).mean()),
            "attacked_owner_mean": float(lam[:, j].mean()) if j is not None else None,
            "fleet_ex_attacked_mean": (
                float(np.delete(lam, j, axis=1).mean()) if j is not None else None),
        }

        # ---- 6/7 theoretical predictions --------------------------------
        e_lin = linear_prediction(W, delta, eta)
        R["linear_prediction"] = {
            "final_vector": e_lin[-1].tolist(),
            "concentration_last": float(concentration_ratio(e_lin[-1:])[0]),
            "concentration_last50_mean": tail(concentration_ratio(e_lin)),
            "aggregate_1T_eK": float(e_lin[-1].sum()),
            "uniform_component_eta_delta_K_over_N": float(eta * delta.sum() / n),
            "residual_bound_1_over_1_minus_sigma2": (
                float(1.0 / (1.0 - R["sigma2"])) if R["sigma2"] < 1 - 1e-12 else None),
            "residual_norm": float(np.linalg.norm(e_lin[-1] - eta * delta.sum() / n)),
        }
        # Theorem thm:mass: 1^T e_K == eta * sum_k 1^T delta_k, any W.
        R["linear_prediction"]["mass_identity_error"] = abs(
            float(e_lin[-1].sum() - eta * delta.sum()))

        # ---- 3 observed displacement vs a real clean run ----------------
        R["observed_displacement"] = None
        if cond == "R" and CLEAN_RING[seed].joinpath("rounds.jsonl").exists():
            C = run_arrays(CLEAN_RING[seed])
            k = min(A["K"], C["K"])
            e_obs = A["lam"][:k] - C["lam"][:k]
            # Gates doc Section 17. A coordinate pinned at lambda=0 in BOTH
            # runs has identically zero displacement, which shrinks the
            # support and makes the concentration ratio rise MECHANICALLY.
            # So concentration is reported twice -- over all rounds, and over
            # the interior rounds where it is interpretable -- and never
            # without the saturation fraction beside it.
            floor_a = (A["lam"][:k] <= 1e-12).sum(axis=1)
            floor_c = (C["lam"][:k] <= 1e-12).sum(axis=1)
            interior = (floor_a == 0) & (floor_c == 0)
            conc_obs = concentration_ratio(e_obs)
            R["observed_displacement"] = {
                "reference": str(CLEAN_RING[seed].relative_to(ROOT)),
                "caveat": "cross-run: the two runs trained different policies; "
                          "this is descriptive, not Proposition cor:spread's e_k",
                "final_vector": e_obs[-1].tolist(),
                "summary_last50": {kk: tail(v) for kk, v in summarize(e_obs).items()
                                   if kk in ("concentration", "l1")},
                "per_agent_mean_last50": e_obs[-50:].mean(axis=0).tolist(),
                "n_coords_at_floor_attacked": floor_a.tolist(),
                "n_coords_at_floor_reference": floor_c.tolist(),
                "n_interior_rounds": int(interior.sum()),
                "interior_round_indices": np.flatnonzero(interior).tolist(),
                "saturation_fraction_all_rounds": float(1.0 - interior.mean()),
                "saturation_fraction_last50": float(1.0 - interior[-50:].mean()),
                "concentration_all_rounds_mean": float(np.nanmean(conc_obs)),
                "concentration_last50_mean": tail(conc_obs),
                "concentration_interior_only_mean": (
                    float(np.nanmean(conc_obs[interior])) if interior.any() else None),
                "concentration_interior_only_final": (
                    float(conc_obs[interior][-1]) if interior.any() else None),
                "concentration_note": (
                    "concentration_last50_mean is computed over the MOST saturated "
                    "window and is not a localization measurement wherever "
                    "saturation_fraction_last50 > 0; compare against "
                    "concentration_interior_only_mean (gates doc Section 17)"),
            }
            # Gates doc Section 18. A concentration ratio is scale-free, so it
            # will happily describe the shape of a vector that is entirely
            # policy-divergence noise. Carry the magnitude next to it always.
            l1_obs = float(np.abs(e_obs[-1]).sum())
            l1_lin = float(np.abs(e_lin[-1]).sum())
            ratio = (l1_obs / l1_lin) if l1_lin > 1e-12 else None
            lam_level = float(A["lam"][-50:].mean())
            warn = None
            if ratio is not None and ratio < 0.10:
                warn = (
                    f"realized displacement is {100 * ratio:.1f}% of the unprojected linear "
                    "prediction over the identical injected delta_k; the "
                    "concentration figures beside this describe a vector within "
                    "policy-divergence noise and are NOT evidence of spreading "
                    "or localization (gates doc Section 18)")
            R["observed_displacement"].update({
                "l1_norm_final": l1_obs,
                "l1_norm_linear_final": l1_lin,
                "l1_ratio_to_linear": ratio,
                "mean_abs_displacement_last50": float(np.abs(e_obs[-50:]).mean()),
                "mean_lambda_level_last50": lam_level,
                "displacement_vs_lambda_level": (
                    float(np.abs(e_obs[-50:]).mean() / lam_level) if lam_level > 1e-12 else None),
                "signs_mixed": bool((e_obs[-1] > 0).any() and (e_obs[-1] < 0).any()),
                "magnitude_warning": warn,
            })
        else:
            R["observed_displacement_unavailable_reason"] = (
                "no clean run exists under W=I (gates doc Section 6); an "
                "attack-attributable displacement cannot be claimed for Condition I"
                if cond == "I" else "clean ring reference missing")

        # ---- 8 aggregate displacement -----------------------------------
        R["aggregate_displacement_1T"] = {
            "linear": float(e_lin[-1].sum()),
            "observed": (float(np.array(R["observed_displacement"]["final_vector"]).sum())
                         if R["observed_displacement"] else None),
        }

        # ---- 9/10 oracle + mechanism gap --------------------------------
        if A["oracle"]:
            o = A["oracle"]
            R["oracle"] = {"n_records": len(o), "fields": sorted(o[-1])}
            for nm in ("true_cost_return", "network_average_true_cost", "detection_gap",
                       "violation_rate", "peak_violation", "episodic_task_return"):
                v = [rec[nm] for rec in o if nm in rec]
                if v and isinstance(v[0], (int, float)):
                    R["oracle"][nm + "_last50_mean"] = tail(v)
        # mechanism-level cost-report gap, learner side
        R["mechanism_reported_cost_return_last50_mean"] = float(A["mech_report"][-50:].mean())

        # ---- monitor -----------------------------------------------------
        tr = median_deviation_trace(lam)
        mon = {
            "statistic": "max_i |lambda_k^i - median_j(lambda_k^j)|",
            "fleet_statistic_mean": float(tr.fleet_statistic.mean()),
            "fleet_statistic_last50_mean": float(tr.fleet_statistic[-50:].mean()),
            "per_agent_mean_deviation": tr.deviation.mean(axis=0).tolist(),
            "attacked_owner_mean_deviation": (
                float(tr.deviation[:, j].mean()) if j is not None else None),
            "sweep_detection_frequency": {str(t): detection_frequency(tr, t) for t in TAU_GRID},
            "sweep_attacked_owner_detection": {
                str(t): float(per_agent_detection_frequency(tr, t)[j]) for t in TAU_GRID
            } if j is not None else None,
        }
        if cond == "R" and CLEAN_RING[seed].joinpath("rounds.jsonl").exists():
            ctr = median_deviation_trace(run_arrays(CLEAN_RING[seed])["lam"])
            tau = calibrate_threshold(ctr, FPR)
            mon["clean_calibrated"] = {
                "tau": tau, "false_positive_rate_target": FPR,
                "clean_detection_frequency": detection_frequency(ctr, tau),
                "attacked_detection_frequency": detection_frequency(tr, tau),
                "clean_fleet_statistic_mean": float(ctr.fleet_statistic.mean()),
            }
            # Gates doc Section 18: the attacked sweep alone is uninterpretable.
            # A detection frequency of 0.99 at tau=0.05 says only that the
            # threshold sits below the fleet's ordinary dispersion -- which the
            # clean column is what reveals.
            mon["sweep_detection_frequency_clean"] = {
                str(t_): detection_frequency(ctr, t_) for t_ in TAU_GRID
            }
            # Report the MARGIN, not a boolean. A bare "exceeds at some tau"
            # fires on a 0.4-point gap (one round in 250) where both columns
            # sit at ~0.99, and reads like detection when it is noise.
            excess = {str(t_): detection_frequency(tr, t_) - detection_frequency(ctr, t_)
                      for t_ in TAU_GRID}
            mon["detection_excess_over_clean"] = excess
            mon["max_detection_excess_over_clean"] = float(max(excess.values()))
            mon["attacked_materially_exceeds_clean"] = bool(
                max(excess.values()) > 0.05)  # > 5 points, i.e. > ~12 rounds of 250
        else:
            mon["clean_calibrated"] = None
            mon["clean_calibration_unavailable"] = (
                "no clean W=I run; a ring-calibrated tau applied to W=I would "
                "measure the topology's intrinsic dispersion, not the adversary"
                if cond == "I" else "clean ring reference missing")
        R["monitor"] = mon

        report["runs"][key] = R
        print(f"--- {key} ({R['topology']}, sigma2={R['sigma2']:.4f}) K={R['K']}")
        print(f"    injected delta concentration : {R['injected']['concentration_mean']:.6f} "
              f"(all exactly 1.0: {R['injected']['concentration_all_one']})")
        print(f"    linear e_K concentration     : {R['linear_prediction']['concentration_last']:.6f}"
              f"   (1/N = {1/n:.4f})")
        print(f"    linear e_K                   : "
              f"{np.array2string(np.array(R['linear_prediction']['final_vector']), precision=3)}")
        print(f"    realized lambda per-agent mean: "
              f"{np.array2string(lam.mean(axis=0), precision=3)}")
        print(f"    lambda at floor/ceiling      : "
              f"{100*R['realized_lambda']['frac_at_floor']:.1f}% / "
              f"{100*R['realized_lambda']['frac_at_ceiling']:.1f}%")
        print(f"    monitor fleet stat (last50)  : {mon['fleet_statistic_last50_mean']:.4f}")
        if R["observed_displacement"]:
            print(f"    observed e_K (vs clean ring) : "
                  f"{np.array2string(np.array(R['observed_displacement']['final_vector']), precision=3)}")
        print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {args.out}")

    _figure(report, Path(args.figure))
    return 0


def _figure(report: dict, path: Path) -> None:
    """Primary figure: the six-agent lambda vector over time, per condition.

    One line per agent. Per-agent structure is never averaged away.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(figure skipped: {exc})")
        return
    runs = report["runs"]
    seeds = sorted({r["seed"] for r in runs.values()})
    conds = [c for c in ("I", "R") if any(r["condition"] == c for r in runs.values())]
    if not seeds or not conds:
        return
    fig, axes = plt.subplots(len(conds), len(seeds), figsize=(4.2 * len(seeds), 3.4 * len(conds)),
                             squeeze=False, sharex=True)
    for ri, cond in enumerate(conds):
        for ci, seed in enumerate(seeds):
            ax = axes[ri][ci]
            key = f"{cond}_seed{seed}"
            if key not in runs:
                ax.set_visible(False)
                continue
            r = runs[key]
            rd = RUNS / key
            lam_t = np.array([[rec["constraints"][a]["lambda_after"]
                               for a in sorted(rec["constraints"])]
                              for rec in load_jsonl(rd / "rounds.jsonl")])
            j = r["attacked_owner_index"]
            for i in range(lam_t.shape[1]):
                attacked = (i == j)
                ax.plot(lam_t[:, i], lw=2.0 if attacked else 1.0,
                        color="crimson" if attacked else "0.55",
                        zorder=3 if attacked else 2,
                        label=f"agent_{i}" + (" (attacked)" if attacked else ""))
            ax.set_title(f"{'W=I' if cond == 'I' else 'ring'}  seed {seed}", fontsize=10)
            ax.set_ylabel(r"$\lambda^i_k$" if ci == 0 else "")
            ax.set_xlabel("round $k$" if ri == len(conds) - 1 else "")
            if ri == 0 and ci == len(seeds) - 1:
                ax.legend(fontsize=6, ncol=2, frameon=False)
    fig.suptitle("P1: per-agent multipliers under a concentrated owner-level corruption\n"
                 "(attacked owner in red; per-agent structure not averaged)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"figure -> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
