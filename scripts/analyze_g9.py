"""G9 analysis: does the parallel trajectory-batch source architecture
survive integration into the live clean training loop?

Every threshold this script applies was fixed in `docs/g9_gates.md` and
committed (0cb5119) before any G9 artifact existed. Nothing here chooses
a bar; it only evaluates the ones already chosen, and where a gate turns
out to have been badly specified it says so and keeps the original
result.

Reads:
  results/runs_constraint_batch_g9/<run>/rounds.jsonl
  results/runs_constraint_batch_g9/<run>/oracle.jsonl
  results/runs_constraint_batch_g9/<run>/validation_reference.jsonl
  results/runs_constraint_batch_g9/<run>/source_seeds.jsonl
  results/runs_constraint_batch_g9/<run>/run_metadata.json
plus, for the pre-declared comparison only, the already-committed
G0/G1/G2 clean seed-0 campaigns.

Writes `g9_report.json` beside the run and prints the report tables.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

BUDGET = 25.0
Z_GOOD = 2.0
Z_POOR = 3.0
OWNERS_REQUIRED = 5
ROUNDS_REQUIRED = 4
G2_FSR_WILSON_LO = 0.3530  # docs/g9_gates.md G9e, from the committed G2 run
VAR_RATIO_BAND = (0.8, 1.25)
LAMBDA_SATURATION_MAX = 0.05
KL_BAND_MULTIPLE = 3.0
G2_KL_MEDIAN = 0.00230
G2_KL_P95 = 0.00516

BASELINES = {
    "G0": "results/runs_g0/pilot_A_clean_seed0",
    "G1": "results/runs_constraint_mc_g1/pilot_A_clean_seed0",
    "G2": "results/runs_constraint_mc_g2/pilot_A_clean_seed0",
}


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


class Run:
    """One run's logs, reduced to the per-round arrays the gates need.

    The per-agent reductions deliberately mirror `scripts/analyze_g2.py`'s
    `Seed` class field for field, so that every G9-vs-G2 comparison in the
    report is a comparison of the same quantity computed the same way.
    """

    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.rounds = read_jsonl(run_dir / "rounds.jsonl")
        self.oracle = read_jsonl(run_dir / "oracle.jsonl")
        self.refs = read_jsonl(run_dir / "validation_reference.jsonl")
        self.seeds = read_jsonl(run_dir / "source_seeds.jsonl")
        meta_p = run_dir / "run_metadata.json"
        self.meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        self.aids = sorted(self.rounds[0]["constraints"]) if self.rounds else []
        self.n = min(len(self.rounds), len(self.oracle))
        r, o = self.rounds[: self.n], self.oracle[: self.n]

        def pa(fn):
            return np.array([[fn(rec["constraints"][a]) for a in self.aids] for rec in r], dtype=float)

        def po(key):
            return np.array([[rec["agents"][a][key] for a in self.aids] for rec in o], dtype=float)

        self.true_cost = po("true_cost_return")
        self.task_return = np.array(
            [rec["agents"][self.aids[0]]["episodic_task_return"] for rec in o], dtype=float
        )
        self.violated = po("violated").astype(bool)
        self.mechanism = pa(lambda c: c["mechanism_reported_cost_return"])
        self.residual = pa(lambda c: c["constraint_residual"])
        self.lam = pa(lambda c: c["lambda_after"])
        self.lam_mixed = (
            pa(lambda c: c["lambda_mixed_before"]) if "lambda_mixed_before" in r[0]["constraints"][self.aids[0]] else None
        )
        self.kl = pa(lambda c: c["ppo"]["approx_kl"])
        self.entropy = pa(lambda c: c["ppo"]["entropy"])
        self.spread = pa(lambda c: c["aggregate"]["spread"])
        self.false_safe = (self.mechanism <= BUDGET) & (self.true_cost > BUDGET)
        self.unsafe = self.true_cost > BUDGET
        self.is_batch = "source_batch" in r[0] if r else False

    def fsr(self) -> tuple[float, tuple[float, float], int, int]:
        k, n = int(self.false_safe.sum()), int(self.unsafe.sum())
        return (k / n if n else float("nan"), wilson(k, n), k, n)


# ---------------------------------------------------------------------- gates


def gate_g9a(run: Run, M: int, R_m: int) -> dict:
    problems: list[str] = []
    for rec in run.rounds:
        sb = rec.get("source_batch")
        if sb is None:
            problems.append(f"round {rec['round_k']}: no source_batch block")
            continue
        if sb["M"] != M or sb["R_m"] != R_m:
            problems.append(f"round {rec['round_k']}: M/R_m = {sb['M']}/{sb['R_m']}, expected {M}/{R_m}")
        if sb["n_trajectories"] != M * R_m:
            problems.append(f"round {rec['round_k']}: {sb['n_trajectories']} trajectories")
        if not sb.get("worker_checksums_all_match"):
            problems.append(f"round {rec['round_k']}: worker checksum mismatch")
        for a in run.aids:
            reports = rec["constraints"][a]["reports"]
            if len(reports) != M:
                problems.append(f"round {rec['round_k']} owner {a}: {len(reports)} sources")
            if {x["source_id"] for x in reports} != set(sb["per_owner"][a]["source_means"]):
                problems.append(f"round {rec['round_k']} owner {a}: source ids disagree with the batch")
    return {
        "pass": not problems,
        "n_rounds_checked": len(run.rounds),
        "n_problems": len(problems),
        "problems": problems[:20],
    }


def gate_g9b(run: Run, M: int, R_m: int) -> dict:
    """Sources graded against the withheld R_ref reference under the same
    pinned theta_k. `sigma` is the round's own pooled per-trajectory sd,
    so `SE_m` and `SE_ref` describe THIS policy, never a pooled one."""
    per_round = {}
    for rec in run.refs:
        k = rec["round_k"]
        R_ref = rec["R_ref"]
        owners = {}
        n_owner_pass = 0
        for a in run.aids:
            src_traj = {rid: np.array(rec["source_per_trajectory"][rid][a], float) for rid in sorted(rec["source_per_trajectory"])}
            ref_traj = np.array(rec["reference_per_trajectory"][a], float)
            # Pooled within-source variance across the M x R_m source
            # trajectories AND the reference pool: all are draws under the
            # same theta_k, so pooling them is legitimate here (and only
            # here) and gives the best available sigma for this policy.
            pooled = np.concatenate(list(src_traj.values()) + [ref_traj])
            sigma = float(np.std(pooled, ddof=1))
            se_m = sigma / math.sqrt(R_m)
            se_ref = sigma / math.sqrt(R_ref)
            se_agg = se_m / math.sqrt(M)
            ref_mean = float(np.mean(ref_traj))
            srcs = []
            for rid, vals in src_traj.items():
                mean = float(np.mean(vals))
                bias = mean - ref_mean
                z = bias / math.sqrt(se_m**2 + se_ref**2)
                srcs.append({
                    "source_id": rid, "mean": mean, "bias": bias, "z": z,
                    "label": "good" if abs(z) <= Z_GOOD else ("poor" if abs(z) > Z_POOR else "between"),
                    "ci95": [mean - 1.96 * se_m, mean + 1.96 * se_m],
                })
            agg = float(np.mean([s["mean"] for s in srcs]))
            biases = np.array([s["bias"] for s in srcs])
            all_good = all(s["label"] == "good" for s in srcs)
            n_owner_pass += int(all_good)
            owners[a] = {
                "sigma_hat": sigma, "se_m": se_m, "se_agg": se_agg, "se_ref": se_ref,
                "reference_mean": ref_mean,
                "reference_ci95": [ref_mean - 1.96 * se_ref, ref_mean + 1.96 * se_ref],
                "sources": srcs,
                "aggregate_mean": agg,
                "aggregate_bias": agg - ref_mean,
                "z_agg": (agg - ref_mean) / math.sqrt(se_agg**2 + se_ref**2),
                "bias": float(biases.mean()), "mae": float(np.abs(biases).mean()),
                "rmse": float(np.sqrt((biases**2).mean())),
                "spread_max_minus_min": float(max(s["mean"] for s in srcs) - min(s["mean"] for s in srcs)),
                "pairwise_differences": {
                    f"{i}-{j}": abs(srcs[i]["mean"] - srcs[j]["mean"])
                    for i in range(len(srcs)) for j in range(i + 1, len(srcs))
                },
                "all_good": all_good,
                # FS-reference (descriptive): both sides refer to theta_k.
                "false_safe_vs_reference": bool(agg <= BUDGET and ref_mean > BUDGET),
            }
        per_round[k] = {
            "owners": owners,
            "n_owners_pass": n_owner_pass,
            "of": len(run.aids),
            "pass": n_owner_pass >= OWNERS_REQUIRED,
        }
    n_pass = sum(1 for v in per_round.values() if v["pass"])
    return {
        "pass": n_pass >= ROUNDS_REQUIRED and len(per_round) > 0,
        "n_rounds_pass": n_pass,
        "n_rounds": len(per_round),
        "required": ROUNDS_REQUIRED,
        "per_round": per_round,
    }


def gate_g9c(run: Run) -> dict:
    arrays = {
        "mechanism": run.mechanism, "residual": run.residual, "lambda": run.lam,
        "kl": run.kl, "entropy": run.entropy, "true_cost": run.true_cost, "spread": run.spread,
    }
    nonfinite = {k: int((~np.isfinite(v)).sum()) for k, v in arrays.items()}
    i_ok = sum(nonfinite.values()) == 0

    first50, last50 = run.task_return[:50].mean(), run.task_return[-50:].mean()
    ii_ok = bool(last50 > first50)

    kl_med, kl_p95 = float(np.median(run.kl)), float(np.percentile(run.kl, 95))
    iii_ok = bool(kl_med <= KL_BAND_MULTIPLE * G2_KL_MEDIAN and kl_p95 <= KL_BAND_MULTIPLE * G2_KL_P95)

    sat = float((run.lam >= 25.0 - 1e-6).mean())
    iv_ok = bool(sat < LAMBDA_SATURATION_MAX)

    rel = np.array(
        [rec.get("policy_param_rel_change", float("nan")) for rec in run.rounds], dtype=float
    )
    have_rel = bool(np.isfinite(rel).any())
    checks = [rec.get("source_batch", {}).get("policy_checksum") for rec in run.rounds]
    distinct_frac = len(set(c for c in checks if c)) / max(1, len([c for c in checks if c]))
    v_ok = bool(
        np.isfinite(run.entropy).all()
        and run.entropy.min() > -20.0
        and ((float(np.nanmean(rel)) > 1e-8) if have_rel else distinct_frac == 1.0)
    )

    return {
        "pass": all([i_ok, ii_ok, iii_ok, iv_ok, v_ok]),
        "i_no_nonfinite": {"pass": i_ok, "counts": nonfinite},
        "ii_task_return_improves": {
            "pass": ii_ok, "first50": float(first50), "last50": float(last50),
            "delta": float(last50 - first50),
        },
        "iii_ppo_kl_stable": {
            "pass": iii_ok, "median": kl_med, "p95": kl_p95,
            "g2_median": G2_KL_MEDIAN, "g2_p95": G2_KL_P95,
            "band": [KL_BAND_MULTIPLE * G2_KL_MEDIAN, KL_BAND_MULTIPLE * G2_KL_P95],
            "max": float(run.kl.max()),
        },
        "iv_no_lambda_saturation": {"pass": iv_ok, "frac_at_max": sat, "threshold": LAMBDA_SATURATION_MAX},
        "v_meaningful_updates": {
            "pass": v_ok,
            "entropy_min": float(run.entropy.min()), "entropy_first10": float(run.entropy[:10].mean()),
            "entropy_last10": float(run.entropy[-10:].mean()),
            "policy_param_rel_change_mean": float(np.nanmean(rel)) if have_rel else None,
            "policy_param_rel_change_logged": have_rel,
            "distinct_theta_checksum_fraction": distinct_frac,
        },
    }


def gate_g9d(run: Run, eta: float, lam_max: float) -> dict:
    if run.lam_mixed is None:
        return {"pass": False, "note": "lambda_mixed_before not logged; G9d-i cannot be checked"}
    unclipped = (run.lam > 1e-9) & (run.lam < lam_max - 1e-9)
    err = np.abs((run.lam - run.lam_mixed) - eta * run.residual)
    worst = float(err[unclipped].max()) if unclipped.any() else 0.0
    i_ok = bool(worst <= 1e-9)

    delta = run.lam - run.lam_mixed
    pos = run.residual > 0
    neg = run.residual < 0
    sign_ok = bool(
        (delta[pos & unclipped] > 0).all() if (pos & unclipped).any() else True
    ) and bool((delta[neg & unclipped] < 0).all() if (neg & unclipped).any() else True)

    frac_positive = float((run.lam > 1e-9).mean())
    iii_ok = bool(frac_positive >= 0.10)

    dlam = np.diff(run.lam, axis=0)
    sgn = np.sign(dlam)
    flips = float(np.mean(sgn[1:] * sgn[:-1] < 0))

    lags = {}
    for lag in (0, 1, 5, 10):
        x = run.lam[: run.n - lag]
        y = run.true_cost[lag:]
        vals = []
        for i in range(x.shape[1]):
            if x[:, i].std() > 1e-12 and y[:, i].std() > 1e-12:
                vals.append(float(np.corrcoef(x[:, i], y[:, i])[0, 1]))
        lags[f"lag_{lag}"] = float(np.mean(vals)) if vals else None

    return {
        "pass": i_ok and sign_ok and iii_ok,
        "i_exact_wiring": {"pass": i_ok, "max_abs_error": worst, "n_unclipped": int(unclipped.sum())},
        "ii_sign": {"pass": sign_ok},
        "iii_response": {"pass": iii_ok, "frac_lambda_positive": frac_positive,
                         "lambda_mean": float(run.lam.mean()), "lambda_max": float(run.lam.max())},
        "iv_sign_flip_rate": {"value": flips, "pure_noise_reference": 0.5, "gated": False},
        "v_lambda_vs_true_cost_corr": {"by_lag": lags, "gated": False},
    }


def gate_g9e(run: Run, baselines: dict) -> dict:
    fsr, ci, k, n = run.fsr()
    joint = float(run.false_safe.mean())
    joint_ci = wilson(int(run.false_safe.sum()), run.false_safe.size)

    per_owner = {}
    for i, a in enumerate(run.aids):
        ki, ni = int(run.false_safe[:, i].sum()), int(run.unsafe[:, i].sum())
        per_owner[a] = {
            "conditional_fsr": ki / ni if ni else float("nan"),
            "n_false_safe": ki, "n_unsafe": ni,
            "joint_rate": float(run.false_safe[:, i].mean()),
        }
    ranked = sorted(
        (a for a in run.aids if per_owner[a]["n_unsafe"] > 0),
        key=lambda a: per_owner[a]["conditional_fsr"],
    )

    third = run.n // 3
    phases = {}
    for name, sl in (("early", slice(0, third)), ("middle", slice(third, 2 * third)),
                     ("late", slice(2 * third, run.n))):
        kk, nn = int(run.false_safe[sl].sum()), int(run.unsafe[sl].sum())
        phases[name] = {"conditional_fsr": kk / nn if nn else float("nan"),
                        "n_false_safe": kk, "n_unsafe": nn,
                        "rounds": [sl.start, sl.stop]}

    margin = run.true_cost - BUDGET
    bands = {}
    for name, lo, hi in (("<1", 0.0, 1.0), ("1-3", 1.0, 3.0), ("3-6", 3.0, 6.0), (">6", 6.0, np.inf)):
        sel = run.unsafe & (margin > lo) & (margin <= hi)
        kk, nn = int((run.false_safe & sel).sum()), int(sel.sum())
        bands[name] = {"conditional_fsr": kk / nn if nn else float("nan"), "n": nn, "n_false_safe": kk}

    return {
        "pass": bool(ci[1] < G2_FSR_WILSON_LO),
        "bar": {"g9_wilson_upper_must_be_below": G2_FSR_WILSON_LO, "source": "G2 seed 0 Wilson lower bound"},
        "conditional_fsr": fsr, "wilson_ci": list(ci), "n_false_safe": k, "n_unsafe": n,
        "joint_rate": joint, "joint_wilson_ci": list(joint_ci),
        "per_owner": per_owner,
        "worst_owner": ranked[-1] if ranked else None,
        "best_owner": ranked[0] if ranked else None,
        "by_phase": phases,
        "by_margin_band": bands,
        "baselines": baselines,
    }


def gate_g9f(run: Run, M: int, R_m: int, rng_seed: int = 20250903) -> dict:
    """Between-source variance against the sampling law it should obey."""
    between, expected = [], []
    for rec in run.rounds:
        sb = rec.get("source_batch")
        if not sb:
            continue
        for a in run.aids:
            po = sb["per_owner"][a]
            between.append(po["s2_between"])
            expected.append(po["sigma_hat"] ** 2 / R_m)
    b = np.array(between, float)
    e = np.array(expected, float)
    ratio = float(b.mean() / e.mean())

    rng = np.random.default_rng(rng_seed)
    boot = []
    idx = np.arange(len(b))
    for _ in range(2000):
        s = rng.choice(idx, size=len(idx), replace=True)
        boot.append(b[s].mean() / e[s].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return {
        "pass": bool(VAR_RATIO_BAND[0] <= ratio <= VAR_RATIO_BAND[1]),
        "ratio": ratio, "band": list(VAR_RATIO_BAND),
        "bootstrap_ci95": [float(lo), float(hi)],
        "mean_s2_between": float(b.mean()), "mean_expected": float(e.mean()),
        "n_cells": len(b),
        "observed_se_agg": float(math.sqrt(b.mean() / M)),
        "predicted_se_agg": float(math.sqrt(e.mean() / M)),
        "mean_se_m": float(math.sqrt(e.mean())),
    }


def gate_g9g(run: Run) -> dict:
    all_env: list[int] = []
    all_torch: list[int] = []
    overlaps: list[dict] = []
    for rec in run.seeds:
        per = {rid: {p[0] for p in pairs} for rid, pairs in rec["seeds"].items()}
        rids = sorted(per)
        for i in range(len(rids)):
            for j in range(i + 1, len(rids)):
                ov = per[rids[i]] & per[rids[j]]
                if ov:
                    overlaps.append({"round": rec["round_k"], "a": rids[i], "b": rids[j], "n": len(ov)})
        for pairs in rec["seeds"].values():
            all_env.extend(p[0] for p in pairs)
            all_torch.extend(p[1] for p in pairs)
        for p in rec.get("reference_seeds", []):
            all_env.append(p[0])
            all_torch.append(p[1])
    dup_env = len(all_env) - len(set(all_env))
    dup_torch = len(all_torch) - len(set(all_torch))
    audit = run.meta.get("source_seed_audit", {})
    return {
        "pass": bool(dup_env == 0 and dup_torch == 0 and not overlaps
                     and audit.get("duplicate_seed_events", 0) == 0),
        "n_env_seeds": len(all_env), "n_unique_env_seeds": len(set(all_env)),
        "n_torch_seeds": len(all_torch), "n_unique_torch_seeds": len(set(all_torch)),
        "duplicate_env": dup_env, "duplicate_torch": dup_torch,
        "replica_overlaps": overlaps[:10],
        "run_audit": {k: v for k, v in audit.items() if k != "spawn_keys"},
    }


def gate_g9h(run: Run) -> dict:
    checks = []
    for rec in run.rounds:
        sb = rec.get("source_batch")
        if not sb:
            checks.append({"round": rec["round_k"], "problem": "no source_batch"})
            continue
        if not sb.get("worker_checksums_all_match"):
            checks.append({"round": rec["round_k"], "problem": "worker checksum mismatch"})
        if not sb.get("theta_k_checksum_stable_through_dual"):
            checks.append({"round": rec["round_k"], "problem": "theta moved before the dual update"})
    seen = [rec["source_batch"]["policy_checksum"] for rec in run.rounds if "source_batch" in rec]
    return {
        "pass": not checks,
        "n_rounds": len(seen),
        "n_distinct_theta_checksums": len(set(seen)),
        "problems": checks[:20],
    }


def load_baseline(path: str) -> dict | None:
    p = Path(path)
    if not (p / "rounds.jsonl").exists():
        return None
    b = Run(p)
    fsr, ci, k, n = b.fsr()
    bias = b.mechanism - b.true_cost
    return {
        "dir": str(p), "n_rounds": b.n,
        "task_return_first50": float(b.task_return[:50].mean()),
        "task_return_last50": float(b.task_return[-50:].mean()),
        "true_cost_mean": float(b.true_cost.mean()),
        "true_cost_last50": float(b.true_cost[-50:].mean()),
        "mechanism_mean": float(b.mechanism.mean()),
        "mechanism_bias": float(bias.mean()),
        "mechanism_mae": float(np.abs(bias).mean()),
        "mechanism_rmse": float(np.sqrt((bias**2).mean())),
        "conditional_fsr": fsr, "wilson_ci": list(ci),
        "joint_false_safe_rate": float(b.false_safe.mean()),
        "violation_rate": float(b.violated.mean()),
        "lambda_mean": float(b.lam.mean()), "lambda_max": float(b.lam.max()),
        "kl_median": float(np.median(b.kl)), "kl_p95": float(np.percentile(b.kl, 95)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="results/runs_constraint_batch_g9/g9_batch_clean")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run = Run(run_dir)
    if not run.rounds:
        print(f"No rounds.jsonl under {run_dir}")
        return 1

    cfg = run.meta.get("config_snapshot", {})
    sc = cfg.get("source_collection", {})
    M, R_m = sc.get("M", 3), sc.get("R_m", 30)
    eta = cfg.get("dual", {}).get("eta_lambda", 0.035)
    lam_max = cfg.get("dual", {}).get("lambda_max", 25.0)

    baselines = {k: load_baseline(v) for k, v in BASELINES.items()}
    baselines = {k: v for k, v in baselines.items() if v}

    gates = {
        "G9a": gate_g9a(run, M, R_m),
        "G9b": gate_g9b(run, M, R_m),
        "G9c": gate_g9c(run),
        "G9d": gate_g9d(run, eta, lam_max),
        "G9e": gate_g9e(run, {k: {"conditional_fsr": v["conditional_fsr"],
                                  "wilson_ci": v["wilson_ci"],
                                  "mechanism_bias": v["mechanism_bias"]} for k, v in baselines.items()}),
        "G9f": gate_g9f(run, M, R_m),
        "G9g": gate_g9g(run),
        "G9h": gate_g9h(run),
    }

    bias = run.mechanism - run.true_cost
    g9_summary = {
        "n_rounds": run.n,
        "task_return_first50": float(run.task_return[:50].mean()),
        "task_return_last50": float(run.task_return[-50:].mean()),
        "true_cost_mean": float(run.true_cost.mean()),
        "true_cost_last50": float(run.true_cost[-50:].mean()),
        "mechanism_mean": float(run.mechanism.mean()),
        "mechanism_bias": float(bias.mean()),
        "mechanism_mae": float(np.abs(bias).mean()),
        "mechanism_rmse": float(np.sqrt((bias**2).mean())),
        "conditional_fsr": gates["G9e"]["conditional_fsr"],
        "wilson_ci": gates["G9e"]["wilson_ci"],
        "joint_false_safe_rate": gates["G9e"]["joint_rate"],
        "violation_rate": float(run.violated.mean()),
        "lambda_mean": float(run.lam.mean()), "lambda_max": float(run.lam.max()),
        "kl_median": float(np.median(run.kl)), "kl_p95": float(np.percentile(run.kl, 95)),
        "source_se_m": gates["G9f"]["mean_se_m"],
        "source_se_agg_observed": gates["G9f"]["observed_se_agg"],
        "source_se_agg_predicted": gates["G9f"]["predicted_se_agg"],
    }

    structural = ["G9a", "G9d", "G9g", "G9h"]
    statistical = ["G9b", "G9c", "G9e", "G9f"]
    struct_ok = all(gates[g]["pass"] for g in structural)
    stat_ok = all(gates[g]["pass"] for g in statistical)
    verdict = "PASS" if (struct_ok and stat_ok) else ("CONDITIONAL PASS" if struct_ok else "FAIL")

    report = {
        "run_dir": str(run_dir),
        "git_sha": run.meta.get("git", {}).get("sha"),
        "config": {"M": M, "R_m": R_m, "R_ref": sc.get("R_ref"), "workers": sc.get("workers"),
                   "validation_rounds": sc.get("validation_rounds"), "budget_d": BUDGET,
                   "eta_lambda": eta, "lambda_max": lam_max},
        "cost": {"timing": run.meta.get("timing", {}), "env_steps": run.meta.get("env_steps", {})},
        "summary": g9_summary,
        "baselines": baselines,
        "gates": gates,
        "verdict": verdict,
        "failed_gates": [g for g in gates if not gates[g]["pass"]],
    }
    out = run_dir / "g9_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ------------------------------------------------------------ print
    print(f"\n{'=' * 78}\nG9 REPORT -- {run_dir}\n{'=' * 78}")
    print(f"git sha: {report['git_sha']}   rounds: {run.n}   M={M} R_m={R_m} "
          f"R_ref={sc.get('R_ref')} workers={sc.get('workers')}")

    t, es = report["cost"]["timing"], report["cost"]["env_steps"]
    if t:
        print(f"\n## Cost\n  wall {t.get('wall_clock_s', 0) / 3600:.2f} h  "
              f"({t.get('s_per_round', 0):.1f}s/round; source {t.get('source_s_per_round', 0):.1f}s/round)")
        print(f"  env steps: ppo={es.get('ppo', 0):,}  source={es.get('source', 0):,}  "
              f"oracle={es.get('oracle', 0):,}  source:ppo = {es.get('source', 0) / max(1, es.get('ppo', 1)):.0f}:1")

    print("\n## Gate results")
    for g, v in gates.items():
        print(f"  {g}: {'PASS' if v['pass'] else 'FAIL'}")

    print("\n## G9b -- source calibration at the validation rounds")
    for k, v in sorted(gates["G9b"]["per_round"].items()):
        print(f"\n  round {k}: {v['n_owners_pass']}/{v['of']} owners all-good "
              f"-> {'PASS' if v['pass'] else 'FAIL'}")
        print(f"    {'owner':8} {'sigma':>6} {'SE_m':>6} {'SE_agg':>7} {'SE_ref':>7} {'ref':>8} "
              f"{'agg':>8} {'z_agg':>7}  z_1..z_M")
        for a, o in v["owners"].items():
            zs = " ".join(f"{s['z']:+.2f}" for s in o["sources"])
            print(f"    {a:8} {o['sigma_hat']:6.2f} {o['se_m']:6.3f} {o['se_agg']:7.3f} "
                  f"{o['se_ref']:7.3f} {o['reference_mean']:8.3f} {o['aggregate_mean']:8.3f} "
                  f"{o['z_agg']:+7.2f}  {zs}")

    print("\n## G9e -- clean false-safe rate")
    e = gates["G9e"]
    print(f"  G9   conditional P(mech<=d | true>d) = {e['conditional_fsr']:.4f} "
          f"[{e['wilson_ci'][0]:.4f}, {e['wilson_ci'][1]:.4f}]  ({e['n_false_safe']}/{e['n_unsafe']})")
    for name, b in e["baselines"].items():
        print(f"  {name}   conditional FSR = {b['conditional_fsr']:.4f} "
              f"[{b['wilson_ci'][0]:.4f}, {b['wilson_ci'][1]:.4f}]   mech bias {b['mechanism_bias']:+.2f}")
    print(f"  bar: G9 Wilson upper < {G2_FSR_WILSON_LO} -> "
          f"{'PASS' if e['pass'] else 'FAIL'} (upper = {e['wilson_ci'][1]:.4f})")
    print(f"  worst owner: {e['worst_owner']}   best owner: {e['best_owner']}")
    print("  by phase:   " + "  ".join(
        f"{n}={v['conditional_fsr']:.3f}({v['n_false_safe']}/{v['n_unsafe']})"
        for n, v in e["by_phase"].items()))
    print("  by margin:  " + "  ".join(
        f"{n}={v['conditional_fsr']:.3f}(n={v['n']})" for n, v in e["by_margin_band"].items()))

    print("\n## G9f -- aggregate precision")
    f = gates["G9f"]
    print(f"  var ratio (observed between / sampling law) = {f['ratio']:.4f} "
          f"[{f['bootstrap_ci95'][0]:.4f}, {f['bootstrap_ci95'][1]:.4f}]  band {f['band']}")
    print(f"  SE_m = {f['mean_se_m']:.4f}   SE_agg observed = {f['observed_se_agg']:.4f}  "
          f"predicted = {f['predicted_se_agg']:.4f}")

    print("\n## G9c -- training health")
    c = gates["G9c"]
    print(f"  task return  {c['ii_task_return_improves']['first50']:.1f} -> "
          f"{c['ii_task_return_improves']['last50']:.1f}  "
          f"({'PASS' if c['ii_task_return_improves']['pass'] else 'FAIL'})")
    print(f"  approx_kl    median {c['iii_ppo_kl_stable']['median']:.5f} (G2 {G2_KL_MEDIAN}) "
          f"p95 {c['iii_ppo_kl_stable']['p95']:.5f} (G2 {G2_KL_P95})  "
          f"({'PASS' if c['iii_ppo_kl_stable']['pass'] else 'FAIL'})")
    print(f"  lambda@max   {c['iv_no_lambda_saturation']['frac_at_max']:.4f}  "
          f"({'PASS' if c['iv_no_lambda_saturation']['pass'] else 'FAIL'})")
    print(f"  entropy      {c['v_meaningful_updates']['entropy_first10']:.3f} -> "
          f"{c['v_meaningful_updates']['entropy_last10']:.3f}")
    print(f"  non-finite   {c['i_no_nonfinite']['counts']}")

    print("\n## G9d -- dual dynamics")
    d = gates["G9d"]
    print(f"  wiring identity max error {d['i_exact_wiring']['max_abs_error']:.2e} over "
          f"{d['i_exact_wiring']['n_unclipped']} unclipped cells")
    print(f"  lambda: mean {d['iii_response']['lambda_mean']:.4f}  max "
          f"{d['iii_response']['lambda_max']:.4f}  frac>0 {d['iii_response']['frac_lambda_positive']:.3f}")
    print(f"  dlambda sign-flip rate {d['iv_sign_flip_rate']['value']:.3f} (pure noise 0.5)")
    print(f"  corr(lambda, true cost) by lag: {d['v_lambda_vs_true_cost_corr']['by_lag']}")

    print("\n## Comparison table")
    hdr = f"  {'metric':28}" + "".join(f"{k:>12}" for k in list(baselines) + ["G9"])
    print(hdr)
    rows = [
        ("task return (last 50)", "task_return_last50", "{:12.1f}"),
        ("true cost (mean)", "true_cost_mean", "{:12.3f}"),
        ("true cost (last 50)", "true_cost_last50", "{:12.3f}"),
        ("mechanism estimate bias", "mechanism_bias", "{:+12.3f}"),
        ("mechanism MAE", "mechanism_mae", "{:12.3f}"),
        ("false-safe (conditional)", "conditional_fsr", "{:12.4f}"),
        ("false-safe (joint)", "joint_false_safe_rate", "{:12.4f}"),
        ("violation rate", "violation_rate", "{:12.4f}"),
        ("mean lambda", "lambda_mean", "{:12.4f}"),
        ("lambda max", "lambda_max", "{:12.4f}"),
    ]
    for label, key, fmt in rows:
        line = f"  {label:28}"
        for k in list(baselines):
            line += fmt.format(baselines[k][key])
        line += fmt.format(g9_summary[key])
        print(line)
    line = f"  {'source SE (aggregate)':28}" + "".join(f"{'--':>12}" for _ in baselines)
    print(line + f"{g9_summary['source_se_agg_observed']:12.4f}")

    print(f"\n{'=' * 78}\nVERDICT: {verdict}")
    if report["failed_gates"]:
        print(f"failed gates: {report['failed_gates']}")
    print(f"{'=' * 78}\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
