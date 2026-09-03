#!/usr/bin/env python
"""G7 (docs/g7_gates.md): estimand audit and multi-source test for the
SAME policy-level J_C^i.

Reads the three fixed pools `scripts/g7_collect_estimand_dataset.py`
already wrote (`results/g7_estimand_source_diagnostic/dataset/*.npz`).
Trains no PPO, no GAE, no dual update, no environment call, no attack, no
RCE, and -- unlike every prior G-series script -- no neural network at
all: every source here is `numpy.mean` over a batch of
`mc_cost_return` values (task section 11). This script is pure arithmetic
on the already-collected `G_r^i` scalars.

Computes, per owner:
  S1  -- pooled mean over all 90 eval_pool_a rounds
  S2  -- M in {3,5} disjoint contiguous blocks of eval_pool_a
  S3  -- M in {3,5} bootstrap resamples (with replacement) of eval_pool_a
  S4  -- summation-order control on S2-M3's block 0 (analytic sanity check)
  reference -- pooled mean over 270 disjoint reference_pool rounds
  eval_pool_b reproduction of S2-M3, scored against the SAME reference

and the closed-form (not just empirical) cross-source covariance/
correlation implied by the sampling design, per docs/g7_gates.md
"Error-dependence / independence protocol".

Usage:
    .venv/Scripts/python.exe scripts/g7_estimand_source_diagnostic.py
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

# --- pre-declared constants, docs/g7_gates.md ---
BUDGET_D = 25.0
Z_GOOD = 2.0
Z_POOR = 3.0
N_POOL = 90  # eval_pool_a size, used by the bootstrap closed-form formula
R_REF = 270
BOOTSTRAP_SEEDS = {3: 90003, 5: 90005}
M_CONDITIONS = [1, 3, 5]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_pools(dataset_dir: Path) -> dict:
    pools = {}
    for name in ["eval_pool_a", "reference_pool", "eval_pool_b"]:
        d = np.load(dataset_dir / f"{name}.npz", allow_pickle=True)
        pools[name] = {
            "mc_cost_return": d["mc_cost_return"],  # (n_rounds, n_agents)
            "round_seeds": d["round_seeds"],
            "agent_ids": [str(a) for a in d["agent_ids"]],
        }
    meta = json.loads((dataset_dir / "meta.json").read_text())
    return pools, meta


# --------------------------------------------------------------------------
# Batch constructions (S1/S2/S3/S4)
# --------------------------------------------------------------------------

def disjoint_blocks(n_rows: int, m: int) -> list[np.ndarray]:
    return [np.asarray(b) for b in np.array_split(np.arange(n_rows), m)]


def bootstrap_blocks(n_rows: int, m: int, block_size: int, seed_base: int) -> list[np.ndarray]:
    blocks = []
    for i in range(m):
        rng = np.random.default_rng(seed_base + i)
        blocks.append(rng.integers(0, n_rows, size=block_size))
    return blocks


def batch_mean_se(values: np.ndarray, idx: np.ndarray) -> tuple[float, float, int]:
    sample = values[idx]
    r = len(sample)
    mean = float(np.mean(sample))
    # sample std, ddof=1 -- undefined for r=1, guarded (not hit in practice
    # since every G7 batch has r>=18).
    sd = float(np.std(sample, ddof=1)) if r > 1 else float("nan")
    se = sd / np.sqrt(r) if r > 1 else float("nan")
    return mean, se, r


# --------------------------------------------------------------------------
# Per-owner analysis
# --------------------------------------------------------------------------

def analyze_owner(
    aid: str,
    agent_idx: int,
    eval_a: np.ndarray,
    reference: np.ndarray,
    eval_b: np.ndarray,
) -> dict:
    col_a = eval_a[:, agent_idx]
    col_ref = reference[:, agent_idx]
    col_b = eval_b[:, agent_idx]
    n_a = len(col_a)

    # sigma_hat: pooled sample std over ALL 450 collected rounds for this
    # owner (docs/g7_gates.md "Reference construction" -- a stability
    # choice for the closed-form variance formulas, not used to compute
    # any individual source's OWN mean).
    pooled_all = np.concatenate([col_a, col_ref, col_b])
    sigma_hat = float(np.std(pooled_all, ddof=1))

    ref_mean, ref_se, ref_r = batch_mean_se(col_ref, np.arange(len(col_ref)))
    assert ref_r == R_REF

    def score(mean: float, se: float) -> dict:
        z = (mean - ref_mean) / np.sqrt(se**2 + ref_se**2)
        if abs(z) <= Z_GOOD:
            label = "good"
        elif abs(z) > Z_POOR:
            label = "poor"
        else:
            label = "mixed"
        return {
            "mean": mean,
            "se": se,
            "bias_vs_reference": mean - ref_mean,
            "abs_bias_frac_of_budget": abs(mean - ref_mean) / BUDGET_D,
            "z": z,
            "label": label,
            "ci95": [mean - 1.96 * se, mean + 1.96 * se],
        }

    result: dict = {
        "agent_id": aid,
        "sigma_hat": sigma_hat,
        "reference": {"mean": ref_mean, "se": ref_se, "r": ref_r, "ci95": [ref_mean - 1.96 * ref_se, ref_mean + 1.96 * ref_se]},
    }

    # S1: pooled mean, all 90 eval_pool_a rounds (== M=1 condition)
    s1_mean, s1_se, s1_r = batch_mean_se(col_a, np.arange(n_a))
    result["S1"] = {"r": s1_r, **score(s1_mean, s1_se)}

    # S2 / S3 for M in {3, 5}
    result["S2"] = {}
    result["S3"] = {}
    for m in [3, 5]:
        blocks = disjoint_blocks(n_a, m)
        s2_sources = []
        for b in blocks:
            mean, se, r = batch_mean_se(col_a, b)
            s2_sources.append({"r": r, **score(mean, se)})
        result["S2"][str(m)] = s2_sources

        block_size = n_a // m
        boot_idx = bootstrap_blocks(n_a, m, block_size, BOOTSTRAP_SEEDS[m])
        s3_sources = []
        for b in boot_idx:
            mean, se, r = batch_mean_se(col_a, b)
            s3_sources.append({"r": r, **score(mean, se)})
        result["S3"][str(m)] = s3_sources

    # S4: summation-order control on S2-M3's block 0 -- the primary
    # estimator has no stochastic-fit component, so this checks (not
    # assumes) that reordering leaves the mean unchanged to float
    # precision.
    block0 = disjoint_blocks(n_a, 3)[0]
    base_mean = float(np.mean(col_a[block0]))
    reorder_means = []
    for seed in [1, 2, 3]:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(block0)
        reorder_means.append(float(np.mean(col_a[perm])))
    max_rel_diff = max(abs(m - base_mean) / max(abs(base_mean), 1e-12) for m in reorder_means)
    result["S4"] = {
        "base_mean": base_mean,
        "reorder_means": reorder_means,
        "max_rel_diff": max_rel_diff,
        "degenerate_as_expected": bool(max_rel_diff < 1e-9),
    }

    # --- Closed-form (theoretical) cross-source covariance/correlation ---
    result["theoretical"] = {}
    for m in [3, 5]:
        r_m = n_a // m
        se_m = sigma_hat / np.sqrt(r_m)
        # S2: disjoint blocks share NO random draw -> covariance 0 exactly.
        corr_s2_raw = 0.0
        # S2 errors (vs reference): share only the reference's own noise.
        var_e_s2 = se_m**2 + ref_se**2
        corr_s2_err = (ref_se**2) / var_e_s2 if var_e_s2 > 0 else float("nan")
        # S3: bootstrap replicas share the pool's own mean (law of total
        # covariance over the fixed 90-row pool).
        var_pool_mean = sigma_hat**2 / N_POOL
        var_boot = var_pool_mean + sigma_hat**2 / r_m
        corr_s3_raw = var_pool_mean / var_boot if var_boot > 0 else float("nan")
        var_e_s3 = var_pool_mean + sigma_hat**2 / r_m + ref_se**2
        corr_s3_err = (var_pool_mean + ref_se**2) / var_e_s3 if var_e_s3 > 0 else float("nan")

        # participation ratio of the theoretical M x M covariance matrix
        def participation_ratio(diag_var: float, off_diag_cov: float, m_: int) -> float:
            cov = np.full((m_, m_), off_diag_cov)
            np.fill_diagonal(cov, diag_var)
            eigs = np.linalg.eigvalsh(cov)
            eigs = np.clip(eigs, 0, None)
            s1_ = eigs.sum()
            s2_ = (eigs**2).sum()
            return float((s1_**2) / s2_) if s2_ > 0 else float("nan")

        pr_s2 = participation_ratio(se_m**2, 0.0, m)
        pr_s3 = participation_ratio(var_boot, var_pool_mean, m)

        result["theoretical"][str(m)] = {
            "r_m": r_m,
            "se_m": se_m,
            "corr_s2_raw_theoretical": corr_s2_raw,
            "corr_s3_raw_theoretical": corr_s3_raw,
            "corr_s2_error_theoretical": corr_s2_err,
            "corr_s3_error_theoretical": corr_s3_err,
            "M_eff_s2_theoretical": pr_s2,
            "M_eff_s3_theoretical": pr_s3,
        }

    # --- eval_pool_b reproduction of S2-M3, scored against SAME reference ---
    blocks_b = disjoint_blocks(len(col_b), 3)
    s2b_sources = []
    for b in blocks_b:
        mean, se, r = batch_mean_se(col_b, b)
        s2b_sources.append({"r": r, **score(mean, se)})
    result["S2_eval_pool_b_M3"] = s2b_sources

    return result


def pooled_cross_owner_empirical(owner_results: list[dict], cond: str, m: int) -> dict:
    """Empirical cross-SOURCE correlation, pooled across the 6 OWNERS.

    A single owner gives only `m` (3 or 5) source-mean scalars -- no
    repeated axis exists to correlate them against, so a per-owner
    empirical Pearson `r` is not computable at all (not merely unstable).
    But `S2`'s round-partition is built once, by round INDEX, and applied
    identically to every owner's column (docs/g7_gates.md: "one round
    produces `G_r^i` for all six owners AT ONCE") -- so source `m`'s batch
    is the SAME 30 (or 18) physical rollouts for every owner, just a
    different cost column. That gives a genuine, if small (n=6 owners),
    paired sample: source `a`'s value vs. source `b`'s value, one pair per
    owner. This mirrors G4/G5's own "pooled across owners" convention
    (`docs/g5_gates.md` "Diversity measures": "concatenating each
    replica's ... error vector"), and per that same convention this MUST
    be computed on `bias_vs_reference` (= error, `Ĵ_{C,m}^i -
    Ĵ_reference^i`), never on the raw `mean`. Owners here have a large,
    real, systematic difference in baseline cost level (measured:
    `agent_0` reference ~19.8 vs `agent_5` reference ~34.1, a ~14-point
    fixed effect vs. a ~0.3-2 point per-source standard error) -- pooling
    raw means across owners would let that shared owner-level location
    effect dominate the correlation and read out near-1.0 regardless of
    genuine source independence, an artifact of NOT removing the owner
    fixed effect rather than a diversity measurement. Subtracting each
    owner's own reference removes exactly that fixed effect and isolates
    the within-owner sampling noise the independence question is actually
    about. Reported as a secondary empirical check on the closed-form
    theoretical correlation above -- per docs/g7_gates.md section 13,
    never as the primary evidence, and n=6 is still small enough that
    this is a directional check, not a precise estimate."""
    errors = np.array([[s["bias_vs_reference"] for s in res[cond][str(m)]] for res in owner_results])  # (6 owners, m sources)
    cov = np.cov(errors, rowvar=False)  # (m, m), ddof=1 by default
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(errors, rowvar=False)
    off_diag = corr[~np.eye(m, dtype=bool)]
    eigs = np.clip(np.linalg.eigvalsh(cov), 0, None)
    pr = float((eigs.sum() ** 2) / (eigs**2).sum()) if (eigs**2).sum() > 0 else float("nan")
    return {
        "n_owners": errors.shape[0],
        "correlation_matrix": corr.tolist(),
        "mean_off_diagonal_corr": float(np.nanmean(off_diag)),
        "max_off_diagonal_corr": float(np.nanmax(off_diag)),
        "participation_ratio_empirical": pr,
    }


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------

def compute_gates(owner_results: list[dict]) -> dict:
    gates: dict = {}

    # G7b: per (condition, M), pass iff ALL M sources in EVERY owner are "good"
    g7b = {}
    for m in [3, 5]:
        for cond in ["S2", "S3"]:
            owners_pass = []
            for res in owner_results:
                sources = res[cond][str(m)]
                all_good = all(s["label"] == "good" for s in sources)
                owners_pass.append({"agent_id": res["agent_id"], "all_good": all_good,
                                     "labels": [s["label"] for s in sources]})
            n_owner_pass = sum(1 for o in owners_pass if o["all_good"])
            g7b[f"{cond}_M{m}"] = {"owners": owners_pass, "n_owners_all_good": n_owner_pass, "of": len(owner_results)}
    gates["G7b"] = g7b

    # G7c: per owner, per M: S2 theoretical corr materially < S3 theoretical corr (>=0.10 absolute)
    g7c = {}
    for m in [3, 5]:
        rows = []
        for res in owner_results:
            th = res["theoretical"][str(m)]
            gap = th["corr_s3_raw_theoretical"] - th["corr_s2_raw_theoretical"]
            rows.append({"agent_id": res["agent_id"], "gap": gap, "pass": bool(gap >= 0.10)})
        g7c[f"M{m}"] = {"rows": rows, "all_pass": all(r["pass"] for r in rows)}
    gates["G7c"] = g7c

    # G7d: M=3 disjoint (S2), >=5/6 owners all-good, AND G7c passes
    g7d_owner_pass = gates["G7b"]["S2_M3"]["n_owners_all_good"]
    g7d_pass = (g7d_owner_pass >= 5) and gates["G7c"]["M3"]["all_pass"]
    gates["G7d"] = {"n_owners_all_good_S2_M3": g7d_owner_pass, "of": len(owner_results),
                     "g7c_M3_all_pass": gates["G7c"]["M3"]["all_pass"], "pass": bool(g7d_pass)}

    # G7e: owner coverage -- structural, always true if this script produced per-owner rows
    gates["G7e"] = {"pass": True, "note": "every table above is per-owner; enforced by construction of owner_results"}

    # G7f: eval_pool_b reproduces S2-M3's all-good finding, per owner
    g7f_rows = []
    for res in owner_results:
        sources = res["S2_eval_pool_b_M3"]
        all_good = all(s["label"] == "good" for s in sources)
        g7f_rows.append({"agent_id": res["agent_id"], "all_good": all_good, "labels": [s["label"] for s in sources]})
    n_repro = sum(1 for r in g7f_rows if r["all_good"])
    gates["G7f"] = {"rows": g7f_rows, "n_owners_all_good": n_repro, "of": len(owner_results),
                     "pass": bool(n_repro >= 5)}

    return gates


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default="results/g7_estimand_source_diagnostic/dataset")
    ap.add_argument("--out", default="results/g7_estimand_source_diagnostic/g7_report.json")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    pools, meta = load_pools(dataset_dir)
    agent_ids = pools["eval_pool_a"]["agent_ids"]

    # G7a re-check at analysis time too (defense in depth: collection
    # already asserted this, but the analysis should not silently trust a
    # stale/hand-edited meta.json).
    seeds_a = set(pools["eval_pool_a"]["round_seeds"].tolist())
    seeds_ref = set(pools["reference_pool"]["round_seeds"].tolist())
    seeds_b = set(pools["eval_pool_b"]["round_seeds"].tolist())
    disjoint_ok = not (seeds_a & seeds_ref) and not (seeds_a & seeds_b) and not (seeds_ref & seeds_b)
    print(f"G7a (analysis-time re-check): round-seed pools pairwise disjoint = {disjoint_ok}")
    if not disjoint_ok:
        print("G7a FAILED at analysis time -- aborting.")
        return 1

    owner_results = []
    for i, aid in enumerate(agent_ids):
        res = analyze_owner(
            aid, i,
            pools["eval_pool_a"]["mc_cost_return"],
            pools["reference_pool"]["mc_cost_return"],
            pools["eval_pool_b"]["mc_cost_return"],
        )
        owner_results.append(res)
        print(f"{aid}: sigma_hat={res['sigma_hat']:.3f}, reference={res['reference']['mean']:.3f} "
              f"+/- {res['reference']['se']:.3f} (r={res['reference']['r']})")
        print(f"  S1 (M=1,R=90):  mean={res['S1']['mean']:.3f} z={res['S1']['z']:.2f} [{res['S1']['label']}]")
        for m in [3, 5]:
            labels2 = [s["label"] for s in res["S2"][str(m)]]
            labels3 = [s["label"] for s in res["S3"][str(m)]]
            print(f"  S2 M={m}: labels={labels2}  |  S3 M={m}: labels={labels3}")
            th = res["theoretical"][str(m)]
            print(f"    theoretical corr: S2={th['corr_s2_raw_theoretical']:.3f} "
                  f"S3={th['corr_s3_raw_theoretical']:.3f}  (M_eff S2={th['M_eff_s2_theoretical']:.2f}, "
                  f"S3={th['M_eff_s3_theoretical']:.2f})")
        print(f"  S4 degenerate_as_expected={res['S4']['degenerate_as_expected']} "
              f"(max_rel_diff={res['S4']['max_rel_diff']:.2e})")

    gates = compute_gates(owner_results)
    print("\n--- GATES ---")
    print(json.dumps(gates, indent=2, default=str)[:4000])

    pooled_empirical = {}
    for cond in ["S2", "S3"]:
        pooled_empirical[cond] = {}
        for m in [3, 5]:
            pe = pooled_cross_owner_empirical(owner_results, cond, m)
            pooled_empirical[cond][str(m)] = pe
            print(f"\n[pooled cross-owner empirical, n={pe['n_owners']}] {cond} M={m}: "
                  f"mean_off_diag_corr={pe['mean_off_diagonal_corr']:.3f} "
                  f"max={pe['max_off_diagonal_corr']:.3f} PR_empirical={pe['participation_ratio_empirical']:.2f}")

    report = {
        "meta": meta,
        "config": {
            "z_good": Z_GOOD, "z_poor": Z_POOR, "n_pool": N_POOL, "r_ref": R_REF,
            "bootstrap_seeds": BOOTSTRAP_SEEDS, "budget_d": BUDGET_D,
        },
        "owners": owner_results,
        "gates": gates,
        "pooled_cross_owner_empirical": pooled_empirical,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
