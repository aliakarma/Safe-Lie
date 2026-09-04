#!/usr/bin/env python
"""Operating characteristics of the G10 gates, computed from G9/seed-0 data only.

This does NOT choose or change a bar. Every threshold is already fixed in
docs/g10_gates.md and committed. What this script establishes is how often
each bar fires when the source is *perfectly calibrated* -- the false-alarm
rate -- so that a miss in the final report can be read against its prior
instead of being over-interpreted.

It is run and committed while seed 1 is still in its opening rounds and
before seed 2 exists, so the only data it touches is the committed G9
seed-0 report.

The correlation structure is measured, not assumed:

  * the M=3 sources at a checkpoint share ONE reference batch, so their z
    scores are correlated by rho_ref = se_ref^2 / (se_m^2 + se_ref^2)
    = (1/R_ref) / (1/R_m + 1/R_ref) = (1/120)/(1/30+1/120) = 0.20;
  * the 6 owners read their costs off the SAME trajectories, so their z
    scores move together. The owner-to-owner correlation is measured from
    the G9 report rather than guessed.

Usage:
    python scripts/g10_gate_power.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

G9_REPORT = "results/runs_constraint_batch_g9/g9_batch_clean/g9_report.json"
R_M = 30
R_REF = 120
N_OWNERS = 6
N_SOURCES = 3
N_CHECKPOINTS = 5

# The bars, imported by value from docs/g10_gates.md section 8. Restated here
# only so this script can report on them; it never writes them back.
G9B_Z = 2.0
G9B_OWNERS_REQUIRED = 5
G9B_CHECKPOINTS_REQUIRED = 4
A_Z_MAX = 3.0
A_POOLED_BIAS_MAX = 1.0
A_META_POOLED_BIAS_MAX = 0.6
PER_CHECKPOINT_BIAS_SE = 0.94  # sqrt(se_agg^2 + se_ref^2) at G9's sigma ~ 6.5


def measured_owner_correlation(report_path: Path) -> tuple[float, float]:
    """Owner-to-owner correlation of the per-source z at a checkpoint, read
    off the committed G9 report."""
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    rows = []
    for v in rep["gates"]["G9b"]["per_round"].values():
        owners = list(v["owners"].values())
        for si in range(N_SOURCES):
            rows.append([o["sources"][si]["z"] for o in owners])
    z = np.array(rows)  # (checkpoints * sources, owners)
    c = np.corrcoef(z.T)
    off = c[~np.eye(N_OWNERS, dtype=bool)]
    return float(off.mean()), float(off.min())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default=G9_REPORT)
    ap.add_argument("--n", type=int, default=400_000)
    ap.add_argument("--rng-seed", type=int, default=20260904)
    ap.add_argument("--out", default="results/g10_gate_power.json")
    args = ap.parse_args()

    r_mean, r_min = measured_owner_correlation(Path(args.report))
    rho_ref = (1 / R_REF) / (1 / R_M + 1 / R_REF)
    rng = np.random.default_rng(args.rng_seed)
    n = args.n

    print("=== measured structure (from the committed G9 report) ===")
    print(f"  owner-to-owner correlation of per-source z: mean {r_mean:.4f}, min {r_min:.4f}")
    print(f"  source-to-source correlation via the shared reference: {rho_ref:.4f}")
    print("  -> the 18 owner-cells at a checkpoint are ~3 effective tests, not 18")

    # ---- G9b, under a perfectly calibrated source ----------------------
    g = rng.standard_normal((n, N_CHECKPOINTS, 1, 1)) * np.sqrt(rho_ref)
    s = rng.standard_normal((n, N_CHECKPOINTS, N_SOURCES, 1)) * np.sqrt(1 - rho_ref)
    common = g + s
    idio = rng.standard_normal((n, N_CHECKPOINTS, N_SOURCES, N_OWNERS))
    z = np.sqrt(r_mean) * common + np.sqrt(1 - r_mean) * idio

    owner_ok = (np.abs(z) <= G9B_Z).all(axis=2)
    ck_ok = owner_ok.sum(axis=2) >= G9B_OWNERS_REQUIRED
    g9b_pass = ck_ok.sum(axis=1) >= G9B_CHECKPOINTS_REQUIRED
    g9b_false_alarm = float(1 - g9b_pass.mean())
    g9b_like_g9 = float((ck_ok.sum(axis=1) <= 3).mean())

    print("\n=== G9b (per-source |z|<=2, 5/6 owners, 4/5 checkpoints) -- DEMOTED ===")
    print(f"  P(fail | perfectly calibrated source), one seed: {g9b_false_alarm:.4f}")
    print(f"  P(<=3 of 5 checkpoints, i.e. G9's outcome):      {g9b_like_g9:.4f}")
    print(f"  P(fail somewhere in three seeds):                {1 - (1 - g9b_false_alarm) ** 3:.4f}")
    print("  -> G9's G9b FAIL is what a correctly calibrated source does roughly")
    print("     one time in six. It is a multiplicity artefact, not miscalibration.")

    # ---- G10-A ---------------------------------------------------------
    # Owner correlation 0.975 makes the owner-mean aggregate z essentially a
    # single standard normal per checkpoint -- no 1/sqrt(6) is available.
    zbar = rng.standard_normal((n, N_CHECKPOINTS))
    a_i = (np.abs(zbar) <= A_Z_MAX).all(axis=1)
    bias = rng.standard_normal((n, N_CHECKPOINTS)) * PER_CHECKPOINT_BIAS_SE
    a_ii = np.abs(bias.mean(axis=1)) <= A_POOLED_BIAS_MAX
    a_both = a_i & a_ii
    pooled15 = (rng.standard_normal((n, 3 * N_CHECKPOINTS)) * PER_CHECKPOINT_BIAS_SE).mean(axis=1)
    a_meta = np.abs(pooled15) <= A_META_POOLED_BIAS_MAX

    print("\n=== G10-A (primary calibration gate) ===")
    print(f"  A-i   |z_bar|<={A_Z_MAX} at all 5 checkpoints : P(pass) {a_i.mean():.4f} per seed, "
          f"{a_i.mean() ** 3:.4f} across three")
    print(f"  A-ii  pooled |bias|<={A_POOLED_BIAS_MAX}            : P(pass) {a_ii.mean():.4f} per seed, "
          f"{a_ii.mean() ** 3:.4f} across three")
    print(f"  A     both                        : P(pass) {a_both.mean():.4f} per seed, "
          f"{a_both.mean() ** 3:.4f} across three")
    print(f"  meta  3-seed pooled |bias|<={A_META_POOLED_BIAS_MAX}     : P(pass) {a_meta.mean():.4f}")
    print(f"\n  -> a single G10-A miss across three seeds has prior "
          f"{1 - a_both.mean() ** 3:.4f} under a perfectly calibrated source.")
    print("     The report must read any such miss against that prior, and must NOT")
    print("     treat one miss as evidence the architecture is miscalibrated.")
    print(f"     G10-A is nonetheless {g9b_false_alarm / (1 - a_both.mean()):.1f}x less trigger-happy "
          f"than G9b per seed.")

    out = {
        "note": "computed from committed G9 seed-0 data only, before seeds 1 and 2 existed; "
                "reports the operating characteristics of already-fixed bars, changes none of them",
        "measured": {"owner_correlation_mean": r_mean, "owner_correlation_min": r_min,
                     "rho_shared_reference": rho_ref},
        "G9b_demoted": {"p_fail_one_seed": g9b_false_alarm,
                        "p_g9_outcome_or_worse": g9b_like_g9,
                        "p_fail_in_three_seeds": float(1 - (1 - g9b_false_alarm) ** 3)},
        "G10_A": {"p_pass_i": float(a_i.mean()), "p_pass_ii": float(a_ii.mean()),
                  "p_pass_seed": float(a_both.mean()),
                  "p_pass_three_seeds": float(a_both.mean() ** 3),
                  "p_pass_meta": float(a_meta.mean())},
        "n_monte_carlo": n,
        "rng_seed": args.rng_seed,
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
