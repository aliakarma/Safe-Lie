#!/usr/bin/env python
"""G10 analysis: is the G9 result reproducible across independent training seeds?

Every threshold this script applies was fixed in `docs/g10_gates.md` and
committed before seed 1 or seed 2 existed. Nothing here chooses a bar; it
only evaluates the ones already chosen.

Two demoted-but-still-computed gates are imported verbatim from
`scripts/analyze_g9.py` rather than reimplemented, so that "G9b and the
conditional false-safe rate are still reported for every seed" is a fact
about the code and not a claim in prose:

    gate_g9b  -- per-source |z| <= 2 calibration      (secondary, see g10_gates.md 4.2)
    gate_g9e  -- conditional / joint false-safe rate  (conditional part secondary, 4.1)
    gate_g9f  -- between-source variance sampling law (primary as G10-B-i)

Reads, per seed:
    <run>/rounds.jsonl  <run>/oracle.jsonl  <run>/validation_reference.jsonl
    <run>/source_seeds.jsonl  <run>/run_metadata.json

Writes `g10_report.json` (three-seed) and prints the report tables.

Usage:
    python scripts/analyze_g10.py
    python scripts/analyze_g10.py --seed0-only      # validate against the G9 numbers
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from analyze_g9 import Run, gate_g9b, gate_g9d, gate_g9e, gate_g9f, wilson  # noqa: E402

from safelie.consensus.topologies import build_topology  # noqa: E402

# ------------------------------------------------------------------ bars
# All of these are docs/g10_gates.md section 8, fixed before seeds 1 and 2 ran.
BUDGET = 25.0
ETA_LAMBDA = 0.035
LAMBDA_MAX = 25.0

A_Z_MAX = 3.0                 # G10-A-i   family-wise ~4% over 15 checkpoints
A_POOLED_BIAS_MAX = 1.0       # G10-A-ii  4% of d
A_META_POOLED_BIAS_MAX = 0.6  # G10-A meta, 3 seeds pooled
B_VAR_RATIO_BAND = (0.80, 1.25)   # G10-B-i, G9f's band verbatim
C_RETURN_DELTA_MIN = 75.0     # G10-C-ii
C_KL_MEDIAN_MAX = 0.0069      # G10-C-iii, 3x the committed G2 clean reference
C_KL_P95_MAX = 0.01548
C_LAMBDA_SATURATION_MAX = 0.05    # G10-C-iv
D_FRAC_LAMBDA_POSITIVE_MIN = 0.10  # G10-D-iii
D_WIRING_TOL = 1e-9               # G10-S9
E_JOINT_WILSON_UPPER_MAX = 0.2433  # G10-E-i, G1's joint rate (best pre-G9 campaign)
F_LAST50_NET_AVG_MAX = 28.0        # G10-F-i, d + 3
F_DIVERGENCE_MAX = 3.0             # G10-F-ii
EXPECTED_ROUNDS = 250
EXPECTED_M = 3
EXPECTED_R_M = 30
EXPECTED_ROLLOUT = 2000

SEED_DIRS = {
    0: "results/runs_constraint_batch_g9/g9_batch_clean",
    1: "results/runs_constraint_batch_g10/seed1",
    2: "results/runs_constraint_batch_g10/seed2",
}

BASELINE_JOINT = {"G0": 0.2640, "G1": 0.2433, "G2": 0.2673}
BASELINE_LAST50_NET = {"G0": 30.61, "G1": 35.63, "G2": 30.97}
BASELINE_PER_AGENT = {
    "G0": [26.06, 26.08, 26.69, 29.62, 34.64, 40.55],
    "G1": [29.67, 29.68, 30.87, 34.75, 40.83, 48.00],
    "G2": [26.27, 26.24, 26.88, 29.71, 35.23, 41.48],
}


def mean_ci(x: np.ndarray) -> tuple[float, float, float, float]:
    """mean, se, lo, hi for a 1-d sample. Never divided by sqrt(n_owners):
    docs/g10_gates.md section 5 -- the six owners share one rollout."""
    x = np.asarray(x, float).ravel()
    m = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")
    return m, se, m - 1.96 * se, m + 1.96 * se


# --------------------------------------------------------- structural gates


def gate_s1_s2_s3(run: Run) -> dict:
    """G10-S1/S2/S3: shape of every round's source batch, the pinned-policy
    checksum, and 250 distinct thetas."""
    problems: list[str] = []
    checksums: list[str] = []
    for rec in run.rounds:
        k = rec["round_k"]
        sb = rec.get("source_batch")
        if sb is None:
            problems.append(f"round {k}: no source_batch block")
            continue
        checksums.append(sb["policy_checksum"])
        if sb["M"] != EXPECTED_M or sb["R_m"] != EXPECTED_R_M:
            problems.append(f"round {k}: M/R_m = {sb['M']}/{sb['R_m']}")
        if sb["n_trajectories"] != EXPECTED_M * EXPECTED_R_M:
            problems.append(f"round {k}: {sb['n_trajectories']} trajectories")
        if sb["env_steps"] != EXPECTED_M * EXPECTED_R_M * EXPECTED_ROLLOUT:
            problems.append(f"round {k}: source env_steps {sb['env_steps']}")
        if not sb.get("worker_checksums_all_match"):
            problems.append(f"round {k}: worker checksum mismatch")
        if not sb.get("theta_k_checksum_stable_through_dual"):
            problems.append(f"round {k}: theta_k moved during collection")
        for a in run.aids:
            reports = rec["constraints"][a]["reports"]
            if len(reports) != EXPECTED_M:
                problems.append(f"round {k} owner {a}: {len(reports)} sources")
            if {x["source_id"] for x in reports} != set(sb["per_owner"][a]["source_means"]):
                problems.append(f"round {k} owner {a}: source ids disagree with the batch")
    n_rounds = len(run.rounds)
    if n_rounds != EXPECTED_ROUNDS:
        problems.append(f"{n_rounds} rounds, expected {EXPECTED_ROUNDS}")
    n_distinct = len(set(checksums))
    if n_distinct != n_rounds:
        problems.append(f"{n_distinct} distinct theta checksums over {n_rounds} rounds")
    return {
        "pass": not problems,
        "n_rounds": n_rounds,
        "n_distinct_theta_checksums": n_distinct,
        "n_problems": len(problems),
        "problems": problems[:20],
    }


def gate_s4(run: Run) -> dict:
    """G10-S4: within-run source seed uniqueness and replica disjointness."""
    env_all: list[int] = []
    torch_all: list[int] = []
    per_replica: dict[str, set[int]] = {}
    for rec in run.seeds:
        for rid, pairs in rec["seeds"].items():
            s = per_replica.setdefault(rid, set())
            for e, t in pairs:
                env_all.append(int(e))
                torch_all.append(int(t))
                s.add(int(e))
    ref_env: set[int] = set()
    for rec in run.refs:
        for e, t in rec.get("reference_seeds", []):
            env_all.append(int(e))
            torch_all.append(int(t))
            ref_env.add(int(e))
    rids = sorted(per_replica)
    overlaps = {
        f"{rids[i]}|{rids[j]}": len(per_replica[rids[i]] & per_replica[rids[j]])
        for i in range(len(rids))
        for j in range(i + 1, len(rids))
    }
    ref_overlaps = {r: len(per_replica[r] & ref_env) for r in rids}
    audit = run.meta.get("source_seed_audit", {})
    ok = (
        len(env_all) == len(set(env_all))
        and len(torch_all) == len(set(torch_all))
        and all(v == 0 for v in overlaps.values())
        and all(v == 0 for v in ref_overlaps.values())
        and int(audit.get("duplicate_seed_events", 0)) == 0
    )
    return {
        "pass": bool(ok),
        "n_env_seeds": len(env_all),
        "n_unique_env_seeds": len(set(env_all)),
        "n_torch_seeds": len(torch_all),
        "n_unique_torch_seeds": len(set(torch_all)),
        "replica_pairwise_overlap": overlaps,
        "replica_vs_reference_overlap": ref_overlaps,
        "run_audit_duplicate_events": int(audit.get("duplicate_seed_events", 0)),
        "seed_entropy": run.meta.get("config_snapshot", {}).get("source_collection", {}).get("seed_entropy"),
    }


def gate_s5(runs: dict[int, Run]) -> dict:
    """G10-S5: source streams disjoint ACROSS seeds. New in G10 -- the
    reason docs/g10_gates.md section 2 had to override seed_entropy at all."""
    sets: dict[int, set[int]] = {}
    for k, run in runs.items():
        s: set[int] = set()
        for rec in run.seeds:
            for pairs in rec["seeds"].values():
                for e, _t in pairs:
                    s.add(int(e))
        sets[k] = s
    ks = sorted(sets)
    pairs = {}
    ok = True
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = sets[ks[i]], sets[ks[j]]
            inter = len(a & b)
            exp = len(a) * len(b) / (2**31 - 1)
            # Two independent uniform samples of ~22.5k from 2**31 share
            # ~0.24 values by chance; anything near |a| means a shared stream.
            pairs[f"{ks[i]}-{ks[j]}"] = {
                "n_shared": inter,
                "expected_by_chance": exp,
                "fraction_of_stream": inter / max(1, len(a)),
            }
            if inter > 20:
                ok = False
    return {
        "pass": bool(ok) and len(ks) > 1,
        "n_seeds_compared": len(ks),
        "entropies": {str(k): runs[k].meta.get("config_snapshot", {})
                      .get("source_collection", {}).get("seed_entropy") for k in ks},
        "pairs": pairs,
    }


def gate_s6(run: Run) -> dict:
    """G10-S6: source trajectories never enter PPO."""
    es = run.meta.get("env_steps", {})
    expected_ppo = EXPECTED_ROUNDS * EXPECTED_ROLLOUT
    expected_src = EXPECTED_ROUNDS * EXPECTED_M * EXPECTED_R_M * EXPECTED_ROLLOUT
    # The reference collection adds R_ref trajectories on validation rounds
    # and is accounted separately below rather than folded into `source`.
    ppo_ok = int(es.get("ppo", -1)) == expected_ppo
    # A round's PPO batch is rollout_length transitions regardless of how many
    # source trajectories were collected -- if source data leaked in, the PPO
    # step count could not stay exactly at 250 x 2000.
    return {
        "pass": bool(ppo_ok),
        "ppo_env_steps": int(es.get("ppo", -1)),
        "expected_ppo_env_steps": expected_ppo,
        "source_env_steps": int(es.get("source", -1)),
        "source_env_steps_ex_reference": expected_src,
        "oracle_env_steps": int(es.get("oracle", -1)),
        "note": "BatchSourceResult carries only scalars and per-trajectory floats; "
                "no transitions. Enforced statically by tests/unit/test_source_batch.py::"
                "TestNoNeuralPathInTheBatchSourcePipeline and ::TestPpoIsUnaffected.",
    }


def _scan_for_value(blob, target: float, path: str, hits: list[str], counter: list[int]) -> None:
    """Walk every numeric leaf of `blob`, counting them and recording any that
    equals `target`. Bools are skipped: `violated` is a bool and comparing it
    numerically to a cost would be meaningless."""
    if isinstance(blob, dict):
        for kk, vv in blob.items():
            _scan_for_value(vv, target, path + "/" + kk, hits, counter)
    elif isinstance(blob, list):
        for idx, vv in enumerate(blob):
            _scan_for_value(vv, target, f"{path}[{idx}]", hits, counter)
    elif isinstance(blob, (int, float)) and not isinstance(blob, bool):
        counter[0] += 1
        if math.isfinite(blob) and target != 0.0 and abs(float(blob) - target) < 1e-12:
            hits.append(path)


def gate_s7_s8(run: Run) -> dict:
    """G10-S7: no oracle true cost reaches any learner-visible field.
    G10-S8: the withheld reference never reaches rounds.jsonl."""
    leaks: list[str] = []
    counter = [0]
    n = min(len(run.rounds), len(run.oracle))
    for rec, orec in zip(run.rounds[:n], run.oracle[:n], strict=True):
        for a in run.aids:
            truth = float(orec["agents"][a]["true_cost_return"])
            hits: list[str] = []
            _scan_for_value(rec["constraints"][a], truth, "constraints/" + a, hits, counter)
            leaks.extend(f"round {rec['round_k']} {a}: {h} == oracle true cost" for h in hits)
        sb = rec.get("source_batch", {})
        if "reference_mean" in sb or "reference_per_trajectory" in sb:
            leaks.append(f"round {rec['round_k']}: reference value present in rounds.jsonl")
    return {
        "pass": not leaks,
        "n_numeric_fields_checked": counter[0],
        "n_leaks": len(leaks),
        "leaks": leaks[:20],
    }


def gate_s9(run: Run, n_agents: int, topology: str, topo_p, graph_seed: int) -> dict:
    """G10-S9: dual arithmetic exact, consensus mixing included.

    (a) lambda_mixed_before == W @ lambda_after[k-1]
    (b) lambda_after - lambda_mixed_before == eta * residual, unclipped cells
    """
    if run.lam_mixed is None:
        return {"pass": False, "note": "lambda_mixed_before not logged"}
    # `build_topology` returns the mixing matrix W itself, exactly as
    # `ExperimentRun` builds it (src/safelie/training/loop.py:116) -- it is
    # NOT an adjacency matrix to be re-weighted.
    W = build_topology(topology, n_agents, p=topo_p, graph_seed=graph_seed)
    prev = run.lam[:-1]
    mix = run.lam_mixed[1:]
    pred = prev @ W.T
    mix_err = float(np.abs(pred - mix).max())
    nomix_err = float(np.abs(prev - mix).max())

    d9 = gate_g9d(run, ETA_LAMBDA, LAMBDA_MAX)
    wiring_err = float(d9["i_exact_wiring"]["max_abs_error"])
    return {
        "pass": bool(mix_err <= D_WIRING_TOL and d9["i_exact_wiring"]["pass"] and d9["ii_sign"]["pass"]),
        "a_consensus_mixing": {
            "pass": bool(mix_err <= D_WIRING_TOL),
            "W": topology,
            "max_abs_error": mix_err,
            "identity_reconstruction_error": nomix_err,
        },
        "b_dual_ascent": {
            "pass": bool(d9["i_exact_wiring"]["pass"]),
            "max_abs_error": wiring_err,
            "n_unclipped": d9["i_exact_wiring"]["n_unclipped"],
        },
        "c_sign": d9["ii_sign"],
    }


# --------------------------------------------------------- scientific gates


def gate_a(g9b: dict) -> dict:
    """G10-A: aggregate calibration against the withheld R_ref reference."""
    per_ck = {}
    for k, v in sorted(g9b["per_round"].items(), key=lambda kv: int(kv[0])):
        zs = [o["z_agg"] for o in v["owners"].values()]
        bs = [o["aggregate_bias"] for o in v["owners"].values()]
        refs = [o["reference_mean"] for o in v["owners"].values()]
        per_ck[int(k)] = {
            "z_bar": float(np.mean(zs)),
            "bias_bar": float(np.mean(bs)),
            "per_owner_z": {a: o["z_agg"] for a, o in v["owners"].items()},
            "per_owner_bias": {a: o["aggregate_bias"] for a, o in v["owners"].items()},
            "reference_mean_net": float(np.mean(refs)),
            "n_owners_over_d": int(sum(r > BUDGET for r in refs)),
            "abs_z_bar_ok": bool(abs(float(np.mean(zs))) <= A_Z_MAX),
        }
    zbars = [v["z_bar"] for v in per_ck.values()]
    bbars = [v["bias_bar"] for v in per_ck.values()]
    pooled = float(np.mean(bbars)) if bbars else float("nan")
    i_ok = bool(per_ck) and all(v["abs_z_bar_ok"] for v in per_ck.values())
    ii_ok = bool(abs(pooled) <= A_POOLED_BIAS_MAX)
    m, se, lo, hi = mean_ci(np.array(bbars)) if len(bbars) > 1 else (pooled, float("nan"), float("nan"), float("nan"))
    return {
        "pass": bool(i_ok and ii_ok),
        "i_per_checkpoint_z": {"pass": i_ok, "bar": A_Z_MAX,
                               "z_bar": {str(k): v["z_bar"] for k, v in per_ck.items()},
                               "max_abs_z_bar": float(np.max(np.abs(zbars))) if zbars else float("nan")},
        "ii_pooled_bias": {"pass": ii_ok, "bar": A_POOLED_BIAS_MAX,
                           "pooled_bias": pooled, "se": se, "ci95": [lo, hi],
                           "per_checkpoint_bias": {str(k): v["bias_bar"] for k, v in per_ck.items()}},
        "per_checkpoint": per_ck,
    }


def gate_c(run: Run) -> dict:
    """G10-C: learning health."""
    arrays = {
        "mechanism": run.mechanism, "residual": run.residual, "lambda": run.lam,
        "kl": run.kl, "entropy": run.entropy, "true_cost": run.true_cost, "spread": run.spread,
    }
    counts = {k: int((~np.isfinite(v)).sum()) for k, v in arrays.items()}
    i_ok = all(v == 0 for v in counts.values())

    first50 = float(run.task_return[:50].mean())
    last50 = float(run.task_return[-50:].mean())
    delta = last50 - first50
    ii_ok = bool(delta >= C_RETURN_DELTA_MIN)

    kl_med = float(np.median(run.kl))
    kl_p95 = float(np.percentile(run.kl, 95))
    iii_ok = bool(kl_med <= C_KL_MEDIAN_MAX and kl_p95 <= C_KL_P95_MAX)

    frac_sat = float((run.lam >= LAMBDA_MAX - 1e-9).mean())
    iv_ok = bool(frac_sat < C_LAMBDA_SATURATION_MAX)

    ent_first10 = float(run.entropy[:10].mean())
    ent_last10 = float(run.entropy[-10:].mean())
    ent_min = float(run.entropy.min())
    checksums = [r["source_batch"]["policy_checksum"] for r in run.rounds if "source_batch" in r]
    distinct_frac = len(set(checksums)) / max(1, len(checksums))
    v_ok = bool(ent_last10 < ent_first10 and ent_min > 0 and math.isclose(distinct_frac, 1.0))

    blocks = [float(run.task_return[i:i + 50].mean()) for i in range(0, len(run.task_return), 50)]
    return {
        "pass": bool(i_ok and ii_ok and iii_ok and iv_ok and v_ok),
        "i_no_nonfinite": {"pass": i_ok, "counts": counts},
        "ii_task_return": {"pass": ii_ok, "bar": C_RETURN_DELTA_MIN,
                           "first50": first50, "last50": last50, "delta": delta,
                           "blocks_of_50": blocks},
        "iii_kl": {"pass": iii_ok, "median": kl_med, "p95": kl_p95,
                   "bar_median": C_KL_MEDIAN_MAX, "bar_p95": C_KL_P95_MAX,
                   "max": float(run.kl.max())},
        "iv_lambda_saturation": {"pass": iv_ok, "frac_at_max": frac_sat, "bar": C_LAMBDA_SATURATION_MAX},
        "v_meaningful_updates": {"pass": v_ok, "entropy_first10": ent_first10,
                                 "entropy_last10": ent_last10, "entropy_min": ent_min,
                                 "distinct_theta_checksum_fraction": distinct_frac},
        "value_loss_note": "value/critic losses are not logged per round in this schema; "
                           "KL, entropy and the finite-value scan cover learning stability.",
    }


def gate_d(run: Run, s9: dict) -> dict:
    """G10-D: dual dynamics."""
    d9 = gate_g9d(run, ETA_LAMBDA, LAMBDA_MAX)
    frac_pos = float((run.lam > 1e-9).mean())
    iii_ok = bool(frac_pos >= D_FRAC_LAMBDA_POSITIVE_MIN and float(run.lam.max()) < LAMBDA_MAX)
    per_agent = {
        a: {"lambda_mean": float(run.lam[:, i].mean()),
            "lambda_last50_mean": float(run.lam[-50:, i].mean()),
            "lambda_max": float(run.lam[:, i].max()),
            "residual_last50_mean": float(run.residual[-50:, i].mean()),
            "mechanism_last50_mean": float(run.mechanism[-50:, i].mean())}
        for i, a in enumerate(run.aids)
    }
    dlam = np.diff(run.lam, axis=0)
    return {
        "pass": bool(s9["pass"] and d9["ii_sign"]["pass"] and iii_ok),
        "i_wiring": s9,
        "ii_sign": d9["ii_sign"],
        "iii_responsive": {"pass": iii_ok, "frac_lambda_positive": frac_pos,
                           "bar": D_FRAC_LAMBDA_POSITIVE_MIN,
                           "lambda_mean": float(run.lam.mean()),
                           "lambda_max": float(run.lam.max()),
                           "lambda_max_config": LAMBDA_MAX},
        "iv_reported": {
            "delta_lambda_mean_abs": float(np.abs(dlam).mean()),
            "sign_flip_rate": d9["iv_sign_flip_rate"]["value"],
            "pure_noise_reference": 0.5,
            "lambda_vs_true_cost_corr": d9["v_lambda_vs_true_cost_corr"]["by_lag"],
            "lambda_cross_agent_spread_last50": float(run.lam[-50:].std(axis=1).mean()),
            "per_agent": per_agent,
            "residual_mean": float(run.residual.mean()),
            "residual_last50_mean": float(run.residual[-50:].mean()),
        },
    }


def gate_e(run: Run, g9e: dict) -> dict:
    """G10-E: joint false-safe rate, with the base rate beside it."""
    joint = float(run.false_safe.mean())
    joint_lo, joint_hi = wilson(int(run.false_safe.sum()), run.false_safe.size)
    base = float(run.unsafe.mean())
    base_lo, base_hi = wilson(int(run.unsafe.sum()), run.unsafe.size)
    cond, cond_ci, k_fs, n_unsafe = run.fsr()
    i_ok = bool(joint_hi < E_JOINT_WILSON_UPPER_MAX)
    return {
        "pass": i_ok,
        "i_joint": {"pass": i_ok, "bar_wilson_upper_below": E_JOINT_WILSON_UPPER_MAX,
                    "bar_source": "G1 joint rate, best of the committed pre-G9 campaigns",
                    "joint_rate": joint, "wilson_ci": [joint_lo, joint_hi],
                    "n_false_safe": int(run.false_safe.sum()), "n_cells": int(run.false_safe.size)},
        "ii_base_rate": {"violation_rate": base, "wilson_ci": [base_lo, base_hi],
                         "n_unsafe": n_unsafe},
        "iii_decomposition": {"conditional_x_base": cond * base, "joint_rate": joint,
                              "note": "joint = P(mech<=d | J>d) * P(J>d); if i fails, this says which factor moved"},
        "iv_conditional_secondary": {
            "conditional_fsr": cond, "wilson_ci": list(cond_ci),
            "per_owner": g9e["per_owner"], "by_phase": g9e["by_phase"],
            "by_margin_band": g9e["by_margin_band"],
            "gated": False,
            "why_not_gated": "docs/g10_gates.md 4.1 -- not invariant to the true-cost base rate",
        },
        "baselines_joint": BASELINE_JOINT,
    }


def gate_f(run: Run) -> dict:
    """G10-F: constraint control, on the expected-cost constraint of
    docs/g10_gates.md section 3, enforced at the network average per 3.1."""
    tc = run.true_cost
    net = tc.mean(axis=1)
    last50 = float(net[-50:].mean())
    mid = float(net[50:150].mean())
    i_ok = bool(last50 <= F_LAST50_NET_AVG_MAX)
    ii_ok = bool(last50 - mid <= F_DIVERGENCE_MAX)

    m, se, lo, hi = mean_ci(net[-50:])
    per_agent = {}
    for i, a in enumerate(run.aids):
        am, ase, alo, ahi = mean_ci(tc[-50:, i])
        per_agent[a] = {
            "last50_mean": am, "se": ase, "ci95": [alo, ahi],
            "last50_violation_rate": float((tc[-50:, i] > BUDGET).mean()),
            "run_mean": float(tc[:, i].mean()),
            "run_violation_rate": float(run.violated[:, i].mean()),
            "exceeds_d_in_expectation": bool(alo > BUDGET),
        }
    n_over = sum(1 for v in per_agent.values() if v["exceeds_d_in_expectation"])

    refs = {}
    for rec in run.refs:
        means = {a: float(np.mean(rec["reference_per_trajectory"][a])) for a in run.aids}
        refs[rec["round_k"]] = {
            "per_agent": means,
            "net_average": float(np.mean(list(means.values()))),
            "n_agents_over_d": int(sum(v > BUDGET for v in means.values())),
            "R_ref": rec["R_ref"],
        }

    peak = np.array([[r["agents"][a]["peak_true_cost"] for a in run.aids] for r in run.oracle], float)
    return {
        "pass": bool(i_ok and ii_ok),
        "i_last50_net_average": {"pass": i_ok, "bar": F_LAST50_NET_AVG_MAX,
                                 "value": last50, "se": se, "ci95": [lo, hi]},
        "ii_no_divergence": {"pass": ii_ok, "bar": F_DIVERGENCE_MAX,
                             "last50": last50, "rounds_51_150": mid, "delta": last50 - mid},
        "iii_reported": {
            "mean": float(tc.mean()), "median": float(np.median(tc)), "sd": float(tc.std(ddof=1)),
            "quantiles": {q: float(np.quantile(tc, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "violation_rate": float(run.violated.mean()),
            "peak_violation_max": float(peak.max()),
            "first_half": float(tc[:len(tc) // 2].mean()), "second_half": float(tc[len(tc) // 2:].mean()),
            "blocks_of_50": [float(net[i:i + 50].mean()) for i in range(0, len(net), 50)],
            "violation_blocks_of_50": [float(run.violated[i:i + 50].mean())
                                       for i in range(0, len(run.violated), 50)],
            "per_agent": per_agent,
            "n_agents_over_d_in_expectation": n_over,
            "reference_J_C_at_checkpoints": refs,
            "constraint_form": "expected cost E[J_C^i] <= d (paper/main_iclr.tex:104-109); "
                               "P(J_C>d) is reported, not gated (docs/g10_gates.md section 3)",
        },
        "baselines_last50_net": BASELINE_LAST50_NET,
    }


# ------------------------------------------------------------------- driver


def analyse_seed(seed: int, run_dir: Path, all_runs: dict[int, Run]) -> dict:
    run = Run(run_dir)
    cs = run.meta.get("config_snapshot", {})
    n_agents = cs.get("env", {}).get("n_agents", 6)
    topology = cs.get("topology", {}).get("name", "ring")

    g9b = gate_g9b(run, EXPECTED_M, EXPECTED_R_M)
    g9e_raw = gate_g9e(run, {})
    g9f = gate_g9f(run, EXPECTED_M, EXPECTED_R_M)

    s1 = gate_s1_s2_s3(run)
    s4 = gate_s4(run)
    s6 = gate_s6(run)
    s78 = gate_s7_s8(run)
    topo = cs.get("topology", {})
    s9 = gate_s9(run, n_agents, topology, topo.get("p"), topo.get("graph_seed", 0))

    a = gate_a(g9b)
    b = {
        "pass": bool(B_VAR_RATIO_BAND[0] <= g9f["ratio"] <= B_VAR_RATIO_BAND[1]),
        "i_variance_ratio": {"ratio": g9f["ratio"], "band": list(B_VAR_RATIO_BAND),
                             "mean_s2_between": g9f["mean_s2_between"],
                             "mean_expected": g9f["mean_expected"], "n_cells": g9f["n_cells"]},
        "ii_reported_se": {
            "observed_se_agg": g9f["observed_se_agg"],
            "predicted_se_agg": g9f["predicted_se_agg"],
            "mean_se_m": g9f["mean_se_m"],
            "observed_over_predicted": g9f["observed_se_agg"] / g9f["predicted_se_agg"],
            "effective_sample_size_note": "docs/g10_gates.md section 5: the six owners share one "
                                          "rollout, so the ~1500 cells are ~250 effective; G9f's "
                                          "bootstrap CI is correspondingly too narrow.",
        },
    }
    c = gate_c(run)
    d = gate_d(run, s9)
    e = gate_e(run, g9e_raw)
    f = gate_f(run)

    meta = run.meta
    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "git_sha": meta.get("git", {}).get("sha"),
        "git_dirty": meta.get("git", {}).get("dirty"),
        "status": meta.get("status"),
        "seed_entropy": cs.get("source_collection", {}).get("seed_entropy"),
        "cost": {"timing": meta.get("timing", {}), "env_steps": meta.get("env_steps", {}),
                 "workers": cs.get("source_collection", {}).get("workers"),
                 "chunks_per_worker": cs.get("source_collection", {}).get("chunks_per_worker")},
        "structural": {"S1_S2_S3": s1, "S4": s4, "S6": s6, "S7_S8": s78, "S9": s9},
        "gates": {"G10_A": a, "G10_B": b, "G10_C": c, "G10_D": d, "G10_E": e, "G10_F": f},
        "secondary": {
            "G9b_per_source_z": {
                "pass": g9b["pass"], "n_rounds_pass": g9b["n_rounds_pass"],
                "n_rounds": g9b["n_rounds"], "required": g9b["required"],
                "per_checkpoint_source_z": {
                    str(k): {a: [{"source_id": s["source_id"], "z": s["z"], "label": s["label"]}
                                 for s in o["sources"]] for a, o in v["owners"].items()}
                    for k, v in g9b["per_round"].items()},
                "gated": False,
                "why_not_gated": "docs/g10_gates.md 4.2 -- 18 owner-cells per checkpoint are ~3 "
                                 "effective tests; a 2-sigma bar fails ~13% of the time under a "
                                 "perfectly calibrated source.",
            },
        },
        "_run": run,
    }


def summarise(per_seed: dict[int, dict], s5: dict) -> dict:
    seeds = sorted(per_seed)
    gate_ids = ["G10_A", "G10_B", "G10_C", "G10_D", "G10_E", "G10_F"]
    struct_ids = ["S1_S2_S3", "S4", "S6", "S7_S8", "S9"]

    table = {}
    for g in gate_ids:
        table[g] = {str(s): bool(per_seed[s]["gates"][g]["pass"]) for s in seeds}
        table[g]["overall"] = all(table[g][str(s)] for s in seeds)
    for g in struct_ids:
        table[g] = {str(s): bool(per_seed[s]["structural"][g]["pass"]) for s in seeds}
        table[g]["overall"] = all(table[g][str(s)] for s in seeds)
    table["S5_cross_seed"] = {str(s): s5["pass"] for s in seeds}
    table["S5_cross_seed"]["overall"] = s5["pass"]

    all_bias = []
    for s in seeds:
        all_bias.extend(per_seed[s]["gates"]["G10_A"]["ii_pooled_bias"]["per_checkpoint_bias"].values())
    pooled_all = float(np.mean(all_bias)) if all_bias else float("nan")
    pm, pse, plo, phi = mean_ci(np.array(all_bias)) if len(all_bias) > 1 else (pooled_all, float("nan"),) * 2
    meta_bias_ok = bool(abs(pooled_all) <= A_META_POOLED_BIAS_MAX)

    struct_ok = all(table[g]["overall"] for g in struct_ids) and s5["pass"]
    sci_misses = [(g, s) for g in gate_ids for s in seeds if not table[g][str(s)]]
    a_ok = table["G10_A"]["overall"] and meta_bias_ok

    if not struct_ok or not table["G10_A"]["overall"] or len(sci_misses) >= 2:
        verdict = "FAIL"
    elif len(sci_misses) == 0 and meta_bias_ok:
        verdict = "PASS"
    else:
        verdict = "CONDITIONAL PASS"

    return {
        "gate_table": table,
        "meta_pooled_bias": {"pass": meta_bias_ok, "bar": A_META_POOLED_BIAS_MAX,
                             "value": pooled_all, "se": pse, "ci95": [plo, phi],
                             "n_checkpoints": len(all_bias)},
        "structural_all_pass": struct_ok,
        "aggregate_calibration_all_pass": a_ok,
        "scientific_misses": [{"gate": g, "seed": s} for g, s in sci_misses],
        "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seed0-only", action="store_true")
    ap.add_argument("--out", default="results/runs_constraint_batch_g10/g10_report.json")
    args = ap.parse_args()
    seeds = [0] if args.seed0_only else args.seeds

    available = {}
    for s in seeds:
        p = Path(SEED_DIRS[s])
        if not (p / "rounds.jsonl").exists():
            print(f"  [skip] seed {s}: {p} has no rounds.jsonl yet")
            continue
        available[s] = p
    if not available:
        print("no runs available")
        return 1

    per_seed = {s: analyse_seed(s, p, {}) for s, p in available.items()}
    runs = {s: per_seed[s].pop("_run") for s in per_seed}
    s5 = gate_s5(runs) if len(runs) > 1 else {"pass": True, "n_seeds_compared": len(runs),
                                              "note": "only one seed available", "pairs": {},
                                              "entropies": {str(s): per_seed[s]["seed_entropy"] for s in per_seed}}
    summary = summarise(per_seed, s5)

    report = {"seeds": {str(s): per_seed[s] for s in per_seed},
              "S5_cross_seed": s5, "summary": summary}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    # ------------------------------------------------------------- print
    print("\n" + "=" * 78)
    print("G10 -- clean three-seed validation (docs/g10_gates.md)")
    print("=" * 78)
    for s in sorted(per_seed):
        r = per_seed[s]
        print(f"\nseed {s}  {r['run_dir']}")
        print(f"  git {r['git_sha']}  dirty={r['git_dirty']}  status={r['status']}")
        print(f"  seed_entropy {r['seed_entropy']}")
        t, es = r["cost"]["timing"], r["cost"]["env_steps"]
        if t:
            print(f"  wall {t.get('wall_clock_s', 0)/3600:.2f} h  source {t.get('source_s', 0)/3600:.2f} h "
                  f"({100*t.get('source_s', 0)/max(1e-9, t.get('wall_clock_s', 1)):.0f}%)  "
                  f"env steps ppo={es.get('ppo')} source={es.get('source')} oracle={es.get('oracle')}")

    print("\n--- structural ---")
    for s in sorted(per_seed):
        st = per_seed[s]["structural"]
        flags = " ".join(f"{k}={'ok' if v['pass'] else 'FAIL'}" for k, v in st.items())
        print(f"  seed {s}: {flags}")
    print(f"  S5 cross-seed stream disjointness: {'ok' if s5['pass'] else 'FAIL'}  {s5.get('pairs', {})}")

    print("\n--- G10-A aggregate calibration (Question A: the source) ---")
    for s in sorted(per_seed):
        a = per_seed[s]["gates"]["G10_A"]
        zb = a["i_per_checkpoint_z"]["z_bar"]
        bb = a["ii_pooled_bias"]["per_checkpoint_bias"]
        print(f"  seed {s}: z_bar " + " ".join(f"{k}:{v:+.2f}" for k, v in zb.items()))
        print("          bias  " + " ".join(f"{k}:{v:+.2f}" for k, v in bb.items())
              + f"   pooled {a['ii_pooled_bias']['pooled_bias']:+.3f}  -> {'PASS' if a['pass'] else 'FAIL'}")
    mb = summary["meta_pooled_bias"]
    print(f"  3-seed pooled bias {mb['value']:+.3f} (bar |b|<={mb['bar']}) "
          f"CI95 [{mb['ci95'][0]:+.3f}, {mb['ci95'][1]:+.3f}] -> {'PASS' if mb['pass'] else 'FAIL'}")

    print("\n--- G10-B source precision ---")
    for s in sorted(per_seed):
        b = per_seed[s]["gates"]["G10_B"]
        print(f"  seed {s}: var ratio {b['i_variance_ratio']['ratio']:.3f} band {B_VAR_RATIO_BAND}  "
              f"SE_agg obs {b['ii_reported_se']['observed_se_agg']:.4f} vs pred "
              f"{b['ii_reported_se']['predicted_se_agg']:.4f} "
              f"({b['ii_reported_se']['observed_over_predicted']:.3f}x) -> {'PASS' if b['pass'] else 'FAIL'}")

    print("\n--- G10-C learning / G10-D dual (Question B: the policy) ---")
    for s in sorted(per_seed):
        c, d = per_seed[s]["gates"]["G10_C"], per_seed[s]["gates"]["G10_D"]
        print(f"  seed {s}: return {c['ii_task_return']['first50']:.1f} -> "
              f"{c['ii_task_return']['last50']:.1f} (D {c['ii_task_return']['delta']:+.1f})  "
              f"KL {c['iii_kl']['median']:.5f}/{c['iii_kl']['p95']:.5f}  "
              f"ent {c['v_meaningful_updates']['entropy_first10']:.2f}->"
              f"{c['v_meaningful_updates']['entropy_last10']:.2f}  -> C {'PASS' if c['pass'] else 'FAIL'}")
        print(f"          lambda mean {d['iii_responsive']['lambda_mean']:.3f} max "
              f"{d['iii_responsive']['lambda_max']:.3f} frac>0 {d['iii_responsive']['frac_lambda_positive']:.3f}  "
              f"mix err {d['i_wiring']['a_consensus_mixing']['max_abs_error']:.2e}  "
              f"ascent err {d['i_wiring']['b_dual_ascent']['max_abs_error']:.2e} -> D "
              f"{'PASS' if d['pass'] else 'FAIL'}")

    print("\n--- G10-E joint false-safe (system) ---")
    for s in sorted(per_seed):
        e = per_seed[s]["gates"]["G10_E"]
        j, ci = e["i_joint"]["joint_rate"], e["i_joint"]["wilson_ci"]
        print(f"  seed {s}: joint {j:.4f} CI [{ci[0]:.4f}, {ci[1]:.4f}] (upper must be < "
              f"{E_JOINT_WILSON_UPPER_MAX})  base P(J>d) {e['ii_base_rate']['violation_rate']:.4f}  "
              f"conditional {e['iv_conditional_secondary']['conditional_fsr']:.4f} (secondary) -> "
              f"{'PASS' if e['pass'] else 'FAIL'}")

    print("\n--- G10-F constraint control ---")
    for s in sorted(per_seed):
        f = per_seed[s]["gates"]["G10_F"]
        i, rep = f["i_last50_net_average"], f["iii_reported"]
        print(f"  seed {s}: last50 net-avg {i['value']:.2f} CI [{i['ci95'][0]:.2f}, {i['ci95'][1]:.2f}] "
              f"(bar <= {F_LAST50_NET_AVG_MAX})  divergence {f['ii_no_divergence']['delta']:+.2f}  "
              f"-> {'PASS' if f['pass'] else 'FAIL'}")
        print("          per-agent last50: " + " ".join(
            f"{v['last50_mean']:.2f}" for v in rep["per_agent"].values())
            + f" | {rep['n_agents_over_d_in_expectation']} of {len(rep['per_agent'])} over d in "
              f"expectation (reported, not gated)")
        print(f"          violation rate {rep['violation_rate']:.4f}  sd {rep['sd']:.2f}  "
              f"median {rep['median']:.2f}")

    print("\n--- secondary (reported, not gating) ---")
    for s in sorted(per_seed):
        g9b = per_seed[s]["secondary"]["G9b_per_source_z"]
        print(f"  seed {s}: G9b per-source |z|<=2 -> {g9b['n_rounds_pass']}/{g9b['n_rounds']} "
              f"checkpoints (G9 required {g9b['required']}): "
              f"{'would pass' if g9b['pass'] else 'would fail'}")

    print("\n" + "=" * 78)
    print("GATE TABLE")
    hdr = "  " + "gate".ljust(16) + "".join(f"seed{s}".rjust(9) for s in sorted(per_seed)) + "   overall"
    print(hdr)
    for g, row in summary["gate_table"].items():
        cells = "".join(("PASS" if row[str(s)] else "FAIL").rjust(9) for s in sorted(per_seed))
        print("  " + g.ljust(16) + cells + "   " + ("PASS" if row["overall"] else "FAIL"))
    print(f"\nVERDICT: {summary['verdict']}")
    if summary["scientific_misses"]:
        print("  misses: " + ", ".join(f"{m['gate']}@seed{m['seed']}" for m in summary["scientific_misses"]))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
