#!/usr/bin/env python
"""G8 (docs/g8_gates.md): does the G7 disjoint-trajectory source
construction survive the nonstationary training regime?

Reads the pools `scripts/g8_collect_nonstationary_dataset.py` wrote for
each of five training anchors, plus the policy snapshots
`scripts/g8_collect_training_checkpoints.py` recovered, and answers one
question:

    when theta is changing during training, can M=3 trajectory-batch
    sources still be treated as estimates of the SAME J_C^i(theta_k) at
    the moment the dual update is made?

Runs no PPO, no GAE, no dual update, no attack, no RCE, no environment
step, and fits no model. Every J_C estimate here is `numpy.mean` over
already-collected `mc_cost_return` scalars; the only network operation is
a closed-form Gaussian KL between two restored policies evaluated on a
fixed state set, used purely as a policy-distance readout.

Usage:
    .venv/Scripts/python.exe scripts/g8_nonstationary_source_diagnostic.py
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from safelie.algos.networks import GaussianPolicy
from safelie.algos.normalization import RunningMeanStd

# --- pre-declared constants, docs/g8_gates.md. None may be changed after
# --- a G8 number has been read.
BUDGET_D = 25.0
Z_GOOD = 2.0            # step 5, inherited verbatim from G7
Z_POOR = 3.0            # step 5, inherited verbatim from G7
Z_TOL = 2.0             # step 7, the central drift criterion
M = 3                   # step 2, held at G7's S2-M3 operating point
R_M = 30
R_REF = 120
OWNERS_REQUIRED = 5     # of 6, G7d's bar, unchanged
DRIFT_BAND_NEGLIGIBLE = 0.5   # step 6
DRIFT_BAND_DOMINANT = 2.0     # step 6
ANCHORS = [25, 75, 125, 175, 225]
STRIDE = {25: 30, 75: 30, 125: 30, 175: 30, 225: 24}
PHASE = {25: "early", 75: "mid-early", 125: "middle", 175: "mid-late", 225: "late"}
SCHEDULE_RM = [1, 5, 10, 30]  # step 8 / gate G8f
ROLES = ["anchor", "plus1", "plus2", "plusdelta"]


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_anchor(dataset_dir: Path, k: int) -> dict:
    d = dataset_dir / f"anchor_{k:03d}"
    meta = json.loads((d / "meta.json").read_text())
    pools = {}
    for role in ROLES:
        for kind in ("src", "ref"):
            name = f"{role}_{kind}"
            z = np.load(d / f"{name}.npz", allow_pickle=True)
            pools[name] = {
                "mc_cost_return": z["mc_cost_return"],
                "mc_task_return": z["mc_task_return"],
                "round_seeds": z["round_seeds"],
            }
    agent_ids = [str(a) for a in np.load(d / "anchor_src.npz", allow_pickle=True)["agent_ids"]]
    kl_states = np.load(d / "kl_eval_states.npz", allow_pickle=True)
    return {"meta": meta, "pools": pools, "agent_ids": agent_ids, "kl_states": kl_states, "dir": d}


def load_policy(ckpt_path: Path, agent_ids: list[str]) -> dict:
    """Restore ONLY each agent's policy net and observation normalizer
    from a snapshot -- enough to evaluate the action distribution, and
    deliberately nothing else (no critic, no optimizer, no head)."""
    state = torch.load(ckpt_path, weights_only=False)
    out = {}
    for aid in agent_ids:
        sd = state["agents"][aid]["policy"]
        obs_dim = sd["mean_net.0.weight"].shape[1]
        action_dim = sd["mean_net.4.weight"].shape[0]
        hidden = sd["mean_net.0.weight"].shape[0]
        pol = GaussianPolicy(obs_dim, action_dim, hidden)
        pol.load_state_dict(sd)
        pol.eval()
        rms = RunningMeanStd(shape=(obs_dim,))
        rms.load_state_dict(state["agents"][aid]["obs_rms"])
        out[aid] = (pol, rms)
    return {"agents": out, "round_index": int(state["round_index"]),
            "policy_flat": {aid: torch.cat([v.reshape(-1) for v in state["agents"][aid]["policy"].values()])
                            for aid in agent_ids}}


# --------------------------------------------------------------------------
# Policy distance (docs/g8_gates.md step 4)
# --------------------------------------------------------------------------

def gaussian_kl(mu1, sd1, mu2, sd2) -> torch.Tensor:
    """KL(N(mu1,sd1) || N(mu2,sd2)) for diagonal Gaussians, summed over
    action dimensions. The policies squash with a FIXED tanh applied
    identically at the same state, so the log-Jacobian cancels exactly and
    this pre-tanh KL is the KL of the squashed action distributions."""
    return (torch.log(sd2 / sd1) + (sd1**2 + (mu1 - mu2) ** 2) / (2 * sd2**2) - 0.5).sum(-1)


def policy_distance(pol_a: dict, pol_b: dict, kl_states, agent_ids: list[str]) -> dict:
    per_agent_kl, per_agent_param = {}, {}
    for aid in agent_ids:
        obs = torch.as_tensor(np.asarray(kl_states[aid]), dtype=torch.float32)
        with torch.no_grad():
            pa, ra = pol_a["agents"][aid]
            pb, rb = pol_b["agents"][aid]
            da = pa.distribution(ra.normalize(obs))
            db = pb.distribution(rb.normalize(obs))
            per_agent_kl[aid] = float(gaussian_kl(da.mean, da.stddev, db.mean, db.stddev).mean())
        fa, fb = pol_a["policy_flat"][aid], pol_b["policy_flat"][aid]
        per_agent_param[aid] = float(torch.linalg.norm(fb - fa) / torch.linalg.norm(fa))
    return {
        "kl_per_agent": per_agent_kl,
        "kl_mean_over_agents": float(np.mean(list(per_agent_kl.values()))),
        "kl_max_over_agents": float(np.max(list(per_agent_kl.values()))),
        "param_l2_rel_per_agent": per_agent_param,
        "param_l2_rel_mean": float(np.mean(list(per_agent_param.values()))),
        "n_eval_states": int(np.asarray(kl_states[agent_ids[0]]).shape[0]),
    }


# --------------------------------------------------------------------------
# Per-owner source statistics
# --------------------------------------------------------------------------

def source_row(mean: float, r: int, se_m: float, ref_mean: float, se_ref: float,
               drift: float | None = None) -> dict:
    bias = mean - ref_mean
    denom = math.sqrt(se_m**2 + se_ref**2)
    z = bias / denom
    p_within = _phi((BUDGET_D - (ref_mean + (drift or 0.0))) / se_m)
    row = {
        "r": r, "mean": mean, "se": se_m, "bias_vs_anchor_reference": bias,
        "abs_bias_frac_of_budget": abs(bias) / BUDGET_D, "z": z,
        "label": "good" if abs(z) <= Z_GOOD else ("poor" if abs(z) > Z_POOR else "mixed"),
        "ci95": [mean - 1.96 * se_m, mean + 1.96 * se_m],
        # P(source reports "within budget") under its own sampling law,
        # centred on the anchor's best-estimate J_C plus this slot's drift.
        "prob_reports_within_budget": p_within,
        # docs/g8_gates.md step 5 defines the false-safe probability as
        # that same number CONDITIONED on the anchor actually being over
        # budget; when the anchor's reference is already within `d` a
        # "within budget" report is correct, not a false safe, so the
        # field is null rather than a large and misleading number.
        "anchor_reference_over_budget": bool(ref_mean > BUDGET_D),
        "false_safe_prob": (p_within if ref_mean > BUDGET_D else None),
    }
    if drift is not None:
        row["target_drift"] = drift
    return row


def condition_block(sources: list[dict], ref_mean: float) -> dict:
    biases = np.array([s["bias_vs_anchor_reference"] for s in sources])
    means = np.array([s["mean"] for s in sources])
    pairs = {f"{a}-{b}": float(means[a] - means[b]) for a, b in itertools.combinations(range(len(means)), 2)}
    return {
        "sources": sources,
        "bias": float(biases.mean()),
        "mae": float(np.abs(biases).mean()),
        "rmse": float(np.sqrt((biases**2).mean())),
        "aggregate_mean": float(means.mean()),
        "aggregate_bias": float(means.mean() - ref_mean),
        "spread_max_minus_min": float(means.max() - means.min()),
        "pairwise_source_differences": pairs,
        "all_good": all(s["label"] == "good" for s in sources),
        "n_good": sum(s["label"] == "good" for s in sources),
        "n_sources": len(sources),
    }


def pooled_cross_owner_empirical(owner_results: list[dict], cond: str) -> dict:
    """Empirical cross-SOURCE correlation, pooled across the 6 OWNERS --
    G7's convention verbatim (`scripts/g7_estimand_source_diagnostic.py::
    pooled_cross_owner_empirical`), computed on the ERROR vector
    (`bias_vs_anchor_reference`), never on raw means, because owners
    differ by a large fixed cost level that would otherwise dominate.

    A single owner supplies only M scalars, so a per-owner Pearson r is
    not computable at all. n=6 owners is still small: this number is a
    supporting readout, never evidence of independence (which is a
    closed-form property of the disjoint draws), and never evidence that
    sequential sources share a target (which is what target drift, not
    correlation, decides -- gate G8e-ii).
    """
    errors = np.array([[s["bias_vs_anchor_reference"] for s in res[cond]["sources"]] for res in owner_results])
    m = errors.shape[1]
    cov = np.cov(errors, rowvar=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(errors, rowvar=False)
    off = corr[~np.eye(m, dtype=bool)]
    eigs = np.clip(np.linalg.eigvalsh(np.atleast_2d(cov)), 0, None)
    pr = float((eigs.sum() ** 2) / (eigs**2).sum()) if (eigs**2).sum() > 0 else float("nan")
    return {
        "n_owners": int(errors.shape[0]), "n_sources": int(m),
        "covariance_matrix": np.atleast_2d(cov).tolist(),
        "correlation_matrix": np.atleast_2d(corr).tolist(),
        "mean_off_diagonal_corr": float(np.nanmean(off)) if off.size else float("nan"),
        "max_off_diagonal_corr": float(np.nanmax(off)) if off.size else float("nan"),
        "participation_ratio_empirical": pr,
        "theoretical_pairwise_covariance": 0.0,
        "theoretical_pairwise_correlation": 0.0,
        "M_eff_theoretical": float(m),
        "caveat": (f"n={errors.shape[0]} owners; unstable. Not evidence of independence "
                   "(that is closed-form: disjoint, independently seeded rollouts) and not "
                   "evidence that these sources estimate a common J_C (see target drift)."),
    }


# --------------------------------------------------------------------------
# Per-anchor analysis
# --------------------------------------------------------------------------

def analyse_anchor(data: dict, ckpt_dir: Path, logged_kl: dict) -> dict:
    meta, pools, agent_ids = data["meta"], data["pools"], data["agent_ids"]
    k = meta["anchor"]
    delta = meta["stride_delta"]
    role_round = {r: meta["policies"][r] for r in ROLES}

    # ---- per-policy per-owner reference and sigma_hat (step 2 / notation)
    pol_stats: dict[str, dict] = {}
    for role in ROLES:
        src = pools[f"{role}_src"]["mc_cost_return"]
        ref = pools[f"{role}_ref"]["mc_cost_return"]
        task_ref = pools[f"{role}_ref"]["mc_task_return"]
        both = np.concatenate([src, ref], axis=0)
        sigma = both.std(axis=0, ddof=1)
        pol_stats[role] = {
            "policy_round": role_round[role],
            "n_src": int(src.shape[0]), "n_ref": int(ref.shape[0]),
            "sigma_hat": {a: float(sigma[j]) for j, a in enumerate(agent_ids)},
            "ref_mean": {a: float(ref[:, j].mean()) for j, a in enumerate(agent_ids)},
            "ref_se": {a: float(sigma[j] / math.sqrt(ref.shape[0])) for j, a in enumerate(agent_ids)},
            "task_ref_mean": {a: float(task_ref[:, j].mean()) for j, a in enumerate(agent_ids)},
            "task_ref_se": {a: float(task_ref[:, j].std(ddof=1) / math.sqrt(task_ref.shape[0]))
                            for j, a in enumerate(agent_ids)},
        }

    # ---- policy distance at each stride (step 4)
    snaps = {role: load_policy(ckpt_dir / f"theta_{role_round[role]:03d}.pt", agent_ids) for role in ROLES}
    for role in ROLES:
        assert snaps[role]["round_index"] == role_round[role]
    strides = {"1": ("plus1", 1), "2": ("plus2", 2), "delta": ("plusdelta", delta)}
    distances = {}
    for label, (role, n_updates) in strides.items():
        d = policy_distance(snaps["anchor"], snaps[role], data["kl_states"], agent_ids)
        d["n_policy_updates"] = n_updates
        d["policy_round"] = role_round[role]
        # PPO's own approx_kl over the same span, from the reproduced
        # trajectory -- an independent measurement of the same step.
        d["ppo_approx_kl_sum_over_span"] = {
            a: float(sum(logged_kl[a][k:k + n_updates])) for a in agent_ids
        }
        d["ppo_approx_kl_mean_per_update"] = float(
            np.mean([np.mean(logged_kl[a][k:k + n_updates]) for a in agent_ids])
        )
        # ---- cost/task return change, per owner (step 4 items 3-4)
        d["delta_J_C"] = {}
        for a in agent_ids:
            dj = pol_stats[role]["ref_mean"][a] - pol_stats["anchor"]["ref_mean"][a]
            se = math.hypot(pol_stats[role]["ref_se"][a], pol_stats["anchor"]["ref_se"][a])
            se_m_anchor = pol_stats["anchor"]["sigma_hat"][a] / math.sqrt(R_M)
            r_drift = abs(dj) / se_m_anchor
            d["delta_J_C"][a] = {
                "delta": dj, "se": se, "ci95": [dj - 1.96 * se, dj + 1.96 * se],
                "abs_delta_frac_of_budget": abs(dj) / BUDGET_D,
                "se_m_one_source": se_m_anchor,
                "R_drift": r_drift,
                "band": ("negligible" if r_drift < DRIFT_BAND_NEGLIGIBLE
                         else "dominant" if r_drift >= DRIFT_BAND_DOMINANT else "material"),
                "R_drift_vs_aggregate_se_secondary": abs(dj) / (se_m_anchor / math.sqrt(M)),
                "delta_task_return": (pol_stats[role]["task_ref_mean"][a]
                                      - pol_stats["anchor"]["task_ref_mean"][a]),
            }
        distances[label] = d

    # ---- per-owner source conditions (step 3)
    anchor_src = pools["anchor_src"]["mc_cost_return"]
    blocks = np.array_split(np.arange(anchor_src.shape[0]), M)
    owner_results = []
    for j, aid in enumerate(agent_ids):
        ref_mean = pol_stats["anchor"]["ref_mean"][aid]
        se_ref = pol_stats["anchor"]["ref_se"][aid]
        sigma_anchor = pol_stats["anchor"]["sigma_hat"][aid]
        se_m_anchor = sigma_anchor / math.sqrt(R_M)

        # FROZEN: three disjoint 30-round blocks, all under theta_k
        frozen_sources = [
            source_row(float(anchor_src[b, j].mean()), len(b),
                       pol_stats["anchor"]["sigma_hat"][aid] / math.sqrt(len(b)),
                       ref_mean, se_ref, drift=0.0)
            for b in blocks
        ]

        # SEQ-1: source 1 IS frozen block 0 (same batch, same policy --
        # shared by construction so drift is the only difference)
        seq1_sources = [frozen_sources[0]]
        for role, label in (("plus1", "1"), ("plus2", "2")):
            vals = pools[f"{role}_src"]["mc_cost_return"][:, j]
            seq1_sources.append(source_row(
                float(vals.mean()), len(vals),
                pol_stats[role]["sigma_hat"][aid] / math.sqrt(len(vals)),
                ref_mean, se_ref, drift=distances[label]["delta_J_C"][aid]["delta"],
            ))

        # SEQ-Delta: source 1 = frozen block 0, source 2 at theta_{k+Delta}.
        # Source 3 would need theta_{k+2*Delta}, which does not exist for
        # every anchor inside the 250-round run -- extrapolated below,
        # never measured.
        vals_d = pools["plusdelta_src"]["mc_cost_return"][:, j]
        seqd_sources = [frozen_sources[0], source_row(
            float(vals_d.mean()), len(vals_d),
            pol_stats["plusdelta"]["sigma_hat"][aid] / math.sqrt(len(vals_d)),
            ref_mean, se_ref, drift=distances["delta"]["delta_J_C"][aid]["delta"],
        )]

        # Drift rate (step 8). Delta-stride is the best-conditioned
        # estimate; the 2-stride value is the cross-check.
        dj_delta = distances["delta"]["delta_J_C"][aid]["delta"]
        rho = abs(dj_delta) / delta
        rho_se = distances["delta"]["delta_J_C"][aid]["se"] / delta
        rho_from_2 = abs(distances["2"]["delta_J_C"][aid]["delta"]) / 2.0
        rho_from_2_se = distances["2"]["delta_J_C"][aid]["se"] / 2.0
        r_m_max = ((2 * sigma_anchor) / (rho * (M - 0.5))) ** (2 / 3) if rho > 0 else float("inf")

        # Step 7 criterion, per schedule.
        drift_seq1 = [0.0, abs(distances["1"]["delta_J_C"][aid]["delta"]),
                      abs(distances["2"]["delta_J_C"][aid]["delta"])]
        drift_seqd_measured = [0.0, abs(dj_delta)]
        drift_seqd_extrap_src3 = 2.0 * abs(dj_delta)  # local linearity, labelled extrapolation
        tol = Z_TOL * se_m_anchor

        # Each Delta J_C is itself estimated from two 120-round reference
        # pools, so it carries its own standard error. The GATE is the
        # pre-declared point-estimate comparison and is not touched; these
        # two extra readings are reported beside it so a marginal verdict
        # can be seen for what it is rather than read as exact.
        se_drift = {"1": distances["1"]["delta_J_C"][aid]["se"],
                    "2": distances["2"]["delta_J_C"][aid]["se"],
                    "delta": distances["delta"]["delta_J_C"][aid]["se"]}

        def _ci_reading(drifts: list[float], ses: list[float]) -> dict:
            hi = [abs(d) + 1.96 * s for d, s in zip(drifts, ses)]
            lo = [max(0.0, abs(d) - 1.96 * s) for d, s in zip(drifts, ses)]
            return {
                "strict_pass_ci_upper_within_tolerance": max(hi) <= tol,
                "weak_pass_cannot_reject_tolerance": max(lo) <= tol,
                "drift_ci_upper": hi, "drift_ci_lower": lo,
            }

        owner_results.append({
            "agent_id": aid,
            "sigma_hat_anchor": sigma_anchor,
            "se_m_one_source": se_m_anchor,
            "se_aggregate_M3": se_m_anchor / math.sqrt(M),
            "reference": {"mean": ref_mean, "se": se_ref, "r": R_REF,
                          "ci95": [ref_mean - 1.96 * se_ref, ref_mean + 1.96 * se_ref]},
            "FROZEN": condition_block(frozen_sources, ref_mean),
            "SEQ1": condition_block(seq1_sources, ref_mean),
            "SEQD": condition_block(seqd_sources, ref_mean),
            "drift_rate": {
                "rho_from_delta": rho, "rho_se": rho_se, "delta_used": delta,
                "rho_from_stride2": rho_from_2, "rho_from_stride2_se": rho_from_2_se,
                "consistent_within_2se": abs(rho - rho_from_2) <= 2 * math.hypot(rho_se, rho_from_2_se),
            },
            "R_m_max_sequential_M3": r_m_max,
            "step7": {
                "tolerance_2se": tol,
                "SEQ1": {"max_abs_drift": max(drift_seq1), "drifts": drift_seq1,
                         "pass": max(drift_seq1) <= tol,
                         **_ci_reading(drift_seq1, [0.0, se_drift["1"], se_drift["2"]])},
                "SEQD_measured_2_sources": {"max_abs_drift": max(drift_seqd_measured),
                                            "drifts": drift_seqd_measured,
                                            "pass": max(drift_seqd_measured) <= tol,
                                            **_ci_reading(drift_seqd_measured, [0.0, se_drift["delta"]])},
                "SEQD_with_extrapolated_source3": {
                    "max_abs_drift": drift_seqd_extrap_src3,
                    "drifts": drift_seqd_measured + [drift_seqd_extrap_src3],
                    "source3_is_extrapolated": True,
                    "pass": drift_seqd_extrap_src3 <= tol,
                    **_ci_reading(drift_seqd_measured + [drift_seqd_extrap_src3],
                                  [0.0, se_drift["delta"], 2 * se_drift["delta"]])},
            },
        })

    empirical = {c: pooled_cross_owner_empirical(owner_results, c) for c in ("FROZEN", "SEQ1", "SEQD")}

    return {
        "anchor": k, "phase": PHASE[k], "stride_delta": delta,
        "delta_is_reduced": delta != 30,
        "policy_stats": pol_stats,
        "policy_distance": distances,
        "owners": owner_results,
        "empirical_cross_owner": empirical,
        "collection_meta": {
            "n_rounds_total": meta["n_rounds_total"],
            "env_steps_total": meta["env_steps_total"],
            "mean_round_wall_clock_s": meta["mean_round_wall_clock_s"],
            "median_round_wall_clock_s": meta["median_round_wall_clock_s"],
            "wall_clock_s": meta["wall_clock_s"],
            "within_anchor_round_seed_disjoint": meta["within_anchor_round_seed_disjoint"],
        },
    }


# --------------------------------------------------------------------------
# Schedule / batch-size analysis (step 8, gate G8f)
# --------------------------------------------------------------------------

def schedule_analysis(anchor_results: list[dict], rollout_length: int, sec_per_round: float) -> dict:
    out = {"rollout_length": rollout_length, "sec_per_rollout": sec_per_round,
           "M": M, "z_tolerance": Z_TOL, "rows": []}
    for ar in anchor_results:
        for own in ar["owners"]:
            rho = own["drift_rate"]["rho_from_delta"]
            sigma = own["sigma_hat_anchor"]
            for r_m in SCHEDULE_RM:
                se_m = sigma / math.sqrt(r_m)
                drift_worst = rho * (M - 0.5) * r_m          # sequential, 1 rollout/round
                out["rows"].append({
                    "anchor": ar["anchor"], "phase": ar["phase"], "agent_id": own["agent_id"],
                    "R_m": r_m, "rho": rho, "sigma_hat": sigma, "se_m": se_m,
                    "sequential": {
                        "env_steps": M * r_m * rollout_length,
                        "training_rounds_consumed": M * r_m,
                        "wall_clock_s": M * r_m * sec_per_round,
                        "worst_source_drift": drift_worst,
                        "R_drift": drift_worst / se_m,
                        "pass_z2": drift_worst <= Z_TOL * se_m,
                    },
                    "parallel": {
                        "env_steps": M * r_m * rollout_length,
                        "training_rounds_stalled": r_m,
                        "wall_clock_s": r_m * sec_per_round,
                        "simulators_required": M,
                        "worst_source_drift": 0.0,
                        "R_drift": 0.0,
                        "pass_z2": True,
                    },
                })
    return out


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------

def compute_gates(anchor_results: list[dict], integrity: dict) -> dict:
    gates: dict = {}

    def per_anchor(pred) -> dict:
        rows, all_pass = {}, True
        for ar in anchor_results:
            n = sum(1 for o in ar["owners"] if pred(o))
            ok = n >= OWNERS_REQUIRED
            rows[str(ar["anchor"])] = {
                "phase": ar["phase"], "n_owners_pass": n, "of": len(ar["owners"]), "pass": ok,
                "failing_owners": [o["agent_id"] for o in ar["owners"] if not pred(o)],
            }
            all_pass &= ok
        return {"per_anchor": rows, "pass": all_pass}

    gates["G8a"] = per_anchor(lambda o: o["FROZEN"]["all_good"])
    gates["G8c_1"] = per_anchor(lambda o: o["step7"]["SEQ1"]["pass"])
    gates["G8c_delta"] = per_anchor(lambda o: o["step7"]["SEQD_with_extrapolated_source3"]["pass"])
    gates["G8c_delta_measured_only"] = per_anchor(
        lambda o: o["step7"]["SEQD_measured_2_sources"]["pass"])

    # G8b: every anchor reports every owner, in every condition.
    expected = {o["agent_id"] for o in anchor_results[0]["owners"]}
    g8b_ok = all(
        {o["agent_id"] for o in ar["owners"]} == expected and len(expected) == 6
        and all(c in o and o[c]["n_sources"] >= 2 for o in ar["owners"] for c in ("FROZEN", "SEQ1", "SEQD"))
        for ar in anchor_results
    )
    gates["G8b"] = {"pass": bool(g8b_ok), "n_owners": len(expected),
                    "note": "per-owner rows present for all 6 owners at every anchor and condition"}

    # G8d: early AND late anchor must hold for G8a and G8c-1.
    early, late = str(ANCHORS[0]), str(ANCHORS[-1])
    g8d = all(gates[g]["per_anchor"][a]["pass"] for g in ("G8a", "G8c_1") for a in (early, late))
    gates["G8d"] = {
        "pass": bool(g8d), "early_anchor": int(early), "late_anchor": int(late),
        "G8a_early": gates["G8a"]["per_anchor"][early]["pass"],
        "G8a_late": gates["G8a"]["per_anchor"][late]["pass"],
        "G8c_1_early": gates["G8c_1"]["per_anchor"][early]["pass"],
        "G8c_1_late": gates["G8c_1"]["per_anchor"][late]["pass"],
    }

    # G8e: (i) disjointness + closed-form zero covariance, (ii) structural
    # separation of sampling independence from target drift.
    g8e_i = integrity["round_seed_disjoint_all_pools"] and all(
        ar["collection_meta"]["within_anchor_round_seed_disjoint"] for ar in anchor_results)
    gates["G8e"] = {
        "pass": bool(g8e_i),
        "i_disjointness_and_zero_theoretical_covariance": bool(g8e_i),
        "ii_independence_reported_separately_from_drift": True,
        "note": ("theoretical pairwise source covariance is exactly 0 in every condition by the "
                 "disjoint-draw construction; the empirical n=6 correlation is reported beside it "
                 "and gates nothing, and target drift is reported as a separate quantity"),
    }

    gates["G8f"] = {"pass": True, "note": "schedule_analysis block reports env steps, training "
                                          "rounds, wall clock and induced drift for sequential and "
                                          "parallel collection at R_m in {1,5,10,30} plus R_m_max"}
    gates["G8g"] = {"pass": bool(integrity["all_pass"]), **integrity}
    return gates


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default="results/g8_nonstationary_source_diagnostic/dataset")
    ap.add_argument("--ckpt-dir", default="results/g8_nonstationary_source_diagnostic/checkpoints")
    ap.add_argument("--training-run", default="results/runs_constraint_mc_g2/pilot_A_clean_seed0")
    ap.add_argument("--out", default="results/g8_nonstationary_source_diagnostic/g8_report.json")
    args = ap.parse_args()

    dataset_dir, ckpt_dir = Path(args.dataset_dir), Path(args.ckpt_dir)
    repro = json.loads((ckpt_dir / "reproduction_check.json").read_text())

    rounds = [json.loads(l) for l in (Path(args.training_run) / "rounds.jsonl").read_text().splitlines()]
    agent_ids_train = list(rounds[0]["constraints"].keys())
    logged_kl = {a: [r["constraints"][a]["ppo"]["approx_kl"] for r in rounds] for a in agent_ids_train}

    data = {k: load_anchor(dataset_dir, k) for k in ANCHORS}

    # ---- G8g integrity, asserted here over ALL pools of ALL anchors
    all_seeds: dict[str, set] = {}
    for k, d in data.items():
        for name, pool in d["pools"].items():
            all_seeds[f"{k}:{name}"] = set(int(s) for s in pool["round_seeds"])
    names = list(all_seeds)
    collisions = [
        {"a": names[i], "b": names[j], "n_overlap": len(all_seeds[names[i]] & all_seeds[names[j]])}
        for i in range(len(names)) for j in range(i + 1, len(names))
        if all_seeds[names[i]] & all_seeds[names[j]]
    ]
    g7_seeds = set()
    g7_ds = Path("results/g7_estimand_source_diagnostic/dataset")
    if g7_ds.exists():
        for f in g7_ds.glob("*.npz"):
            g7_seeds |= set(int(s) for s in np.load(f, allow_pickle=True)["round_seeds"])
    g7_overlap = sorted({n for n in names if all_seeds[n] & g7_seeds})

    integrity = {
        "checkpoint_reproduction_bitwise_identical": bool(repro["bitwise_identical"]),
        "checkpoint_reproduction_comparisons": repro["n_comparisons"],
        "n_pools": len(names),
        "round_seed_disjoint_all_pools": not collisions,
        "collisions": collisions,
        "disjoint_from_g7_pools": not g7_overlap,
        "g7_overlapping_pools": g7_overlap,
        "estimator": ("every Jhat is numpy.mean of AgentRollout.finalize()['mc_cost_return'] "
                      "(discounted_window_return on reported_cost); no GAE, no ret_c, no critic "
                      "forward pass, no fitted head anywhere in G8"),
        "no_training_code_path": ("collection calls only env.reset/env.step and "
                                  "policy.distribution().sample(); no ppo_lagrangian_update, "
                                  "no compute_gae, no dual_update, no apply_attack, no aggregate, "
                                  "no refit in any G8 script"),
    }
    integrity["all_pass"] = bool(
        integrity["checkpoint_reproduction_bitwise_identical"]
        and integrity["round_seed_disjoint_all_pools"]
        and integrity["disjoint_from_g7_pools"]
    )
    print(f"[G8g] bitwise repro={integrity['checkpoint_reproduction_bitwise_identical']} "
          f"pools={integrity['n_pools']} disjoint={integrity['round_seed_disjoint_all_pools']} "
          f"disjoint_from_g7={integrity['disjoint_from_g7_pools']}")

    anchor_results = []
    for k in ANCHORS:
        print(f"\n=== anchor {k} ({PHASE[k]}), Delta={STRIDE[k]} ===")
        ar = analyse_anchor(data[k], ckpt_dir, logged_kl)
        anchor_results.append(ar)
        for lab in ("1", "2", "delta"):
            d = ar["policy_distance"][lab]
            print(f"  stride {lab:>5} ({d['n_policy_updates']:>2} updates): "
                  f"KL={d['kl_mean_over_agents']:.5f}  |dtheta|/|theta|={d['param_l2_rel_mean']:.4f}  "
                  f"mean |dJ_C|={np.mean([abs(v['delta']) for v in d['delta_J_C'].values()]):.2f}  "
                  f"mean R_drift={np.mean([v['R_drift'] for v in d['delta_J_C'].values()]):.2f}")
        for o in ar["owners"]:
            print(f"  {o['agent_id']}: sigma={o['sigma_hat_anchor']:.2f} SE_m={o['se_m_one_source']:.2f} "
                  f"ref={o['reference']['mean']:.2f} | FROZEN {o['FROZEN']['n_good']}/3 good "
                  f"| SEQ1 step7={o['step7']['SEQ1']['pass']} "
                  f"| SEQD step7={o['step7']['SEQD_with_extrapolated_source3']['pass']} "
                  f"| R_m_max={o['R_m_max_sequential_M3']:.1f}")

    # Median, not mean: this machine ran a Windows Update install during
    # part of the collection and the mean is contaminated by that stall.
    sec_per_round = float(np.median([ar["collection_meta"]["median_round_wall_clock_s"]
                                     for ar in anchor_results]))
    rollout_length = data[ANCHORS[0]]["meta"]["rollout_length"]
    sched = schedule_analysis(anchor_results, rollout_length, sec_per_round)
    gates = compute_gates(anchor_results, integrity)

    report = {
        "meta": {
            "anchors": ANCHORS, "phase": PHASE, "stride": STRIDE,
            "dataset_dir": str(dataset_dir), "ckpt_dir": str(ckpt_dir),
            "training_run": args.training_run,
            "rollout_length": rollout_length,
            "budget_d": BUDGET_D,
            "reproduction_check": repro,
        },
        "config": {
            "M": M, "R_m": R_M, "R_ref": R_REF, "z_good": Z_GOOD, "z_poor": Z_POOR,
            "z_tolerance_step7": Z_TOL, "owners_required": OWNERS_REQUIRED,
            "drift_band_negligible": DRIFT_BAND_NEGLIGIBLE, "drift_band_dominant": DRIFT_BAND_DOMINANT,
            "schedule_R_m": SCHEDULE_RM,
        },
        "integrity": integrity,
        "anchors": anchor_results,
        "schedule_analysis": sched,
        "gates": gates,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))

    print("\n=== G8 GATES ===")
    for g, v in gates.items():
        detail = ""
        if "per_anchor" in v:
            parts = [str(a) + ":" + str(row["pass"]) for a, row in v["per_anchor"].items()]
            detail = "  per-anchor={" + ", ".join(parts) + "}"
        print(f"  {g}: pass={v['pass']}" + detail)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
