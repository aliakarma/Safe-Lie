#!/usr/bin/env python
"""G4 (docs/g4_gates.md): identity-conditioned source estimator diagnostic.

Reads the SAME fixed dataset G3 collected
(`results/g3_source_diagnostic/dataset/agent_*.npz` --
`scripts/g3_collect_calibration_dataset.py`) and trains three small
regression variants on it:

  B1 -- separate per-agent normalized heads (positive control, same
        recipe as G3's normalized condition)
  B2 -- one identity-conditioned shared network h(obs, one_hot_identity)
  B3 -- one shared network h(obs) with no identity input (control)

B4 (the current peer_critic design) is NOT retrained here -- G3's already
-computed normalized V2 numbers are read from
`results/g3_source_diagnostic/g3_report.json` and reused verbatim.

No PPO, no GAE, no dual update, no environment call, no attack, no RCE.
Pure supervised regression on a fixed, disk-persisted dataset, split
rounds 0-39 (fit) / 40-59 (eval) exactly as G3 collected them.

Usage:
    .venv/Scripts/python.exe scripts/g4_identity_source_diagnostic.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from safelie.algos.normalization import RunningMeanStd

# --- pre-declared hyperparameters, docs/g4_gates.md "Model variants" ---
B1_HIDDEN = 16
B1_STEPS = 2500
B1_LR = 1e-3
B1_BATCH = 2048

SHARED_HIDDEN = 32
SHARED_STEPS = 5000
SHARED_LR = 1e-3
SHARED_BATCH = 2048

N_REPLICAS = 3
REPLICA_SEEDS = [7001, 7002, 7003]

BUDGET_D = 25.0
GOOD_BIAS = 0.20 * BUDGET_D  # 5.0
GOOD_CORR = 0.5
POOR_CORR = 0.2
FSR_ACCEPTABLE = 0.20


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class NormalizedHead:
    """`Linear(in_dim,hidden) -> Tanh -> Linear(hidden,1)`, Xavier-uniform
    init, Adam, full-batch-bootstrap training with input/target
    RunningMeanStd normalization -- shared machinery for B1 (per-agent
    fit) and B2/B3 (pooled fit), which differ only in `in_dim`, `hidden`,
    training steps/batch, and what gets fit (per-agent vs. pooled RMS)."""

    def __init__(self, in_dim: int, hidden: int, seed: int):
        gen = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for p in self.net.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, generator=gen)
            else:
                nn.init.zeros_(p)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=B1_LR)
        self.rng = np.random.default_rng(seed)

    def train_steps(self, feat: np.ndarray, target_n: np.ndarray, steps: int, batch: int, lr: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        feat_t = torch.as_tensor(feat, dtype=torch.float32)
        target_t = torch.as_tensor(target_n, dtype=torch.float32)
        n = len(feat)
        bs = min(batch, n)
        for _ in range(steps):
            idx = self.rng.integers(0, n, size=bs)
            pred = self.net(feat_t[idx]).squeeze(-1)
            loss = F.mse_loss(pred, target_t[idx])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict_raw(self, feat: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.net(torch.as_tensor(feat, dtype=torch.float32)).squeeze(-1).numpy()


def one_hot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v


def calibration_stats(pred: np.ndarray, truth: np.ndarray, d: float = BUDGET_D) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    err = pred - truth
    bias = float(np.mean(err))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    if np.std(pred) < 1e-12 or np.std(truth) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(pred, truth)[0, 1])
    unsafe_mask = truth > d
    if unsafe_mask.sum() == 0:
        fsr = float("nan")
    else:
        fsr = float(np.mean(pred[unsafe_mask] <= d))
    return {"n": int(len(pred)), "bias": bias, "mae": mae, "rmse": rmse, "corr": corr, "fsr": fsr,
            "n_unsafe": int(unsafe_mask.sum())}


def classify(stats: dict) -> str:
    corr = stats["corr"]
    bias = stats["bias"]
    if np.isnan(corr):
        return "undefined"
    if abs(bias) <= GOOD_BIAS and corr >= GOOD_CORR:
        return "good"
    if corr < POOR_CORR or abs(bias) > GOOD_BIAS:
        return "poor"
    return "mixed"


def pool(list_of_pred: list[np.ndarray], list_of_truth: list[np.ndarray]) -> dict:
    return calibration_stats(np.concatenate(list_of_pred), np.concatenate(list_of_truth))


def participation_ratio(cov: np.ndarray) -> float:
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    s1 = eig.sum()
    s2 = (eig**2).sum()
    if s2 <= 1e-18:
        return float("nan")
    return float((s1 * s1) / s2)


# --------------------------------------------------------------------------
# B1: separate per-agent normalized heads
# --------------------------------------------------------------------------

def fit_b1(data: dict, agent_ids: list[str], obs_dim: int) -> dict:
    heads = {}
    obs_rms = {}
    target_rms = {}
    for i, aid in enumerate(agent_ids):
        seed = 6000 + i  # identical seeds to G3's normalized condition
        h = NormalizedHead(obs_dim, B1_HIDDEN, seed=seed)
        rms_o = RunningMeanStd(shape=(obs_dim,))
        rms_t = RunningMeanStd(shape=())
        rms_o.update(data[aid]["fit_obs"])
        rms_t.update(data[aid]["fit_target"])
        obs_n = rms_o.normalize(torch.as_tensor(data[aid]["fit_obs"], dtype=torch.float32)).numpy()
        target_n = rms_t.normalize(torch.as_tensor(data[aid]["fit_target"], dtype=torch.float32)).numpy()
        h.train_steps(obs_n, target_n, B1_STEPS, B1_BATCH, B1_LR)
        heads[aid] = h
        obs_rms[aid] = rms_o
        target_rms[aid] = rms_t
    return {"heads": heads, "obs_rms": obs_rms, "target_rms": target_rms}


def predict_b1(b1: dict, agent_ids: list[str], data: dict) -> dict:
    """diagonal only: h_i(o^i) for each owner i."""
    pred = {}
    for aid in agent_ids:
        q = data[aid]["eval_query_obs"]
        q_n = b1["obs_rms"][aid].normalize(torch.as_tensor(q, dtype=torch.float32)).numpy()
        raw = b1["heads"][aid].predict_raw(q_n)
        pred[aid] = b1["target_rms"][aid].denormalize(torch.as_tensor(raw, dtype=torch.float32)).numpy()
    return pred


# --------------------------------------------------------------------------
# B2 / B3: pooled shared estimator, with / without identity
# --------------------------------------------------------------------------

def build_pooled_fit_arrays(data: dict, agent_ids: list[str], obs_dim: int, with_identity: bool):
    n_agents = len(agent_ids)
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    target_rms = RunningMeanStd(shape=())
    for aid in agent_ids:
        obs_rms.update(data[aid]["fit_obs"])
        target_rms.update(data[aid]["fit_target"])

    feats = []
    targets = []
    for i, aid in enumerate(agent_ids):
        obs = data[aid]["fit_obs"]
        obs_n = obs_rms.normalize(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        if with_identity:
            ident = np.tile(one_hot(i, n_agents), (obs_n.shape[0], 1))
            feat = np.concatenate([obs_n, ident], axis=1)
        else:
            feat = obs_n
        feats.append(feat)
        target_n = target_rms.normalize(torch.as_tensor(data[aid]["fit_target"], dtype=torch.float32)).numpy()
        targets.append(target_n)
    feat_all = np.concatenate(feats, axis=0).astype(np.float32)
    target_all = np.concatenate(targets, axis=0).astype(np.float32)
    return feat_all, target_all, obs_rms, target_rms


def fit_shared(data: dict, agent_ids: list[str], obs_dim: int, with_identity: bool, seed: int) -> dict:
    feat_all, target_all, obs_rms, target_rms = build_pooled_fit_arrays(data, agent_ids, obs_dim, with_identity)
    in_dim = obs_dim + (len(agent_ids) if with_identity else 0)
    net = NormalizedHead(in_dim, SHARED_HIDDEN, seed=seed)
    net.train_steps(feat_all, target_all, SHARED_STEPS, SHARED_BATCH, SHARED_LR)
    return {"net": net, "obs_rms": obs_rms, "target_rms": target_rms, "with_identity": with_identity}


def predict_shared(model: dict, agent_ids: list[str], data: dict, query_identity: int | None = None) -> np.ndarray:
    """`query_identity`: which agent's obs to score (index into agent_ids).
    Returns predictions for that agent's 20 eval rows, using the model's
    own identity axis if `with_identity`, one-hot(query_identity)."""
    aid = agent_ids[query_identity]
    obs = data[aid]["eval_query_obs"]
    obs_n = model["obs_rms"].normalize(torch.as_tensor(obs, dtype=torch.float32)).numpy()
    if model["with_identity"]:
        ident = np.tile(one_hot(query_identity, len(agent_ids)), (obs_n.shape[0], 1))
        feat = np.concatenate([obs_n, ident], axis=1)
    else:
        feat = obs_n
    raw = model["net"].predict_raw(feat)
    return model["target_rms"].denormalize(torch.as_tensor(raw, dtype=torch.float32)).numpy()


def predict_shared_matrix(model: dict, agent_ids: list[str], data: dict) -> dict:
    """Full 6x6: pred[owner_obs][queried_identity]. Only meaningful with
    identity; for B3 every column is identical (no identity axis) and the
    caller should only read the diagonal."""
    n = len(agent_ids)
    out = {agent_ids[i]: {} for i in range(n)}
    for i, owner in enumerate(agent_ids):
        obs = data[owner]["eval_query_obs"]
        obs_n = model["obs_rms"].normalize(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        for j, queried in enumerate(agent_ids):
            if model["with_identity"]:
                ident = np.tile(one_hot(j, n), (obs_n.shape[0], 1))
                feat = np.concatenate([obs_n, ident], axis=1)
            else:
                feat = obs_n
            raw = model["net"].predict_raw(feat)
            out[owner][queried] = model["target_rms"].denormalize(torch.as_tensor(raw, dtype=torch.float32)).numpy()
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default="results/g3_source_diagnostic/dataset")
    ap.add_argument("--g3-report", default="results/g3_source_diagnostic/g3_report.json")
    ap.add_argument("--out", default="results/g4_identity_source_diagnostic/g4_report.json")
    ap.add_argument("--seed", type=int, default=9000, help="B2/B3 main-model seed")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    meta = json.loads((dataset_dir / "meta.json").read_text())
    agent_ids: list[str] = meta["agent_ids"]
    obs_dim = int(meta["obs_dim"])
    n_agents = len(agent_ids)

    data = {aid: np.load(dataset_dir / f"{aid}.npz") for aid in agent_ids}
    for aid in agent_ids:
        print(f"{aid}: fit_rows={data[aid]['fit_obs'].shape[0]}, eval_rounds={data[aid]['eval_truth'].shape[0]}")

    g3_report = json.loads(Path(args.g3_report).read_text())
    b4_v2_normalized = g3_report["conditions"]["normalized"]

    truth = {aid: data[aid]["eval_truth"] for aid in agent_ids}

    report: dict = {
        "meta": {**meta, "g3_dataset_reused": str(dataset_dir), "g3_report_reused": args.g3_report},
        "hyperparameters": {
            "B1": {"hidden": B1_HIDDEN, "steps": B1_STEPS, "lr": B1_LR, "batch": B1_BATCH,
                   "architecture": "Linear(63,16)-Tanh-Linear(16,1)", "normalization": "per-agent RunningMeanStd"},
            "B2_B3": {"hidden": SHARED_HIDDEN, "steps": SHARED_STEPS, "lr": SHARED_LR, "batch": SHARED_BATCH,
                      "architecture_B2": "Linear(69,32)-Tanh-Linear(32,1) [obs 63 + one-hot identity 6]",
                      "architecture_B3": "Linear(63,32)-Tanh-Linear(32,1) [obs only]",
                      "normalization": "pooled RunningMeanStd across all 6 agents' fit rows"},
            "optimizer": "Adam", "init": "xavier_uniform (weights), zeros (bias)",
        },
        "train_eval_split": {"fit_rounds": "0-39", "eval_rounds": "40-59", "n_eval_rounds_per_agent": 20},
    }

    # ---------------- B1: separate normalized heads ----------------
    print("\n=== Fitting B1 (separate normalized owner heads) ===")
    b1 = fit_b1(data, agent_ids, obs_dim)
    b1_pred = predict_b1(b1, agent_ids, data)
    b1_stats_per_agent = {aid: calibration_stats(b1_pred[aid], truth[aid]) for aid in agent_ids}
    b1_pooled = pool([b1_pred[a] for a in agent_ids], [truth[a] for a in agent_ids])
    print(f"  B1 pooled: bias={b1_pooled['bias']:.3f} MAE={b1_pooled['mae']:.3f} "
          f"RMSE={b1_pooled['rmse']:.3f} corr={b1_pooled['corr']:.3f} [{classify(b1_pooled)}]")

    # ---------------- B2: identity-conditioned shared ----------------
    print("\n=== Fitting B2 (identity-conditioned shared estimator) ===")
    b2 = fit_shared(data, agent_ids, obs_dim, with_identity=True, seed=args.seed)
    b2_matrix_pred = predict_shared_matrix(b2, agent_ids, data)  # [owner_obs][queried_identity]
    b2_diag_pred = {aid: b2_matrix_pred[aid][aid] for aid in agent_ids}
    b2_stats_per_agent = {aid: calibration_stats(b2_diag_pred[aid], truth[aid]) for aid in agent_ids}
    b2_pooled = pool([b2_diag_pred[a] for a in agent_ids], [truth[a] for a in agent_ids])
    print(f"  B2 diagonal pooled: bias={b2_pooled['bias']:.3f} MAE={b2_pooled['mae']:.3f} "
          f"RMSE={b2_pooled['rmse']:.3f} corr={b2_pooled['corr']:.3f} [{classify(b2_pooled)}]")

    b2_bias_matrix = {q: {s: calibration_stats(b2_matrix_pred[q][s], truth[q])["bias"] for s in agent_ids} for q in agent_ids}
    b2_mae_matrix = {q: {s: calibration_stats(b2_matrix_pred[q][s], truth[q])["mae"] for s in agent_ids} for q in agent_ids}
    b2_rmse_matrix = {q: {s: calibration_stats(b2_matrix_pred[q][s], truth[q])["rmse"] for s in agent_ids} for q in agent_ids}
    b2_corr_matrix = {q: {s: calibration_stats(b2_matrix_pred[q][s], truth[q])["corr"] for s in agent_ids} for q in agent_ids}
    b2_full_matrix = {q: {s: calibration_stats(b2_matrix_pred[q][s], truth[q]) for s in agent_ids} for q in agent_ids}

    # ---------------- B3: shared, no identity ----------------
    print("\n=== Fitting B3 (shared estimator, no identity control) ===")
    b3 = fit_shared(data, agent_ids, obs_dim, with_identity=False, seed=args.seed)
    b3_pred = {aid: predict_shared(b3, agent_ids, data, query_identity=i) for i, aid in enumerate(agent_ids)}
    b3_stats_per_agent = {aid: calibration_stats(b3_pred[aid], truth[aid]) for aid in agent_ids}
    b3_pooled = pool([b3_pred[a] for a in agent_ids], [truth[a] for a in agent_ids])
    print(f"  B3 pooled: bias={b3_pooled['bias']:.3f} MAE={b3_pooled['mae']:.3f} "
          f"RMSE={b3_pooled['rmse']:.3f} corr={b3_pooled['corr']:.3f} [{classify(b3_pooled)}]")

    report["B1_separate_heads"] = {"pooled": b1_pooled, "pooled_classification": classify(b1_pooled),
                                    "per_agent": b1_stats_per_agent}
    report["B2_identity_conditioned"] = {
        "diagonal_pooled": b2_pooled, "diagonal_pooled_classification": classify(b2_pooled),
        "diagonal_per_agent": b2_stats_per_agent,
        "identity_query_matrix": {"bias": b2_bias_matrix, "mae": b2_mae_matrix, "rmse": b2_rmse_matrix,
                                   "corr": b2_corr_matrix, "full": b2_full_matrix},
    }
    report["B3_no_identity"] = {"pooled": b3_pooled, "pooled_classification": classify(b3_pooled),
                                 "per_agent": b3_stats_per_agent}
    report["B4_peer_critic_reused_from_g3"] = {
        "diagonal_pooled_note": "B4 diagonal == G3 normalized V1_V3 (in-distribution); off-diagonal == G3 normalized V2",
        "V1_V3_pooled": b4_v2_normalized["V1_V3_pooled"],
        "V1_V3_per_agent": b4_v2_normalized["V1_V3_per_agent"],
        "V2_pooled": b4_v2_normalized["V2_pooled"],
        "V2_per_pair": b4_v2_normalized["V2_per_pair"],
    }

    # ---------------- Identity necessity (B2 vs B3), per agent ----------------
    identity_necessity = {}
    for aid in agent_ids:
        b2s, b3s = b2_stats_per_agent[aid], b3_stats_per_agent[aid]
        bias_ratio = abs(b3s["bias"]) / max(abs(b2s["bias"]), 1e-9)
        mae_ratio = b3s["mae"] / max(b2s["mae"], 1e-9)
        identity_necessity[aid] = {
            "B2_classification": classify(b2s), "B3_classification": classify(b3s),
            "B3_bias_over_B2_bias": bias_ratio, "B3_mae_over_B2_mae": mae_ratio,
            "flips_from_good": classify(b2s) == "good" and classify(b3s) != "good",
        }
    pooled_bias_ratio = abs(b3_pooled["bias"]) / max(abs(b2_pooled["bias"]), 1e-9)
    pooled_mae_ratio = b3_pooled["mae"] / max(b2_pooled["mae"], 1e-9)
    identity_materially_necessary = (
        any(v["flips_from_good"] for v in identity_necessity.values())
        or pooled_bias_ratio >= 1.5 or pooled_mae_ratio >= 1.5
    )
    report["identity_necessity"] = {
        "per_agent": identity_necessity,
        "pooled_bias_ratio_B3_over_B2": pooled_bias_ratio,
        "pooled_mae_ratio_B3_over_B2": pooled_mae_ratio,
        "identity_materially_necessary": identity_materially_necessary,
    }

    # ---------------- G3 vs G4 comparison ----------------
    g3_v2_per_agent_diag_equiv = {aid: b4_v2_normalized["V1_V3_per_agent"][aid] for aid in agent_ids}
    report["g3_vs_g4_comparison"] = {
        "g3_cross_agent_peer_V2_pooled": b4_v2_normalized["V2_pooled"],
        "g4_identity_conditioned_diagonal_pooled": b2_pooled,
        "g3_in_distribution_V1_V3_pooled_reference": b4_v2_normalized["V1_V3_pooled"],
    }

    # ---------------- Source diversity: residual correlation for B2 ----------------
    residuals = np.stack([b2_diag_pred[aid] - truth[aid] for aid in agent_ids], axis=0)  # (6, 20)
    resid_corr = np.corrcoef(residuals)
    resid_cov = np.cov(residuals)
    pr = participation_ratio(resid_cov)
    report["source_diversity_B2"] = {
        "residual_correlation_matrix": {agent_ids[i]: {agent_ids[j]: float(resid_corr[i, j]) for j in range(n_agents)} for i in range(n_agents)},
        "residual_covariance_matrix": {agent_ids[i]: {agent_ids[j]: float(resid_cov[i, j]) for j in range(n_agents)} for i in range(n_agents)},
        "participation_ratio": pr,
        "max_possible_participation_ratio": n_agents,
        "structural_classification": "M_model = 1 (one shared parameterization; see docs/g4_gates.md source-diversity protocol -- PR/correlation are descriptive, not a re-derivation of M)",
    }
    print(f"\n  B2 residual participation ratio: {pr:.3f} (max possible {n_agents})")

    # ---------------- Independent replicas (B2 x 3 seeds) ----------------
    print("\n=== Fitting 3 independent B2 replicas (initialization-seed diversity only) ===")
    replica_diag_pred = []
    for k, rseed in enumerate(REPLICA_SEEDS):
        rep = fit_shared(data, agent_ids, obs_dim, with_identity=True, seed=rseed)
        rep_matrix = predict_shared_matrix(rep, agent_ids, data)
        rep_diag = {aid: rep_matrix[aid][aid] for aid in agent_ids}
        replica_diag_pred.append(rep_diag)
        rep_pooled = pool([rep_diag[a] for a in agent_ids], [truth[a] for a in agent_ids])
        print(f"  replica {k} (seed={rseed}) pooled: bias={rep_pooled['bias']:.3f} MAE={rep_pooled['mae']:.3f} "
              f"corr={rep_pooled['corr']:.3f} [{classify(rep_pooled)}]")

    replica_residuals = [np.stack([replica_diag_pred[k][aid] - truth[aid] for aid in agent_ids], axis=0).ravel() for k in range(N_REPLICAS)]
    replica_pairwise_corr = {}
    for a in range(N_REPLICAS):
        for b in range(a + 1, N_REPLICAS):
            c = float(np.corrcoef(replica_residuals[a], replica_residuals[b])[0, 1])
            replica_pairwise_corr[f"replica_{a}_vs_replica_{b}"] = c
    replica_stack = np.stack(replica_residuals, axis=0)  # (3, 120)
    replica_cov = np.cov(replica_stack)
    replica_pr = participation_ratio(replica_cov)
    report["independent_replicas_B2"] = {
        "seeds": REPLICA_SEEDS,
        "per_replica_pooled": [pool([replica_diag_pred[k][a] for a in agent_ids], [truth[a] for a in agent_ids]) for k in range(N_REPLICAS)],
        "pairwise_residual_correlation": replica_pairwise_corr,
        "participation_ratio": replica_pr,
        "max_possible_participation_ratio": N_REPLICAS,
        "note": "initialization-seed diversity only -- same architecture, same pooled training data, same recipe; not independent data or independent training procedure (docs/g4_gates.md, Independent-replica protocol)",
    }
    print(f"  replica pairwise residual correlations: {replica_pairwise_corr}")
    print(f"  replica participation ratio: {replica_pr:.3f} (max possible {N_REPLICAS})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, allow_nan=True))
    print(f"\nFull report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
