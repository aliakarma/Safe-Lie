#!/usr/bin/env python
"""G5 (docs/g5_gates.md): owner-specific replica diversity diagnostic.

Reads the SAME fixed dataset G3 collected
(`results/g3_source_diagnostic/dataset/agent_*.npz` --
`scripts/g3_collect_calibration_dataset.py`) and trains ONLY owner-specific
B1-style heads (`Linear(63,16)-Tanh-Linear(16,1)`, the one estimator family
G3/G4 validated as calibrated in-distribution):

  B1 -- one head per owner, full 61,640-row fit set (positive control,
        byte-identical recipe to G3's normalized condition / G4's B1)
  R1 -- k independent replicas per owner, each trained on an independent
        bootstrap resample (with replacement, same size) of that owner's
        fit rows
  R2 -- k independent replicas per owner, each trained on a disjoint
        contiguous round-block of that owner's fit rows (no bootstrap
        resample on top, no cross-block leakage)

for k in {3, 5}. No peer heads, no identity-conditioned input, no pooled/
shared network anywhere in this script. No PPO, no GAE, no dual update, no
environment call, no attack, no RCE. Pure supervised regression on a fixed,
disk-persisted dataset, split rounds 0-39 (fit) / 40-59 (eval) exactly as
G3 collected them.

Usage:
    .venv/Scripts/python.exe scripts/g5_source_diversity_diagnostic.py
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

# --- pre-declared hyperparameters, docs/g5_gates.md ---
HIDDEN = 16
STEPS = 2500
LR = 1e-3
BATCH = 2048

K_VALUES = [3, 5]
R1_SEED_BASE = {3: 21000, 5: 22000}
R2_SEED_BASE = {3: 23000, 5: 24000}
BOOTSTRAP_RESAMPLE_OFFSET = 1_000_000

BUDGET_D = 25.0
GOOD_BIAS = 0.20 * BUDGET_D  # 5.0
GOOD_CORR = 0.5
POOR_CORR = 0.2
FSR_ACCEPTABLE = 0.20

G4_REPLICA_REPORT = "results/g4_identity_source_diagnostic/g4_report.json"


# --------------------------------------------------------------------------
# Model (identical to G3's normalized condition / G4's B1)
# --------------------------------------------------------------------------

class NormalizedHead:
    def __init__(self, obs_dim: int, seed: int):
        gen = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, 1),
        )
        for p in self.net.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, generator=gen)
            else:
                nn.init.zeros_(p)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.rng = np.random.default_rng(seed)

    def train_steps(self, obs_n: np.ndarray, target_n: np.ndarray) -> bool:
        """Returns True iff every logged loss was finite (G5f)."""
        obs_t = torch.as_tensor(obs_n, dtype=torch.float32)
        target_t = torch.as_tensor(target_n, dtype=torch.float32)
        n = len(obs_n)
        bs = min(BATCH, n)
        finite = True
        for _ in range(STEPS):
            idx = self.rng.integers(0, n, size=bs)
            pred = self.net(obs_t[idx]).squeeze(-1)
            loss = F.mse_loss(pred, target_t[idx])
            if not torch.isfinite(loss):
                finite = False
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return finite

    def predict_raw(self, obs_n: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.net(torch.as_tensor(obs_n, dtype=torch.float32)).squeeze(-1).numpy()


def fit_owner_head(obs: np.ndarray, target: np.ndarray, query_obs: np.ndarray, seed: int) -> tuple[np.ndarray, bool]:
    """Fits one NormalizedHead on (obs, target) with its own RunningMeanStd,
    returns denormalized predictions on query_obs and a stability flag."""
    obs_rms = RunningMeanStd(shape=(obs.shape[1],))
    target_rms = RunningMeanStd(shape=())
    obs_rms.update(obs)
    target_rms.update(target)
    obs_n = obs_rms.normalize(torch.as_tensor(obs, dtype=torch.float32)).numpy()
    target_n = target_rms.normalize(torch.as_tensor(target, dtype=torch.float32)).numpy()
    head = NormalizedHead(obs.shape[1], seed=seed)
    finite = head.train_steps(obs_n, target_n)
    q_n = obs_rms.normalize(torch.as_tensor(query_obs, dtype=torch.float32)).numpy()
    raw = head.predict_raw(q_n)
    pred = target_rms.denormalize(torch.as_tensor(raw, dtype=torch.float32)).numpy()
    finite = finite and bool(np.all(np.isfinite(pred)))
    return pred, finite


# --------------------------------------------------------------------------
# Metrics (identical definitions to G3/G4)
# --------------------------------------------------------------------------

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


def diversity_block(error_vectors: list[np.ndarray]) -> dict:
    """error_vectors: list of k 1-D arrays (same length, aligned samples).
    Returns pairwise corr matrix, cov matrix, eigenvalues, PR, M_eff."""
    k = len(error_vectors)
    stack = np.stack(error_vectors, axis=0)
    corr = np.corrcoef(stack) if k > 1 else np.array([[1.0]])
    cov = np.cov(stack) if k > 1 else np.array([[float(np.var(stack))]])
    cov = np.atleast_2d(cov)
    corr = np.atleast_2d(corr)
    eig = np.clip(np.linalg.eigvalsh(cov), 0.0, None).tolist()
    pr = participation_ratio(cov)
    off_diag = corr[np.triu_indices(k, k=1)] if k > 1 else np.array([])
    return {
        "k": k,
        "correlation_matrix": corr.tolist(),
        "covariance_matrix": cov.tolist(),
        "eigenvalues": eig,
        "participation_ratio": pr,
        "m_eff": min(k, int(round(pr))) if not np.isnan(pr) else None,
        "mean_pairwise_corr": float(np.mean(off_diag)) if off_diag.size else float("nan"),
        "max_pairwise_corr": float(np.max(off_diag)) if off_diag.size else float("nan"),
        "min_pairwise_corr": float(np.min(off_diag)) if off_diag.size else float("nan"),
    }


# --------------------------------------------------------------------------
# Replica construction
# --------------------------------------------------------------------------

def bootstrap_resample(obs: np.ndarray, target: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + BOOTSTRAP_RESAMPLE_OFFSET)
    idx = rng.integers(0, len(obs), size=len(obs))
    return obs[idx], target[idx]


def round_blocks(obs: np.ndarray, target: np.ndarray, n_rounds: int, rows_per_round: int, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    round_groups = np.array_split(np.arange(n_rounds), k)
    blocks = []
    for grp in round_groups:
        row_idx = np.concatenate([np.arange(r * rows_per_round, (r + 1) * rows_per_round) for r in grp])
        blocks.append((obs[row_idx], target[row_idx]))
    return blocks


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default="results/g3_source_diagnostic/dataset")
    ap.add_argument("--g4-report", default=G4_REPLICA_REPORT)
    ap.add_argument("--out", default="results/g5_source_diagnostic/g5_report.json")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    meta = json.loads((dataset_dir / "meta.json").read_text())
    agent_ids: list[str] = meta["agent_ids"]
    n_agents = len(agent_ids)
    n_fit_rounds = int(meta["n_fit_rounds"])

    data = {aid: np.load(dataset_dir / f"{aid}.npz") for aid in agent_ids}
    for aid in agent_ids:
        print(f"{aid}: fit_rows={data[aid]['fit_obs'].shape[0]}, eval_rounds={data[aid]['eval_truth'].shape[0]}")

    rows_per_round = data[agent_ids[0]]["fit_obs"].shape[0] // n_fit_rounds
    assert rows_per_round * n_fit_rounds == data[agent_ids[0]]["fit_obs"].shape[0], \
        "fit rows do not divide evenly by round count -- round-block split (R2) assumption violated"
    for aid in agent_ids:
        assert data[aid]["fit_obs"].shape[0] == rows_per_round * n_fit_rounds

    truth = {aid: data[aid]["eval_truth"] for aid in agent_ids}
    stability_failures: list[str] = []

    report: dict = {
        "meta": {**meta, "g3_dataset_reused": str(dataset_dir), "rows_per_round": rows_per_round},
        "hyperparameters": {
            "architecture": "Linear(63,16)-Tanh-Linear(16,1)", "steps": STEPS, "lr": LR, "batch": BATCH,
            "optimizer": "Adam", "init": "xavier_uniform (weights), zeros (bias)",
            "normalization": "RunningMeanStd on obs+target, fit on that replica's OWN training rows",
            "k_values": K_VALUES, "r1_seed_base": R1_SEED_BASE, "r2_seed_base": R2_SEED_BASE,
        },
        "train_eval_split": {"fit_rounds": "0-39", "eval_rounds": "40-59", "n_eval_rounds_per_agent": 20},
    }

    # ---------------- B1: full-data owner-specific control ----------------
    print("\n=== Fitting B1 (full-data owner-specific control) ===")
    b1_pred: dict[str, np.ndarray] = {}
    for i, aid in enumerate(agent_ids):
        seed = 6000 + i
        pred, finite = fit_owner_head(data[aid]["fit_obs"], data[aid]["fit_target"], data[aid]["eval_query_obs"], seed)
        b1_pred[aid] = pred
        if not finite:
            stability_failures.append(f"B1/{aid}")
        s = calibration_stats(pred, truth[aid])
        print(f"  {aid} (seed={seed}): bias={s['bias']:.3f} MAE={s['mae']:.3f} corr={s['corr']:.3f} [{classify(s)}]")
    b1_stats_per_agent = {aid: calibration_stats(b1_pred[aid], truth[aid]) for aid in agent_ids}
    b1_pooled = pool([b1_pred[a] for a in agent_ids], [truth[a] for a in agent_ids])
    report["B1_control"] = {
        "per_agent": b1_stats_per_agent,
        "per_agent_classification": {aid: classify(b1_stats_per_agent[aid]) for aid in agent_ids},
        "pooled": b1_pooled, "pooled_classification": classify(b1_pooled),
    }

    def run_condition(name: str, k: int, seed_base: int, builder) -> dict:
        """builder(aid, i, k) -> list of k (obs, target) arrays for that owner's k replicas."""
        print(f"\n=== Fitting {name} (k={k}) ===")
        per_owner_pred: dict[str, list[np.ndarray]] = {aid: [] for aid in agent_ids}
        for i, aid in enumerate(agent_ids):
            replica_sets = builder(aid, i, k)
            for j, (obs_j, target_j) in enumerate(replica_sets):
                seed = seed_base + i * 10 + j
                pred, finite = fit_owner_head(obs_j, target_j, data[aid]["eval_query_obs"], seed)
                per_owner_pred[aid].append(pred)
                if not finite:
                    stability_failures.append(f"{name}-k{k}/{aid}/replica{j}")
                s = calibration_stats(pred, truth[aid])
                print(f"  {aid} replica {j} (seed={seed}, n_train={len(obs_j)}): "
                      f"bias={s['bias']:.3f} MAE={s['mae']:.3f} corr={s['corr']:.3f} [{classify(s)}]")

        per_owner_stats = {
            aid: [calibration_stats(per_owner_pred[aid][j], truth[aid]) for j in range(k)]
            for aid in agent_ids
        }
        per_owner_classification = {
            aid: [classify(s) for s in per_owner_stats[aid]] for aid in agent_ids
        }
        pooled_per_replica = [
            pool([per_owner_pred[aid][j] for aid in agent_ids], [truth[aid] for aid in agent_ids])
            for j in range(k)
        ]

        per_owner_diversity = {}
        for aid in agent_ids:
            errs = [per_owner_pred[aid][j] - truth[aid] for j in range(k)]
            per_owner_diversity[aid] = diversity_block(errs)

        pooled_errs = [
            np.concatenate([per_owner_pred[aid][j] - truth[aid] for aid in agent_ids])
            for j in range(k)
        ]
        pooled_diversity = diversity_block(pooled_errs)

        n_good_owners = sum(
            1 for aid in agent_ids if all(c == "good" for c in per_owner_classification[aid])
        )

        return {
            "k": k,
            "seed_base": seed_base,
            "per_owner_stats": per_owner_stats,
            "per_owner_classification": per_owner_classification,
            "pooled_per_replica": pooled_per_replica,
            "pooled_per_replica_classification": [classify(s) for s in pooled_per_replica],
            "n_owners_all_replicas_good": n_good_owners,
            "n_owners_total": n_agents,
            "g5a_pass": n_good_owners >= 5,
            "per_owner_diversity": per_owner_diversity,
            "pooled_diversity": pooled_diversity,
        }

    def r1_builder(aid: str, i: int, k: int):
        seed_base = R1_SEED_BASE[k]
        out = []
        for j in range(k):
            seed = seed_base + i * 10 + j
            obs_j, target_j = bootstrap_resample(data[aid]["fit_obs"], data[aid]["fit_target"], seed)
            out.append((obs_j, target_j))
        return out

    def r2_builder(aid: str, i: int, k: int):
        blocks = round_blocks(data[aid]["fit_obs"], data[aid]["fit_target"], n_fit_rounds, rows_per_round, k)
        return blocks

    for k in K_VALUES:
        report[f"R1_bootstrap_k{k}"] = run_condition("R1-bootstrap", k, R1_SEED_BASE[k], r1_builder)
    for k in K_VALUES:
        report[f"R2_disjoint_k{k}"] = run_condition("R2-disjoint", k, R2_SEED_BASE[k], r2_builder)

    # ---------------- G4 reference (read, not retrained) ----------------
    g4_report = json.loads(Path(args.g4_report).read_text())
    g4_replicas = g4_report["independent_replicas_B2"]
    report["g4_initialization_only_reference"] = {
        "source": args.g4_report,
        "architecture": "shared identity-conditioned B2 (NOT owner-specific -- architecturally different from every G5 replica)",
        "pairwise_residual_correlation": g4_replicas["pairwise_residual_correlation"],
        "participation_ratio": g4_replicas["participation_ratio"],
        "max_possible_participation_ratio": g4_replicas["max_possible_participation_ratio"],
    }
    g4_max_corr = max(g4_replicas["pairwise_residual_correlation"].values())
    g4_min_corr = min(g4_replicas["pairwise_residual_correlation"].values())

    # ---------------- G5d/G5e gate evaluation ----------------
    gate_results = {}
    for cond in ["R1_bootstrap_k3", "R1_bootstrap_k5", "R2_disjoint_k3", "R2_disjoint_k5"]:
        pd = report[cond]["pooled_diversity"]
        max_corr = pd["max_pairwise_corr"]
        g5d_pass = (not np.isnan(max_corr)) and max_corr <= 0.90
        g5d_materially_better = (not np.isnan(max_corr)) and max_corr <= 0.70
        gate_results[cond] = {
            "g5a_pass": report[cond]["g5a_pass"],
            "n_owners_all_replicas_good": report[cond]["n_owners_all_replicas_good"],
            "pooled_max_pairwise_corr": max_corr,
            "pooled_participation_ratio": pd["participation_ratio"],
            "g5d_pass": g5d_pass,
            "g5d_materially_better": g5d_materially_better,
            "g5e_pass": (report[cond]["k"] == 3) and report[cond]["g5a_pass"] and g5d_pass,
        }
        if report[cond]["k"] == 3:
            if gate_results[cond]["g5a_pass"] and g5d_materially_better:
                outcome = "A"
            elif gate_results[cond]["g5a_pass"] and not g5d_pass:
                outcome = "B"
            elif not gate_results[cond]["g5a_pass"] and g5d_pass:
                outcome = "C"
            else:
                outcome = "D" if not gate_results[cond]["g5a_pass"] else "A_marginal"
            gate_results[cond]["outcome"] = outcome

    report["g4_reference_summary"] = {"max_pairwise_corr": g4_max_corr, "min_pairwise_corr": g4_min_corr}
    report["gate_results"] = gate_results

    # ---------------- G5f stability ----------------
    n_heads_expected = n_agents * (1 + sum(K_VALUES) * 2)  # B1 + (R1+R2) x k in {3,5}
    report["stability"] = {
        "n_heads_expected": n_heads_expected,
        "n_stability_failures": len(stability_failures),
        "stability_failures": stability_failures,
        "g5f_pass": len(stability_failures) == 0,
    }

    # ---------------- accuracy-diversity tradeoff (R2 vs B1) ----------------
    tradeoff = {}
    for k in K_VALUES:
        cond = report[f"R2_disjoint_k{k}"]
        degraded = []
        for aid in agent_ids:
            b1_class = report["B1_control"]["per_agent_classification"][aid]
            r2_classes = cond["per_owner_classification"][aid]
            if b1_class == "good" and any(c != "good" for c in r2_classes):
                degraded.append(aid)
        tradeoff[f"k{k}"] = {"degraded_owners": degraded, "tradeoff_observed": len(degraded) > 0}
    report["accuracy_diversity_tradeoff_R2_vs_B1"] = tradeoff

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, allow_nan=True))
    print(f"\nFull report written to {out_path}")

    print("\n=== Gate summary ===")
    for cond, g in gate_results.items():
        print(f"  {cond}: g5a={g['g5a_pass']} ({g['n_owners_all_replicas_good']}/6 owners), "
              f"pooled_max_corr={g['pooled_max_pairwise_corr']:.4f}, g5d={g['g5d_pass']}, "
              f"materially_better={g['g5d_materially_better']}")
    print(f"  G4 reference max pairwise corr: {g4_max_corr:.4f} (range {g4_min_corr:.4f}-{g4_max_corr:.4f})")
    print(f"  Stability: {report['stability']['n_stability_failures']} failures out of {n_heads_expected} heads")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
