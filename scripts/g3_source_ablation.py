#!/usr/bin/env python
"""G3 (docs/g3_gates.md): the factorial source-estimator ablation.

Reads the ONE fixed dataset `scripts/g3_collect_calibration_dataset.py`
wrote (never collects its own data), trains one small regression head per
physical agent under two normalization conditions (raw / normalized), and
reads V1-V4 plus the full 6x6 owner x source matrix off that single set of
12 heads -- see docs/g3_gates.md "Matrix construction and its relation to
V1-V4" for why one set of heads suffices for all four variants.

No PPO, no GAE, no dual update, no environment call, no attack, no RCE.
Pure supervised regression on a fixed, disk-persisted dataset.

Usage:
    .venv/Scripts/python.exe scripts/g3_source_ablation.py
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

# --- pre-declared hyperparameters, docs/g3_gates.md "Model variants" ---
HIDDEN_DIM = 16
STEPS = 2500
LR = 1e-3
BATCH_SIZE = 2048
BUDGET_D = 25.0
GOOD_BIAS = 0.20 * BUDGET_D  # 5.0
GOOD_CORR = 0.5
POOR_CORR = 0.2
FSR_ACCEPTABLE = 0.20


class G3Head:
    """`Linear(obs_dim,16) -> Tanh -> Linear(16,1)`, Xavier-uniform init,
    Adam -- the same architecture as `safelie.sources.estimators.
    DiversifiedReplica`, kept fixed across both normalization conditions
    (docs/g3_gates.md: capacity is not the axis under test here). Only the
    training budget (steps/lr/batch) and the optional normalization wrap
    differ from the deployed `DiversifiedReplica`, per the task's
    instruction to rule out "insufficient training steps" before asking
    about the cross-agent query."""

    def __init__(self, obs_dim: int, seed: int, normalize: bool):
        gen = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        for p in self.net.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, generator=gen)
            else:
                nn.init.zeros_(p)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.normalize = normalize
        self.obs_rms = RunningMeanStd(shape=(obs_dim,)) if normalize else None
        self.target_rms = RunningMeanStd(shape=()) if normalize else None
        self.rng = np.random.default_rng(seed)

    def fit(self, obs: np.ndarray, target: np.ndarray) -> None:
        if self.normalize:
            self.obs_rms.update(obs)
            self.target_rms.update(target)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        target_t = torch.as_tensor(target, dtype=torch.float32)
        n = len(obs)
        bs = min(BATCH_SIZE, n)
        for _ in range(STEPS):
            idx = self.rng.integers(0, n, size=bs)
            batch_obs = obs_t[idx]
            batch_target = target_t[idx]
            if self.normalize:
                batch_obs = self.obs_rms.normalize(batch_obs)
                batch_target = self.target_rms.normalize(batch_target)
            pred = self.net(batch_obs).squeeze(-1)
            loss = F.mse_loss(pred, batch_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, query_obs: np.ndarray) -> np.ndarray:
        q = torch.as_tensor(query_obs, dtype=torch.float32)
        with torch.no_grad():
            q_in = self.obs_rms.normalize(q) if self.normalize else q
            raw = self.net(q_in).squeeze(-1)
            out = self.target_rms.denormalize(raw) if self.normalize else raw
        return out.numpy()


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default="results/g3_source_diagnostic/dataset")
    ap.add_argument("--out", default="results/g3_source_diagnostic/g3_report.json")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    meta = json.loads((dataset_dir / "meta.json").read_text())
    agent_ids: list[str] = meta["agent_ids"]
    obs_dim = int(meta["obs_dim"])
    n_agents = len(agent_ids)

    data = {aid: np.load(dataset_dir / f"{aid}.npz") for aid in agent_ids}
    for aid in agent_ids:
        print(f"{aid}: fit_rows={data[aid]['fit_obs'].shape[0]}, eval_rounds={data[aid]['eval_truth'].shape[0]}")

    report: dict = {"meta": meta, "hyperparameters": {
        "hidden_dim": HIDDEN_DIM, "steps": STEPS, "lr": LR, "batch_size": BATCH_SIZE,
        "architecture": "Linear(obs_dim,16)-Tanh-Linear(16,1)", "optimizer": "Adam",
        "init": "xavier_uniform (weights), zeros (bias)",
    }, "conditions": {}}

    for normalize in (False, True):
        cond_name = "normalized" if normalize else "raw"
        print(f"\n=== Fitting condition: {cond_name} ===")
        heads: dict[str, G3Head] = {}
        for i, aid in enumerate(agent_ids):
            seed = (6000 if normalize else 5000) + i
            head = G3Head(obs_dim, seed=seed, normalize=normalize)
            head.fit(data[aid]["fit_obs"], data[aid]["fit_target"])
            heads[aid] = head
            print(f"  fit {aid} (seed={seed})")

        # pred[q][s] : shape (n_eval,), agent s's head queried at agent q's obs
        pred = {q: {s: heads[s].predict(data[q]["eval_query_obs"]) for s in agent_ids} for q in agent_ids}
        truth = {aid: data[aid]["eval_truth"] for aid in agent_ids}

        # -- full 6x6 matrices --
        vs_owner_matrix = {q: {s: calibration_stats(pred[q][s], truth[q]) for s in agent_ids} for q in agent_ids}
        vs_source_matrix = {q: {s: calibration_stats(pred[q][s], truth[s]) for s in agent_ids} for q in agent_ids}

        # -- V1/V3: diagonal (q == s) --
        v1_per_agent = {aid: vs_owner_matrix[aid][aid] for aid in agent_ids}
        v1_pool = pool([pred[a][a] for a in agent_ids], [truth[a] for a in agent_ids])

        # -- V2: off-diagonal, scored against owner truth --
        v2_cells = [(q, s) for q in agent_ids for s in agent_ids if q != s]
        v2_per_pair = {f"{q}<-{s}": vs_owner_matrix[q][s] for q, s in v2_cells}
        v2_pool = pool([pred[q][s] for q, s in v2_cells], [truth[q] for q, s in v2_cells])

        # -- V4: off-diagonal, scored against the head's OWN (source) truth --
        v4_per_pair = {f"{q}<-{s}": vs_source_matrix[q][s] for q, s in v2_cells}
        v4_pool = pool([pred[q][s] for q, s in v2_cells], [truth[s] for q, s in v2_cells])

        report["conditions"][cond_name] = {
            "V1_V3_pooled": v1_pool, "V1_V3_pooled_classification": classify(v1_pool),
            "V1_V3_per_agent": v1_per_agent,
            "V2_pooled": v2_pool, "V2_pooled_classification": classify(v2_pool),
            "V2_per_pair": v2_per_pair,
            "V4_pooled": v4_pool, "V4_pooled_classification": classify(v4_pool),
            "V4_per_pair": v4_per_pair,
            "vs_owner_matrix": vs_owner_matrix,
            "vs_source_matrix": vs_source_matrix,
        }

        print(f"  V1/V3 (in-distribution) pooled: bias={v1_pool['bias']:.3f} MAE={v1_pool['mae']:.3f} "
              f"RMSE={v1_pool['rmse']:.3f} corr={v1_pool['corr']:.3f} FSR={v1_pool['fsr']:.3f} "
              f"[{classify(v1_pool)}]")
        print(f"  V2 (current peer design) pooled: bias={v2_pool['bias']:.3f} MAE={v2_pool['mae']:.3f} "
              f"RMSE={v2_pool['rmse']:.3f} corr={v2_pool['corr']:.3f} FSR={v2_pool['fsr']:.3f} "
              f"[{classify(v2_pool)}]")
        print(f"  V4 (wrong-target control) pooled: bias={v4_pool['bias']:.3f} MAE={v4_pool['mae']:.3f} "
              f"RMSE={v4_pool['rmse']:.3f} corr={v4_pool['corr']:.3f} FSR={v4_pool['fsr']:.3f} "
              f"[{classify(v4_pool)}]")

    # -- exchangeability analysis (normalization-independent: ground truth only) --
    truth_matrix = np.stack([data[aid]["eval_truth"] for aid in agent_ids], axis=0)  # (n_agents, n_eval)
    per_agent_mean = truth_matrix.mean(axis=1)
    per_agent_var = truth_matrix.var(axis=1)
    corr_matrix = np.corrcoef(truth_matrix)
    pooled_mean = float(truth_matrix.mean())
    exchangeable = bool(
        np.all(np.abs(per_agent_mean - pooled_mean) <= 2 * pooled_mean if pooled_mean > 0 else True)
        and np.all(corr_matrix[np.triu_indices(n_agents, k=1)] >= 0.5)
    )
    report["exchangeability"] = {
        "per_agent_mean_true_cost": dict(zip(agent_ids, per_agent_mean.tolist(), strict=True)),
        "per_agent_var_true_cost": dict(zip(agent_ids, per_agent_var.tolist(), strict=True)),
        "pooled_mean_true_cost": pooled_mean,
        "cross_agent_correlation_matrix": {
            agent_ids[i]: {agent_ids[j]: float(corr_matrix[i, j]) for j in range(n_agents)}
            for i in range(n_agents)
        },
        "called_exchangeable": exchangeable,
    }
    print("\n=== Exchangeability (realized MC cost, eval rounds) ===")
    for aid in agent_ids:
        print(f"  {aid}: mean={per_agent_mean[agent_ids.index(aid)]:.3f} var={per_agent_var[agent_ids.index(aid)]:.3f}")
    print(f"  called exchangeable: {exchangeable}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, allow_nan=True))
    print(f"\nFull report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
