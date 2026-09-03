#!/usr/bin/env python
"""G6 (docs/g6_gates.md): source-estimation ceiling diagnostic.

Reads the SAME fixed dataset G3/G4/G5 already collected
(`results/g3_source_diagnostic/dataset/agent_*.npz` --
`scripts/g3_collect_calibration_dataset.py`). Trains no PPO, no GAE, no dual
update, no environment call, no attack, no RCE -- pure supervised regression
on a fixed, disk-persisted dataset, exactly as G5 did.

Asks: is the owner-specific source-estimation ceiling G5 measured
(`B1` pooled corr 0.79 but per-owner 0.47-0.64, 2/6 owners "mixed") caused
by (H1) estimator capacity, (H2) missing temporal context, or (H3) a
fundamental information/stochastic ceiling in `J_C^i(o)`.

Two protocols (docs/g6_gates.md, "Protocols"):
  P1 -- t0-eval: byte-comparable to G3/G5's B1 (fit rounds 0-39 pooled,
        eval at eval_query_obs/eval_truth rounds 40-59, t=0 only). F1 only.
  P2 -- round-CV, general-t: 5-fold CV over the 40 fit rounds, scored at
        every 20th valid query time t>=L-1 in each fold's 8 held-out
        rounds. Supports F1/F2/F3.

Usage:
    .venv/Scripts/python.exe scripts/g6_source_ceiling_diagnostic.py
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)  # this run already lost one multi-hour pass to a silently-buffered log

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from safelie.algos.normalization import RunningMeanStd

# --- pre-declared hyperparameters, docs/g6_gates.md ---
GAMMA = 0.99
BUDGET_D = 25.0
GOOD_BIAS = 0.20 * BUDGET_D  # 5.0
GOOD_CORR = 0.5
POOR_CORR = 0.2

N_FIT_ROUNDS = 40
N_FOLDS = 5
EVAL_STRIDE = 20
PRIMARY_L = 16
SECONDARY_LS = [4, 8]
KNN_K = 5
RIDGE_ALPHA = 1.0

CAPACITIES = ["C1", "C2", "C3"]
CAPACITY_SPEC = {
    "C1": {"hidden": [16], "steps": 2500, "lr": 1e-3, "batch": 2048},
    "C2": {"hidden": [32], "steps": 2500, "lr": 1e-3, "batch": 2048},
    "C3": {"hidden": [128, 64], "steps": 8000, "lr": 1e-3, "batch": 4096},
}


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class MLPHead:
    def __init__(self, in_dim: int, hidden: list[int], seed: int):
        gen = torch.Generator().manual_seed(seed)
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        for p in self.net.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, generator=gen)
            else:
                nn.init.zeros_(p)
        self.rng = np.random.default_rng(seed)

    def train_steps(self, obs_n: np.ndarray, target_n: np.ndarray, steps: int, lr: float, batch: int) -> bool:
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        obs_t = torch.as_tensor(obs_n, dtype=torch.float32)
        target_t = torch.as_tensor(target_n, dtype=torch.float32)
        n = len(obs_n)
        bs = min(batch, n)
        finite = True
        for _ in range(steps):
            idx = self.rng.integers(0, n, size=bs)
            pred = self.net(obs_t[idx]).squeeze(-1)
            loss = F.mse_loss(pred, target_t[idx])
            if not torch.isfinite(loss):
                finite = False
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return finite

    def predict_raw(self, obs_n: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.net(torch.as_tensor(obs_n, dtype=torch.float32)).squeeze(-1).numpy()


def fit_predict_mlp(train_x, train_y, query_x, capacity: str, seed: int) -> tuple[np.ndarray, bool]:
    spec = CAPACITY_SPEC[capacity]
    x_rms = RunningMeanStd(shape=(train_x.shape[1],))
    y_rms = RunningMeanStd(shape=())
    x_rms.update(train_x)
    y_rms.update(train_y)
    x_n = x_rms.normalize(torch.as_tensor(train_x, dtype=torch.float32)).numpy()
    y_n = y_rms.normalize(torch.as_tensor(train_y, dtype=torch.float32)).numpy()
    head = MLPHead(train_x.shape[1], spec["hidden"], seed=seed)
    finite = head.train_steps(x_n, y_n, spec["steps"], spec["lr"], spec["batch"])
    q_n = x_rms.normalize(torch.as_tensor(query_x, dtype=torch.float32)).numpy()
    raw = head.predict_raw(q_n)
    pred = y_rms.denormalize(torch.as_tensor(raw, dtype=torch.float32)).numpy()
    finite = finite and bool(np.all(np.isfinite(pred)))
    return pred, finite


def fit_predict_ridge(train_x, train_y, query_x, alpha: float) -> np.ndarray:
    """Closed-form ridge on standardized features/target (no sklearn dep)."""
    x_rms = RunningMeanStd(shape=(train_x.shape[1],))
    y_rms = RunningMeanStd(shape=())
    x_rms.update(train_x)
    y_rms.update(train_y)
    x_n = x_rms.normalize(torch.as_tensor(train_x, dtype=torch.float32)).numpy().astype(np.float64)
    y_n = y_rms.normalize(torch.as_tensor(train_y, dtype=torch.float32)).numpy().astype(np.float64)
    d = x_n.shape[1]
    xtx = x_n.T @ x_n + alpha * np.eye(d)
    xty = x_n.T @ y_n
    # lstsq (not solve): OLS (alpha=0) is singular whenever a feature has
    # zero variance in the training fold (confirmed present in this
    # dataset -- some obs dims are constant), and the least-squares
    # pseudo-inverse solution is the correct minimum-norm answer there.
    w, *_ = np.linalg.lstsq(xtx, xty, rcond=None)
    q_n = x_rms.normalize(torch.as_tensor(query_x, dtype=torch.float32)).numpy().astype(np.float64)
    raw = q_n @ w
    pred = y_rms.denormalize(torch.as_tensor(raw, dtype=torch.float32)).numpy()
    return pred


# --------------------------------------------------------------------------
# Metrics
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
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    r2 = float(1.0 - np.sum(err**2) / ss_tot) if ss_tot > 1e-12 else float("nan")
    unsafe_mask = truth > d
    fsr = float(np.mean(pred[unsafe_mask] <= d)) if unsafe_mask.sum() > 0 else float("nan")
    return {"n": int(len(pred)), "bias": bias, "mae": mae, "rmse": rmse, "corr": corr, "r2": r2,
            "fsr": fsr, "n_unsafe": int(unsafe_mask.sum())}


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


def pool(preds: list[np.ndarray], truths: list[np.ndarray]) -> dict:
    return calibration_stats(np.concatenate(preds), np.concatenate(truths))


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------

def reconstruct_costs(target: np.ndarray, gamma: float) -> np.ndarray:
    """c_s = G_s - gamma*G_{s+1} for s, s+1 both in the unmasked prefix.
    Last row has no s+1 available; repeats the second-to-last reconstructed
    cost (never used as a query time by construction -- window features only
    read c_s for s <= t, and the population's max valid t always keeps at
    least one row of margin -- see build_windows)."""
    c = np.empty_like(target)
    c[:-1] = target[:-1] - gamma * target[1:]
    c[-1] = c[-2] if len(c) > 1 else 0.0
    return c


def window_views(obs_round: np.ndarray, cost_round: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """obs_round (T,D), cost_round (T,). Returns (obs_win, cost_win) with
    obs_win[i] == obs_round[i:i+L] (L,D, oldest first) and cost_win[i] ==
    cost_round[i:i+L], for i in 0..T-L (obs_win[i] is the window ENDING at
    original index i+L-1). Vectorized via sliding_window_view -- no
    per-timestep Python loop (the earlier per-t loop implementation made a
    full P2 run computationally infeasible; this is a correctness-preserving
    rewrite, verified against the loop version on a small case below)."""
    from numpy.lib.stride_tricks import sliding_window_view
    obs_swv = sliding_window_view(obs_round, L, axis=0)  # (T-L+1, D, L)
    obs_win = np.ascontiguousarray(np.moveaxis(obs_swv, -1, 1))  # (T-L+1, L, D)
    cost_win = np.ascontiguousarray(sliding_window_view(cost_round, L))  # (T-L+1, L)
    return obs_win, cost_win


def build_f1(obs: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
    return obs[t_idx]


def build_f2(obs_win: np.ndarray, t_idx: np.ndarray, L: int) -> np.ndarray:
    rel = t_idx - (L - 1)
    win = obs_win[rel]  # (n, L, D)
    return win.reshape(win.shape[0], -1).astype(np.float32)


def build_f3(obs: np.ndarray, obs_win: np.ndarray, cost_win: np.ndarray, t_idx: np.ndarray, L: int) -> np.ndarray:
    rel = t_idx - (L - 1)
    ow = obs_win[rel]  # (n, L, D)
    cw = cost_win[rel]  # (n, L)
    o_t = obs[t_idx]  # (n, D)
    o_mean = ow.mean(axis=1)
    o_std = ow.std(axis=1)
    o_delta = ow[:, -1, :] - ow[:, 0, :]
    c_mean = cw.mean(axis=1)
    c_std = cw.std(axis=1)
    c_max = cw.max(axis=1)
    x = np.arange(L, dtype=np.float64)
    x_mean = x.mean()
    x_var = float(np.sum((x - x_mean) ** 2))
    if x_var > 1e-12:
        c_trend = ((cw - cw.mean(axis=1, keepdims=True)) @ (x - x_mean)) / x_var
    else:
        c_trend = np.zeros(cw.shape[0])
    out = np.concatenate(
        [o_t, o_mean, o_std, o_delta, c_mean[:, None], c_std[:, None], c_max[:, None], c_trend[:, None]], axis=1
    )
    return out.astype(np.float32)


def valid_t_range(rows_per_round: int, L: int, stride: int = 1) -> np.ndarray:
    return np.arange(L - 1, rows_per_round, stride)


# --------------------------------------------------------------------------
# Nearest-neighbor local target dispersion
# --------------------------------------------------------------------------

def nn_local_dispersion(x: np.ndarray, y: np.ndarray, k: int = KNN_K) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    xs = (x - mu) / sd
    n = len(xs)
    if n <= k:
        return {"n": n, "global_var": float(np.var(y)), "local_var": float("nan"), "ratio": float("nan")}
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b -- an (n,n) matrix via one matmul,
    # NOT the (n,n,d) broadcast this replaced: that materialized 3080 x 3080 x
    # 256 float64 (18.1 GiB) for the F3 pooled population and crashed the
    # process after the 3+ hour training portion of this run had already
    # completed and been checkpointed.
    sq = np.sum(xs**2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (xs @ xs.T)
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argpartition(d2, kth=k, axis=1)[:, :k]
    local_vars = np.array([np.var(y[nn_idx[i]]) for i in range(n)])
    local_var = float(np.mean(local_vars))
    global_var = float(np.var(y))
    ratio = float(local_var / global_var) if global_var > 1e-12 else float("nan")
    return {"n": n, "global_var": global_var, "local_var": local_var, "ratio": ratio, "k": k}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default="results/g3_source_diagnostic/dataset")
    ap.add_argument("--out", default="results/g6_source_ceiling_diagnostic/g6_report.json")
    args = ap.parse_args()
    t_start = time.time()

    dataset_dir = Path(args.dataset_dir)
    meta = json.loads((dataset_dir / "meta.json").read_text())
    agent_ids: list[str] = meta["agent_ids"]
    n_agents = len(agent_ids)
    gamma = float(meta["gamma"])
    assert abs(gamma - GAMMA) < 1e-12

    data = {aid: np.load(dataset_dir / f"{aid}.npz") for aid in agent_ids}
    rows_per_round = data[agent_ids[0]]["fit_obs"].shape[0] // N_FIT_ROUNDS
    assert rows_per_round * N_FIT_ROUNDS == data[agent_ids[0]]["fit_obs"].shape[0]
    for aid in agent_ids:
        assert data[aid]["fit_obs"].shape[0] == rows_per_round * N_FIT_ROUNDS

    # per-agent reconstructed per-step cost (for F3), and per-round views
    costs = {aid: reconstruct_costs(data[aid]["fit_target"], gamma) for aid in agent_ids}
    obs_by_round = {aid: data[aid]["fit_obs"].reshape(N_FIT_ROUNDS, rows_per_round, -1) for aid in agent_ids}
    target_by_round = {aid: data[aid]["fit_target"].reshape(N_FIT_ROUNDS, rows_per_round) for aid in agent_ids}
    cost_by_round = {aid: costs[aid].reshape(N_FIT_ROUNDS, rows_per_round) for aid in agent_ids}

    checkpoint_path = Path(args.out).with_name(Path(args.out).stem + "_checkpoint.json")
    if checkpoint_path.exists():
        report = json.loads(checkpoint_path.read_text())
        print(f"Resuming from {checkpoint_path}: last_completed_stage="
              f"{report.get('last_completed_stage')!r}, elapsed_at_checkpoint="
              f"{report.get('elapsed_s_at_checkpoint')}s -- already-completed stages are NOT recomputed.")
    else:
        report = {
            "meta": {**meta, "rows_per_round": rows_per_round, "g6_protocol": "P1 (t0-eval, F1 only) + P2 (round-CV, general-t, F1/F2/F3)"},
            "hyperparameters": {
                "capacities": CAPACITY_SPEC, "n_folds": N_FOLDS, "eval_stride": EVAL_STRIDE,
                "primary_L": PRIMARY_L, "secondary_Ls": SECONDARY_LS, "knn_k": KNN_K, "ridge_alpha": RIDGE_ALPHA,
            },
            "status": "in_progress",
        }

    def save_checkpoint(stage: str) -> None:
        """Writes the report as it stands so far -- a prior run of this
        script lost ~3 hours of compute to a process that died with no
        intermediate output at all. Cheap relative to any single fold's
        training cost; called after every major stage below."""
        report["last_completed_stage"] = stage
        report["elapsed_s_at_checkpoint"] = time.time() - t_start
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(report, indent=2, allow_nan=True))
        print(f"  [checkpoint saved: {stage}, elapsed={report['elapsed_s_at_checkpoint']:.1f}s -> {checkpoint_path}]")

    truth_eval = {aid: data[aid]["eval_truth"] for aid in agent_ids}
    # Persisted across resumes: only NEW training calls append to this list,
    # so on a resume that skips already-checkpointed P1/P2 stages, this
    # starts from whatever was recorded before the crash (see G6g_stability
    # below for the honesty caveat this implies).
    stability_failures: list[str] = report.get("_stability_failures", [])
    report["_stability_failures"] = stability_failures

    # ======================================================================
    # B. Target Definition Audit
    # ======================================================================
    if "B_target_audit" not in report:
        print("=== B. Target definition audit ===")
        audit = {}
        for aid in agent_ids:
            t0_fit = target_by_round[aid][:, 0]
            t0_all = np.concatenate([t0_fit, data[aid]["eval_truth"]])
            obs0_fit = obs_by_round[aid][:, 0, :]
            obs0_all = np.concatenate([obs0_fit, data[aid]["eval_query_obs"]], axis=0)
            overall_obs_var = data[aid]["fit_obs"].var(axis=0)
            obs0_var = obs0_all.var(axis=0)
            nz = overall_obs_var > 1e-8
            ratio = (obs0_var[nz] / overall_obs_var[nz])
            audit[aid] = {
                "n_t0_samples_all60rounds": int(len(t0_all)),
                "target_t0_mean": float(t0_all.mean()),
                "target_t0_var": float(t0_all.var()),
                "target_t0_std": float(t0_all.std()),
                "eval_truth_only_var_20rounds": float(data[aid]["eval_truth"].var()),
                "obs0_vs_overall_var_ratio_mean": float(ratio.mean()),
                "obs0_vs_overall_var_ratio_max": float(ratio.max()),
            }
            print(f"  {aid}: target_t0 var={audit[aid]['target_t0_var']:.2f} (60 rounds), "
                  f"obs0/overall var ratio mean={audit[aid]['obs0_vs_overall_var_ratio_mean']:.2f}")
        report["B_target_audit"] = audit
        save_checkpoint("B_target_audit")
    else:
        print("=== B. Target definition audit (already checkpointed, skipping) ===")

    # ======================================================================
    # P1 -- t0-eval, F1 only (direct G3/G5 comparison)
    # ======================================================================
    if "P1_t0_eval" not in report:
        print("\n=== P1 (t0-eval): F1 x C1/C2/C3 + ridge/OLS ===")
        p1: dict = {}
        for cap in CAPACITIES:
            per_agent_pred = {}
            for i, aid in enumerate(agent_ids):
                seed = 60000 + i
                pred, finite = fit_predict_mlp(
                    data[aid]["fit_obs"], data[aid]["fit_target"], data[aid]["eval_query_obs"], cap, seed)
                per_agent_pred[aid] = pred
                if not finite:
                    stability_failures.append(f"P1/F1/{cap}/{aid}")
            per_agent_stats = {aid: calibration_stats(per_agent_pred[aid], truth_eval[aid]) for aid in agent_ids}
            pooled = pool([per_agent_pred[a] for a in agent_ids], [truth_eval[a] for a in agent_ids])
            p1[cap] = {
                "per_agent": per_agent_stats,
                "per_agent_classification": {aid: classify(per_agent_stats[aid]) for aid in agent_ids},
                "pooled": pooled, "pooled_classification": classify(pooled),
            }
            print(f"  F1,{cap}: pooled corr={pooled['corr']:.3f} MAE={pooled['mae']:.3f} [{classify(pooled)}]")

        for base_name, alpha in [("ridge", RIDGE_ALPHA), ("ols", 0.0)]:
            per_agent_pred = {}
            for aid in agent_ids:
                per_agent_pred[aid] = fit_predict_ridge(data[aid]["fit_obs"], data[aid]["fit_target"], data[aid]["eval_query_obs"], alpha)
            per_agent_stats = {aid: calibration_stats(per_agent_pred[aid], truth_eval[aid]) for aid in agent_ids}
            pooled = pool([per_agent_pred[a] for a in agent_ids], [truth_eval[a] for a in agent_ids])
            p1[base_name] = {
                "per_agent": per_agent_stats,
                "per_agent_classification": {aid: classify(per_agent_stats[aid]) for aid in agent_ids},
                "pooled": pooled, "pooled_classification": classify(pooled),
            }
            print(f"  F1,{base_name}: pooled corr={pooled['corr']:.3f} MAE={pooled['mae']:.3f} [{classify(pooled)}]")
        report["P1_t0_eval"] = p1
        save_checkpoint("P1_t0_eval")
    else:
        print("=== P1 (t0-eval) already checkpointed, skipping ===")
        p1 = report["P1_t0_eval"]

    # ======================================================================
    # P2 -- round-CV, general-t: F1/F2/F3 x C1/C2/C3, folds
    # ======================================================================
    print("\n=== P2 (round-CV, general-t) ===")
    fold_round_groups = np.array_split(np.arange(N_FIT_ROUNDS), N_FOLDS)

    def build_feature(name: str, aid: str, round_idx: int, t_idx: np.ndarray, L: int,
                       win_cache: dict) -> np.ndarray:
        o = obs_by_round[aid][round_idx]
        if name == "F1":
            return build_f1(o, t_idx)
        key = (aid, round_idx, L)
        if key not in win_cache:
            win_cache[key] = window_views(o, cost_by_round[aid][round_idx], L)
        obs_win, cost_win = win_cache[key]
        if name == "F2":
            return build_f2(obs_win, t_idx, L)
        if name == "F3":
            return build_f3(o, obs_win, cost_win, t_idx, L)
        raise ValueError(name)

    def build_fold_features(feat_name: str, L: int) -> dict:
        """Builds (train_x, train_y, query_x, query_y) once per (fold, agent)
        for this (feat_name, L) -- shared across every capacity/baseline that
        reuses it, instead of rebuilding on every call (rebuilding is what
        made a naive implementation of this loop computationally infeasible:
        3 capacities + 2 baselines would otherwise recompute identical
        features 5 times each)."""
        t_pop = valid_t_range(rows_per_round, L, stride=1)
        t_eval_pop = valid_t_range(rows_per_round, L, stride=EVAL_STRIDE)
        win_cache: dict = {}
        cache: dict = {}
        for f_idx, held_rounds in enumerate(fold_round_groups):
            held_set = set(int(r) for r in held_rounds)
            train_rounds = [r for r in range(N_FIT_ROUNDS) if r not in held_set]
            assert len(set(train_rounds) & held_set) == 0, "G6a leakage check failed"
            for aid in agent_ids:
                train_x = np.concatenate(
                    [build_feature(feat_name, aid, r, t_pop, L, win_cache) for r in train_rounds], axis=0)
                train_y = np.concatenate([target_by_round[aid][r, t_pop] for r in train_rounds], axis=0)
                query_parts = [build_feature(feat_name, aid, int(r), t_eval_pop, L, win_cache) for r in held_rounds]
                truth_parts = [target_by_round[aid][int(r), t_eval_pop] for r in held_rounds]
                query_x = np.concatenate(query_parts, axis=0)
                query_y = np.concatenate(truth_parts, axis=0)
                cache[(f_idx, aid)] = (train_x, train_y, query_x, query_y)
        return cache

    def run_p2_condition(feat_name: str, L: int, capacity_or_baseline: str, fold_cache: dict) -> dict:
        """Returns per-owner per-fold stats + pooled-across-folds stats.
        `fold_cache` is `build_fold_features(feat_name, L)`, passed in so it
        is built once and reused across every model variant for this
        (feat_name, L)."""
        per_owner_fold_stats: dict[str, list[dict]] = {aid: [] for aid in agent_ids}
        per_owner_all_pred: dict[str, list[np.ndarray]] = {aid: [] for aid in agent_ids}
        per_owner_all_truth: dict[str, list[np.ndarray]] = {aid: [] for aid in agent_ids}
        for f_idx in range(N_FOLDS):
            for i, aid in enumerate(agent_ids):
                train_x, train_y, query_x, query_y = fold_cache[(f_idx, aid)]
                seed = 70000 + f_idx * 100 + i
                if capacity_or_baseline in CAPACITY_SPEC:
                    pred, finite = fit_predict_mlp(train_x, train_y, query_x, capacity_or_baseline, seed)
                    if not finite:
                        stability_failures.append(f"P2/{feat_name}L{L}/{capacity_or_baseline}/fold{f_idx}/{aid}")
                elif capacity_or_baseline == "ridge":
                    pred = fit_predict_ridge(train_x, train_y, query_x, RIDGE_ALPHA)
                elif capacity_or_baseline == "ols":
                    pred = fit_predict_ridge(train_x, train_y, query_x, 0.0)
                else:
                    raise ValueError(capacity_or_baseline)

                s = calibration_stats(pred, query_y)
                per_owner_fold_stats[aid].append(s)
                per_owner_all_pred[aid].append(pred)
                per_owner_all_truth[aid].append(query_y)

        per_owner_pooled = {aid: pool(per_owner_all_pred[aid], per_owner_all_truth[aid]) for aid in agent_ids}
        pooled_all = pool(
            [p for aid in agent_ids for p in per_owner_all_pred[aid]],
            [t for aid in agent_ids for t in per_owner_all_truth[aid]],
        )
        return {
            "feature": feat_name, "L": L, "model": capacity_or_baseline,
            "per_owner_fold_stats": per_owner_fold_stats,
            "per_owner_pooled": per_owner_pooled,
            "per_owner_pooled_classification": {aid: classify(per_owner_pooled[aid]) for aid in agent_ids},
            "pooled_across_owners": pooled_all,
            "pooled_across_owners_classification": classify(pooled_all),
            "n_folds": N_FOLDS,
        }

    p2: dict = report.get("P2_round_cv", {})
    report["P2_round_cv"] = p2  # same dict object -- mutations below are visible to save_checkpoint immediately
    # Core comparisons at L=16: F1/F2/F3 x C1/C2/C3, plus ridge/ols on F1/F2/F3
    for feat in ["F1", "F2", "F3"]:
        feat_keys = [f"{feat}_L{PRIMARY_L}_{m}" for m in [*CAPACITIES, "ridge", "ols"]]
        if all(k in p2 for k in feat_keys):
            print(f"  {feat} L={PRIMARY_L} already checkpointed for all of {CAPACITIES + ['ridge', 'ols']}, skipping")
            continue
        print(f"  building {feat} L={PRIMARY_L} features (once, shared across C1/C2/C3/ridge/ols) ...")
        t_build0 = time.time()
        fold_cache = build_fold_features(feat, PRIMARY_L)
        print(f"    feature build took {time.time() - t_build0:.1f}s")
        for cap in CAPACITIES:
            key = f"{feat}_L{PRIMARY_L}_{cap}"
            if key in p2:
                print(f"  {key} already checkpointed, skipping")
                continue
            print(f"  running {key} ...")
            p2[key] = run_p2_condition(feat, PRIMARY_L, cap, fold_cache)
            pc = p2[key]["pooled_across_owners"]
            print(f"    pooled corr={pc['corr']:.3f} MAE={pc['mae']:.3f} R2={pc['r2']:.3f} "
                  f"[{p2[key]['pooled_across_owners_classification']}]")
            save_checkpoint(f"P2/{key}")
        for base in ["ridge", "ols"]:
            key = f"{feat}_L{PRIMARY_L}_{base}"
            if key in p2:
                print(f"  {key} already checkpointed, skipping")
                continue
            print(f"  running {key} ...")
            p2[key] = run_p2_condition(feat, PRIMARY_L, base, fold_cache)
            pc = p2[key]["pooled_across_owners"]
            print(f"    pooled corr={pc['corr']:.3f} MAE={pc['mae']:.3f} R2={pc['r2']:.3f} "
                  f"[{p2[key]['pooled_across_owners_classification']}]")
            save_checkpoint(f"P2/{key}")
        del fold_cache

    # Secondary: F2 at L=4,8, C3 only (robustness of the temporal comparison to L)
    for L in SECONDARY_LS:
        key = f"F2_L{L}_C3"
        if key in p2:
            print(f"  {key} already checkpointed, skipping")
            continue
        print(f"  running {key} (secondary L check) ...")
        p2[key] = run_p2_condition("F2", L, "C3", build_fold_features("F2", L))
        pc = p2[key]["pooled_across_owners"]
        print(f"    pooled corr={pc['corr']:.3f} MAE={pc['mae']:.3f} [{p2[key]['pooled_across_owners_classification']}]")
        save_checkpoint(f"P2/{key}")

    # ======================================================================
    # L. Irreducible-variability / nearest-neighbor dispersion analysis
    # ======================================================================
    if "L_nn_dispersion" not in report:
        print("\n=== L. Nearest-neighbor local target dispersion ===")
        nn_analysis: dict = {}
        # (a) P1 population: F1 = o_0 across all 60 rounds per owner
        nn_p1 = {}
        for aid in agent_ids:
            t0_fit = target_by_round[aid][:, 0]
            t0_all = np.concatenate([t0_fit, data[aid]["eval_truth"]])
            obs0_fit = obs_by_round[aid][:, 0, :]
            obs0_all = np.concatenate([obs0_fit, data[aid]["eval_query_obs"]], axis=0)
            nn_p1[aid] = nn_local_dispersion(obs0_all, t0_all)
        nn_analysis["P1_t0_population_F1"] = nn_p1
        print("  P1 (t=0, F1=o_0):", {aid: round(v["ratio"], 3) for aid, v in nn_p1.items()})

        # (b) P2 population: F1 (o_t) and F3 (rich features) at L=16, pooled over all 40 fit rounds
        t_pop_dense = valid_t_range(rows_per_round, PRIMARY_L, stride=EVAL_STRIDE)
        nn_p2_f1, nn_p2_f3 = {}, {}
        for aid in agent_ids:
            win_cache_nn: dict = {}
            f1_x = np.concatenate(
                [build_feature("F1", aid, r, t_pop_dense, PRIMARY_L, win_cache_nn) for r in range(N_FIT_ROUNDS)], axis=0)
            f3_x = np.concatenate(
                [build_feature("F3", aid, r, t_pop_dense, PRIMARY_L, win_cache_nn) for r in range(N_FIT_ROUNDS)], axis=0)
            y = np.concatenate([target_by_round[aid][r, t_pop_dense] for r in range(N_FIT_ROUNDS)], axis=0)
            nn_p2_f1[aid] = nn_local_dispersion(f1_x, y)
            nn_p2_f3[aid] = nn_local_dispersion(f3_x, y)
        nn_analysis["P2_general_t_population_F1"] = nn_p2_f1
        nn_analysis["P2_general_t_population_F3"] = nn_p2_f3
        print("  P2 general-t (F1=o_t):", {aid: round(v["ratio"], 3) for aid, v in nn_p2_f1.items()})
        print("  P2 general-t (F3=rich):", {aid: round(v["ratio"], 3) for aid, v in nn_p2_f3.items()})
        report["L_nn_dispersion"] = nn_analysis
        save_checkpoint("L_nn_dispersion")
    else:
        print("=== L. Nearest-neighbor local target dispersion (already checkpointed, skipping) ===")
        nn_p2_f1 = report["L_nn_dispersion"]["P2_general_t_population_F1"]
        nn_p2_f3 = report["L_nn_dispersion"]["P2_general_t_population_F3"]

    pooled_f3_ratio = float(np.mean([v["ratio"] for v in nn_p2_f3.values() if not np.isnan(v["ratio"])]))
    pooled_f1_ratio = float(np.mean([v["ratio"] for v in nn_p2_f1.values() if not np.isnan(v["ratio"])]))
    report["G6f_ceiling_flag"] = {
        "pooled_f1_ratio": pooled_f1_ratio, "pooled_f3_ratio": pooled_f3_ratio,
        "ceiling_flag_set": pooled_f3_ratio >= 0.85,
        "ceiling_flag_cleared": pooled_f3_ratio <= 0.5,
    }

    # ======================================================================
    # Gates: G6a (leakage, asserted inline above), G6e (nonlinearity), G6g (stability)
    # ======================================================================
    g6e = {}
    for feat in ["F1", "F2", "F3"]:
        mlp_mae = p2[f"{feat}_L{PRIMARY_L}_C3"]["pooled_across_owners"]["mae"]
        ridge_mae = p2[f"{feat}_L{PRIMARY_L}_ridge"]["pooled_across_owners"]["mae"]
        rel_improve = (ridge_mae - mlp_mae) / ridge_mae if ridge_mae > 1e-12 else float("nan")
        g6e[feat] = {"mlp_c3_mae": mlp_mae, "ridge_mae": ridge_mae, "relative_improvement": rel_improve,
                     "g6e_pass": rel_improve >= 0.10}
    report["G6e_nonlinearity"] = g6e

    report["G6g_stability"] = {
        "n_stability_failures": len(stability_failures),
        "stability_failures": stability_failures,
        "g6g_pass": len(stability_failures) == 0,
        "caveat": (
            "If this run was resumed from a checkpoint, stability failures from "
            "any SKIPPED (already-checkpointed) training stage are not re-verified "
            "here -- only stages actually (re-)trained in this process append to "
            "this list. Every printed pooled bias/MAE/corr for every checkpointed "
            "P1/P2 condition in this run's log was a finite, non-NaN number, which "
            "is the empirical basis for treating prior stages as stable, but it is "
            "not the same guarantee as this process's own per-step finite-loss check."
        ),
    }
    del report["_stability_failures"]

    # ======================================================================
    # Comparisons A/B/C banding
    # ======================================================================
    def band(corr_gain: float, n_flips_good: int, n_flips_bad: int) -> str:
        if n_flips_bad > 0:
            pass  # still band on corr_gain/flips; flips_bad noted separately
        if corr_gain >= 0.30:
            return "decisive"
        if corr_gain >= 0.15 or n_flips_good >= 2:
            return "meaningful"
        if corr_gain >= 0.10:
            return "minimal"
        return "none"

    def owner_flip_counts(before_key: str, after_key: str) -> tuple[int, int]:
        before_cls = p2[before_key]["per_owner_pooled_classification"]
        after_cls = p2[after_key]["per_owner_pooled_classification"]
        flips_good = sum(1 for aid in agent_ids if before_cls[aid] != "good" and after_cls[aid] == "good")
        flips_bad = sum(1 for aid in agent_ids if before_cls[aid] == "good" and after_cls[aid] != "good")
        return flips_good, flips_bad

    comparisons = {}
    # Comparison A: F1,C1 -> F1,C3 (P2)
    c1_corr = p2[f"F1_L{PRIMARY_L}_C1"]["pooled_across_owners"]["corr"]
    c3_corr = p2[f"F1_L{PRIMARY_L}_C3"]["pooled_across_owners"]["corr"]
    fg, fb = owner_flip_counts(f"F1_L{PRIMARY_L}_C1", f"F1_L{PRIMARY_L}_C3")
    comparisons["A_capacity"] = {
        "before": f"F1_L{PRIMARY_L}_C1", "after": f"F1_L{PRIMARY_L}_C3",
        "corr_before": c1_corr, "corr_after": c3_corr, "corr_gain": c3_corr - c1_corr,
        "owners_flipped_to_good": fg, "owners_flipped_to_notgood": fb,
        "band": band(c3_corr - c1_corr, fg, fb),
        "p1_confirmatory": {
            "corr_before": p1["C1"]["pooled"]["corr"], "corr_after": p1["C3"]["pooled"]["corr"],
        },
    }
    # Comparison B: F1,C3 -> F2,C3 (P2)
    b_before_corr = p2[f"F1_L{PRIMARY_L}_C3"]["pooled_across_owners"]["corr"]
    b_after_corr = p2[f"F2_L{PRIMARY_L}_C3"]["pooled_across_owners"]["corr"]
    fg, fb = owner_flip_counts(f"F1_L{PRIMARY_L}_C3", f"F2_L{PRIMARY_L}_C3")
    comparisons["B_temporal"] = {
        "before": f"F1_L{PRIMARY_L}_C3", "after": f"F2_L{PRIMARY_L}_C3",
        "corr_before": b_before_corr, "corr_after": b_after_corr, "corr_gain": b_after_corr - b_before_corr,
        "owners_flipped_to_good": fg, "owners_flipped_to_notgood": fb,
        "band": band(b_after_corr - b_before_corr, fg, fb),
        "secondary_L_check": {
            f"L{L}": p2[f"F2_L{L}_C3"]["pooled_across_owners"]["corr"] for L in SECONDARY_LS
        },
    }
    # Comparison C: F2,C3 -> F3,C3 (P2)
    c_before_corr = p2[f"F2_L{PRIMARY_L}_C3"]["pooled_across_owners"]["corr"]
    c_after_corr = p2[f"F3_L{PRIMARY_L}_C3"]["pooled_across_owners"]["corr"]
    fg, fb = owner_flip_counts(f"F2_L{PRIMARY_L}_C3", f"F3_L{PRIMARY_L}_C3")
    comparisons["C_rich_features"] = {
        "before": f"F2_L{PRIMARY_L}_C3", "after": f"F3_L{PRIMARY_L}_C3",
        "corr_before": c_before_corr, "corr_after": c_after_corr, "corr_gain": c_after_corr - c_before_corr,
        "owners_flipped_to_good": fg, "owners_flipped_to_notgood": fb,
        "band": band(c_after_corr - c_before_corr, fg, fb),
    }
    report["comparisons"] = comparisons

    # ======================================================================
    # Verdict
    # ======================================================================
    band_rank = {"none": 0, "minimal": 1, "meaningful": 2, "decisive": 3}
    a_band = comparisons["A_capacity"]["band"]
    b_band = comparisons["B_temporal"]["band"]
    c_band = comparisons["C_rich_features"]["band"]

    per_owner_any_good = {}
    for aid in agent_ids:
        any_good = any(p1[cap]["per_agent_classification"][aid] == "good" for cap in CAPACITIES)
        any_good = any_good or any(
            p2[f"{feat}_L{PRIMARY_L}_{cap}"]["per_owner_pooled_classification"][aid] == "good"
            for feat in ["F1", "F2", "F3"] for cap in CAPACITIES
        )
        per_owner_any_good[aid] = any_good
    n_owners_resolve_any = sum(per_owner_any_good.values())

    if band_rank[a_band] >= 2 and band_rank[b_band] < 2 and band_rank[c_band] < 2:
        verdict = "CAPACITY-LIMITED"
    elif band_rank[b_band] >= 2 and band_rank[b_band] >= band_rank[a_band] and band_rank[c_band] < 2:
        verdict = "TEMPORAL-LIMITED"
    elif band_rank[c_band] >= 2 and band_rank[c_band] > max(band_rank[a_band], band_rank[b_band]):
        verdict = "FEATURE-LIMITED"
    elif band_rank[a_band] < 2 and band_rank[b_band] < 2 and band_rank[c_band] < 2 and report["G6f_ceiling_flag"]["ceiling_flag_set"]:
        verdict = "FUNDAMENTAL-LIMITED"
    else:
        verdict = "MIXED"

    if n_owners_resolve_any >= 4 and verdict in ("CAPACITY-LIMITED",):
        decision = "YES"
    elif verdict == "FUNDAMENTAL-LIMITED" or (verdict == "MIXED" and n_owners_resolve_any <= 2):
        decision = "NO"
    else:
        decision = "CONDITIONAL"

    report["verdict"] = {
        "comparison_bands": {"A_capacity": a_band, "B_temporal": b_band, "C_rich_features": c_band},
        "per_owner_resolves_under_any_condition": per_owner_any_good,
        "n_owners_resolve_any": n_owners_resolve_any,
        "verdict": verdict,
        "continuation_decision": decision,
    }

    report["wall_clock_s"] = time.time() - t_start
    report["status"] = "complete"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, allow_nan=True))
    print(f"\nFull report written to {out_path}")
    print(f"\n=== VERDICT: {verdict} | continuation decision: {decision} ===")
    print(f"Comparison A (capacity): {a_band} (gain={comparisons['A_capacity']['corr_gain']:.3f})")
    print(f"Comparison B (temporal): {b_band} (gain={comparisons['B_temporal']['corr_gain']:.3f})")
    print(f"Comparison C (features): {c_band} (gain={comparisons['C_rich_features']['corr_gain']:.3f})")
    print(f"Owners resolving under ANY condition: {n_owners_resolve_any}/6")
    print(f"Ceiling flag (F3 NN ratio): set={report['G6f_ceiling_flag']['ceiling_flag_set']} "
          f"(pooled_f3_ratio={pooled_f3_ratio:.3f})")
    print(f"Wall clock: {report['wall_clock_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
