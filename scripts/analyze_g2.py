#!/usr/bin/env python
"""Evaluate the G2-peer source-layer repair against its pre-declared gates.

Usage:
    python scripts/analyze_g2.py --runs-dir results/runs_constraint_mc_g2

Every threshold, window and direction below is transcribed from
`docs/g2_gates.md`, which was written and committed BEFORE any G2 run
existed -- as was this script. Neither chooses criteria; they apply them.
If a number here disagrees with that document, the document is right and
this script is a bug.

This script covers gates G2b through G2i (all numeric/log-derived). Gate
G2a (source-target correctness) is a unit-test gate --
`pytest tests/unit/test_constraint_report_head_wiring.py` -- and is not
re-implemented here; its pass/fail is recorded separately in the final
report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BUDGET = 25.0
N_ROUNDS = 250
LEARN_WINDOW = 20                 # G2f: first/last 20 rounds
TAIL_FRACTION = 0.20              # G2g/G2h: last 20% of rounds
G3A_TOL = 1e-9                    # G2e-a: residual identity
G3B_SIGN_MIN = 0.99               # G2e-b: sign agreement on interior cells
G2F_SD_MULTIPLE = 1.0             # G2f-b: improvement >= 1.0 sd of first window
G2G_MAX_RATIO = 1.15              # G2g: tail <= 1.15 * head
G6B_LAMBDA_FRACTION = 0.9         # G2h-b: max lambda < 0.9 * lambda_max
G6C_KL_TAIL_MAX = 0.05            # G2h-c: mean tail approx_kl
G6C_KL_ABS_MAX = 0.5              # G2h-c: max approx_kl anywhere
G6D_ENTROPY_MIN = 0.5             # G2h-d: mean tail entropy
G6E_DLAMBDA_MAX = 0.45            # G2h-e: mean |delta lambda|

# --- G1 baselines, transcribed verbatim from docs/g2_gates.md, keyed by
# --- seed index (0, 1, 2). These are frozen numbers from a completed,
# --- immutable campaign (results/runs_constraint_mc_g1/); they are inputs
# --- to this script, never recomputed from it.
G1_PEER_BIAS = {0: 7.897, 1: 7.865, 2: 3.272}          # |mean bias|, peer_critic, pooled 24 pairs
G1_PEER_MAE = {0: 15.987, 1: 15.118, 2: 14.470}
G1_MECH_BIAS_ABS = {0: 5.850, 1: 5.415, 2: 2.192}
G1_MECH_MAE = {0: 10.553, 1: 9.653, 2: 9.551}
G1_MECH_RMSE = {0: 13.318, 1: 12.212, 2: 11.938}
G1_FSR = {0: 0.381, 1: 0.398, 2: 0.457}

G2B_BIAS_RELATIVE = 0.5
G2B_MAE_RELATIVE = 0.75
G2C_BIAS_ABS = 0.20 * BUDGET       # 5.0
G2C_MAE_RELATIVE = 0.75
G2C_RMSE_RELATIVE = 0.75
G2D_FSR_RELATIVE = 0.5
G2D_FSR_ABS = 0.20


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _seed_index_from_name(name: str) -> int:
    # "pilot_A_clean_seed0" -> 0
    digits = "".join(ch for ch in name.rsplit("seed", 1)[-1] if ch.isdigit())
    return int(digits)


class Seed:
    """One G2 seed's logs, reduced to per-round arrays. Mirrors
    scripts/analyze_g1.py's `Seed` exactly, plus the false-safe indicator
    and per-source-type reductions G2 needs that G1 did not."""

    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.name = run_dir.name
        self.seed_idx = _seed_index_from_name(self.name)
        self.rounds = _read_jsonl(run_dir / "rounds.jsonl")
        self.oracle = _read_jsonl(run_dir / "oracle.jsonl")
        meta_path = run_dir / "run_metadata.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.agent_ids = list(self.rounds[0]["constraints"]) if self.rounds else []

        self.n = min(len(self.rounds), len(self.oracle))
        r, o = self.rounds[: self.n], self.oracle[: self.n]

        def per_agent_round(fn):
            return np.array([[fn(rec["constraints"][a]) for a in self.agent_ids] for rec in r])

        def per_agent_oracle(key):
            return np.array([[rec["agents"][a][key] for a in self.agent_ids] for rec in o])

        self.true_cost = per_agent_oracle("true_cost_return")
        self.reported_cost = per_agent_oracle("episodic_reported_cost_return")
        self.task_return = np.array([rec["agents"][self.agent_ids[0]]["episodic_task_return"] for rec in o])
        self.peak_true_cost = per_agent_oracle("peak_true_cost")
        self.violated = per_agent_oracle("violated").astype(bool)

        self.mechanism = per_agent_round(lambda c: c["mechanism_reported_cost_return"])
        self.residual = per_agent_round(lambda c: c["constraint_residual"])
        self.lam = per_agent_round(lambda c: c["lambda_after"])
        self.kl = per_agent_round(lambda c: c["ppo"]["approx_kl"])
        self.entropy = per_agent_round(lambda c: c["ppo"]["entropy"])

        source_ids = [x["source_id"] for x in r[0]["constraints"][self.agent_ids[0]]["reports"]] if r else []
        self.reports = {
            sid: per_agent_round(lambda c, s=sid: next(x["value"] for x in c["reports"] if x["source_id"] == s))
            for sid in source_ids
        }

        self.bias_mech = self.mechanism - self.true_cost
        self.tail = int(self.n * (1 - TAIL_FRACTION))
        self.false_safe = (self.mechanism <= BUDGET) & (self.true_cost > BUDGET)
        self.unsafe = self.true_cost > BUDGET

    def rmse(self, bias: np.ndarray) -> float:
        return float(np.sqrt((bias**2).mean()))

    def corr_per_agent(self, x: np.ndarray, y: np.ndarray) -> float:
        vals = []
        for i in range(x.shape[1]):
            xi, yi = x[:, i], y[:, i]
            if xi.std() < 1e-12 or yi.std() < 1e-12:
                continue
            vals.append(np.corrcoef(xi, yi)[0, 1])
        return float(np.mean(vals)) if vals else float("nan")

    def fsr(self) -> float:
        n_unsafe = int(self.unsafe.sum())
        if n_unsafe == 0:
            return float("nan")
        return float(self.false_safe.sum()) / n_unsafe


def gate_g2b(s: Seed, diversity: dict) -> dict:
    peer = [x for x in diversity["per_source"] if x["source_type"] == "peer_critic"]
    if not peer:
        return {"G2b-bias": (False, "no peer_critic sources found in source_diversity.json"),
                "G2b-mae": (False, "no peer_critic sources found")}
    bias = abs(float(np.mean([x["bias"] for x in peer])))
    mae = float(np.mean([x["mae"] for x in peer]))
    g1_bias, g1_mae = G1_PEER_BIAS[s.seed_idx], G1_PEER_MAE[s.seed_idx]
    bias_ok = bias <= G2B_BIAS_RELATIVE * g1_bias
    mae_ok = mae <= G2B_MAE_RELATIVE * g1_mae
    return {
        "G2b-bias": (bias_ok, f"|mean bias_peer|={bias:.3f} <= {G2B_BIAS_RELATIVE}*G1({g1_bias:.3f})={G2B_BIAS_RELATIVE*g1_bias:.3f}"),
        "G2b-mae": (mae_ok, f"mean MAE_peer={mae:.3f} <= {G2B_MAE_RELATIVE}*G1({g1_mae:.3f})={G2B_MAE_RELATIVE*g1_mae:.3f}"),
    }


def gate_g2c(s: Seed) -> dict:
    bias = abs(float(s.bias_mech.mean()))
    mae = float(np.mean(np.abs(s.bias_mech)))
    rmse = s.rmse(s.bias_mech)
    g1_mae, g1_rmse = G1_MECH_MAE[s.seed_idx], G1_MECH_RMSE[s.seed_idx]
    bias_ok = bias <= G2C_BIAS_ABS
    mae_ok = mae <= G2C_MAE_RELATIVE * g1_mae
    rmse_ok = rmse <= G2C_RMSE_RELATIVE * g1_rmse
    return {
        "G2c-bias": (bias_ok, f"|mean bias_mech|={bias:.3f} <= {G2C_BIAS_ABS}"),
        "G2c-mae": (mae_ok, f"mech MAE={mae:.3f} <= {G2C_MAE_RELATIVE}*G1({g1_mae:.3f})={G2C_MAE_RELATIVE*g1_mae:.3f}"),
        "G2c-rmse": (rmse_ok, f"mech RMSE={rmse:.3f} <= {G2C_RMSE_RELATIVE}*G1({g1_rmse:.3f})={G2C_RMSE_RELATIVE*g1_rmse:.3f}"),
    }


def gate_g2d(s: Seed) -> dict:
    fsr = s.fsr()
    g1_fsr = G1_FSR[s.seed_idx]
    rel_ok = fsr <= G2D_FSR_RELATIVE * g1_fsr
    abs_ok = fsr <= G2D_FSR_ABS
    return {
        "G2d-relative": (rel_ok, f"FSR={fsr:.3f} <= {G2D_FSR_RELATIVE}*G1({g1_fsr:.3f})={G2D_FSR_RELATIVE*g1_fsr:.3f}"),
        "G2d-absolute": (abs_ok, f"FSR={fsr:.3f} <= {G2D_FSR_ABS}"),
    }


def gate_g2e(s: Seed, eta: float, lam_max: float, W: np.ndarray) -> dict:
    ident = float(np.abs(s.residual - (s.mechanism - BUDGET)).max())
    lam_prev = np.vstack([np.zeros((1, s.lam.shape[1])), s.lam[:-1]])
    mixed = lam_prev @ W.T
    pre = mixed + eta * s.residual
    interior = (pre > 1e-12) & (pre < lam_max - 1e-12)
    delta = s.lam - mixed
    nz = interior & (np.abs(s.residual) > 1e-12)
    agree = float((np.sign(delta[nz]) == np.sign(s.residual[nz])).mean()) if nz.any() else float("nan")
    dlam = np.diff(s.lam, axis=0)
    resp = s.corr_per_agent(s.residual[:-1], dlam)
    return {
        "G2e-a": (ident <= G3A_TOL, f"max|residual-(mech-d)|={ident:.3e} <= {G3A_TOL:.0e}"),
        "G2e-b": (agree >= G3B_SIGN_MIN, f"sign agreement on {int(nz.sum())} interior cells = {agree:.4f} >= {G3B_SIGN_MIN}"),
        "G2e-c": (resp > 0, f"corr(residual_k, dlambda) = {resp:.3f} > 0"),
    }


def gate_g2f(seeds: list[Seed]) -> dict:
    firsts = [s.task_return[:LEARN_WINDOW] for s in seeds]
    lasts = [s.task_return[-LEARN_WINDOW:] for s in seeds]
    improvements = [float(last.mean() - first.mean()) for first, last in zip(firsts, lasts, strict=True)]
    g2f_a = all(d > 0 for d in improvements)
    pooled_sd = float(np.sqrt(np.mean([f.var(ddof=1) for f in firsts])))
    mean_imp = float(np.mean(improvements))
    g2f_b = mean_imp >= G2F_SD_MULTIPLE * pooled_sd
    return {
        "G2f-a": (g2f_a, f"per-seed improvement {[round(d, 1) for d in improvements]} all > 0"),
        "G2f-b": (g2f_b, f"mean improvement {mean_imp:.1f} >= {G2F_SD_MULTIPLE} x pooled sd {pooled_sd:.1f}"),
    }


def gate_g2g(s: Seed) -> dict:
    head_n = int(s.n * TAIL_FRACTION)
    head = float(s.true_cost[:head_n].mean())
    tail = float(s.true_cost[s.tail:].mean())
    ok = tail <= G2G_MAX_RATIO * head
    ratio = tail / head if head else float("nan")
    return {"G2g": (ok, f"true cost head={head:.2f} -> tail={tail:.2f} (ratio {ratio:.3f} <= {G2G_MAX_RATIO})")}


def gate_g2h(s: Seed, lam_max: float) -> dict:
    arrays = [s.true_cost, s.mechanism, s.residual, s.lam, s.kl, s.entropy, s.task_return]
    finite = all(bool(np.isfinite(a).all()) for a in arrays)
    lam_bar = G6B_LAMBDA_FRACTION * lam_max
    lam_max_seen = float(s.lam.max())
    saturated = int((s.lam >= lam_bar).sum())
    kl_tail, kl_max = float(s.kl[s.tail:].mean()), float(s.kl.max())
    ent_tail = float(s.entropy[s.tail:].mean())
    dlam = float(np.abs(np.diff(s.lam, axis=0)).mean())
    return {
        "G2h-a": (finite, "all logged fields finite" if finite else "NON-FINITE VALUE PRESENT"),
        "G2h-b": (lam_max_seen < lam_bar and saturated == 0,
                  f"max lambda {lam_max_seen:.3f} < {lam_bar} ({lam_max_seen / lam_max:.2f}x lambda_max), {saturated} saturated cells"),
        "G2h-c": (kl_tail <= G6C_KL_TAIL_MAX and kl_max <= G6C_KL_ABS_MAX,
                  f"tail mean KL {kl_tail:.4f} <= {G6C_KL_TAIL_MAX}, max KL {kl_max:.4f} <= {G6C_KL_ABS_MAX}"),
        "G2h-d": (ent_tail > G6D_ENTROPY_MIN, f"tail mean entropy {ent_tail:.3f} > {G6D_ENTROPY_MIN}"),
        "G2h-e": (dlam <= G6E_DLAMBDA_MAX, f"mean |delta lambda| {dlam:.4f} <= {G6E_DLAMBDA_MAX}"),
    }


def gate_g2i(seeds: list[Seed]) -> dict:
    complete = all(
        s.meta.get("status") == "complete" and len(s.rounds) == N_ROUNDS and len(s.oracle) == N_ROUNDS
        and (s.dir / "checkpoint.pt").exists()
        for s in seeds
    )
    shas = {s.meta.get("git", {}).get("sha") for s in seeds}
    dirty = {bool(s.meta.get("git", {}).get("dirty")) for s in seeds}
    estimators = {s.meta.get("config_snapshot", {}).get("constraint_estimator") for s in seeds}
    g2i_b = len(shas) == 1 and dirty == {False} and estimators == {"mc_window"}
    diversity_present = all((s.dir / "source_diversity.json").exists() for s in seeds)
    return {
        "G2i-a": (complete, f"{sum(1 for s in seeds if s.meta.get('status') == 'complete')}/{len(seeds)} complete, round counts {[len(s.rounds) for s in seeds]}"),
        "G2i-b": (g2i_b, f"sha={list(shas)[0] if len(shas) == 1 else shas}, dirty={dirty}, estimator={estimators}"),
        "G2i-d": (diversity_present, f"source_diversity.json present: {diversity_present}"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True, help="e.g. results/runs_constraint_mc_g2")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if runs_dir.name in ("runs", "runs_g0", "runs_constraint_mc_g1"):
        raise SystemExit(f"refusing to analyze {runs_dir}: that is an earlier campaign, not G2")
    seed_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir() and (d / "rounds.jsonl").exists())
    if not seed_dirs:
        raise SystemExit(f"no runs found under {runs_dir}")
    seeds = [Seed(d) for d in seed_dirs]

    from safelie.consensus.topologies import build_topology
    from safelie.governance.empirical_diversity import compute_empirical_diversity
    from safelie.sources.registry import effective_M
    from safelie.utils.config import load_experiment_config

    cfg0 = seeds[0].meta.get("config_snapshot", {})
    eta = cfg0.get("dual", {}).get("eta_lambda", 0.035)
    lam_max = cfg0.get("dual", {}).get("lambda_max", 25.0)
    topo = cfg0.get("topology", {})
    W = build_topology(topo.get("name", "ring"), topo.get("n_agents", 6),
                        p=topo.get("p"), graph_seed=topo.get("graph_seed", 0))

    config_path = REPO / "configs" / "experiment" / "pilot_A_clean.yaml"
    source_specs = None
    nominal_eff_m = None
    if config_path.exists():
        cfg = load_experiment_config(str(config_path))
        source_specs = {s.source_id: (s.source_type, s.independence_class) for s in cfg.sources.sources}
        nominal_eff_m = effective_M(cfg.sources.sources)

    diversity_by_seed = {}
    for s in seeds:
        report = compute_empirical_diversity(
            s.dir, nominal_effective_m=nominal_eff_m, source_specs=source_specs,
        )
        diversity_by_seed[s.name] = report.to_dict()
        out = s.dir / "source_diversity.json"
        out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    print("=" * 78)
    print("G2-PEER SOURCE-LAYER REPAIR -- pre-declared gates (docs/g2_gates.md)")
    print("=" * 78)

    print("\n## Seed-by-seed headline\n")
    hdr = ("seed", "task(f20)", "task(l20)", "trueC(f20)", "trueC(l20)", "mech", "b_mech",
           "mae_mech", "rmse_mech", "FSR", "lam_mu", "lam_max", "viol")
    print(" ".join(f"{h:>10s}" for h in hdr))
    for s in seeds:
        head_n = int(s.n * TAIL_FRACTION)
        row = (s.name, f"{s.task_return[:LEARN_WINDOW].mean():.1f}", f"{s.task_return[-LEARN_WINDOW:].mean():.1f}",
               f"{s.true_cost[:head_n].mean():.2f}", f"{s.true_cost[s.tail:].mean():.2f}",
               f"{s.mechanism.mean():.2f}", f"{s.bias_mech.mean():.2f}",
               f"{np.mean(np.abs(s.bias_mech)):.2f}", f"{s.rmse(s.bias_mech):.2f}",
               f"{s.fsr():.3f}", f"{s.lam.mean():.3f}", f"{s.lam.max():.3f}", f"{s.violated.mean():.3f}")
        print(" ".join(f"{v:>10s}" for v in row))

    print("\n## Per-source-type calibration (this run) vs G1 baseline\n")
    for s in seeds:
        d = diversity_by_seed[s.name]
        by_type: dict[str, list[dict]] = {}
        for x in d["per_source"]:
            by_type.setdefault(x["source_type"], []).append(x)
        print(f"  {s.name}:")
        for t, items in by_type.items():
            bias = float(np.mean([i["bias"] for i in items]))
            mae = float(np.mean([i["mae"] for i in items]))
            rmse = float(np.mean([i["rmse"] for i in items]))
            corr = float(np.mean([i["correlation_with_true"] for i in items]))
            print(f"    {t:16s} n={len(items):3d} bias={bias:8.3f} mae={mae:8.3f} rmse={rmse:8.3f} corr={corr:6.3f}")
        print(f"    statistical_effective_m_overall = {d['statistical_effective_m_overall']:.3f} "
              f"(nominal_effective_m={d['nominal_effective_m']})")

    print("\n## Clean false-safe rate by phase (5 x 50-round blocks)\n")
    for s in seeds:
        phase = s.n // 5
        cells = []
        for p in range(5):
            sl = slice(p * phase, (p + 1) * phase if p < 4 else s.n)
            u = s.unsafe[sl]
            fs = s.false_safe[sl]
            r = fs.sum() / max(1, u.sum())
            cells.append(f"{r:.2f}")
        worst = max(s.agent_ids, key=lambda a: s.false_safe[:, s.agent_ids.index(a)].mean())
        best = min(s.agent_ids, key=lambda a: s.false_safe[:, s.agent_ids.index(a)].mean())
        print(f"  {s.name}: phases={cells}  worst_agent={worst}  best_agent={best}  overall_fsr={s.fsr():.3f}")

    print("\n## Dual dynamics\n")
    for s in seeds:
        dlam = np.diff(s.lam, axis=0)
        resp = s.corr_per_agent(s.residual[:-1], dlam)
        print(f"  {s.name}: mean_lambda={s.lam.mean():.3f} max_lambda={s.lam.max():.3f} "
              f"corr(residual,dlambda)={resp:.3f}")

    print("\n" + "=" * 78)
    print("GATE RESULTS")
    print("=" * 78)
    per_seed_gates: dict[str, dict[str, tuple[bool, str]]] = {}
    for s in seeds:
        g: dict[str, tuple[bool, str]] = {}
        g.update(gate_g2b(s, diversity_by_seed[s.name]))
        g.update(gate_g2c(s))
        g.update(gate_g2d(s))
        g.update(gate_g2e(s, eta, lam_max, W))
        g.update(gate_g2g(s))
        g.update(gate_g2h(s, lam_max))
        per_seed_gates[s.name] = g

    cross = {}
    cross.update(gate_g2f(seeds))
    cross.update(gate_g2i(seeds))

    for s in seeds:
        print(f"\n{s.name}:")
        for k, (ok, detail) in per_seed_gates[s.name].items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {k}: {detail}")
    print("\ncross-seed:")
    for k, (ok, detail) in cross.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}: {detail}")

    groups = {
        "G2b": ["G2b-bias", "G2b-mae"], "G2c": ["G2c-bias", "G2c-mae", "G2c-rmse"],
        "G2d": ["G2d-relative", "G2d-absolute"], "G2e": ["G2e-a", "G2e-b", "G2e-c"],
        "G2f": ["G2f-a", "G2f-b"], "G2g": ["G2g"],
        "G2h": ["G2h-a", "G2h-b", "G2h-c", "G2h-d", "G2h-e"], "G2i": ["G2i-a", "G2i-b", "G2i-d"],
    }
    print("\n" + "=" * 78)
    print(f"{'gate':>6s} " + " ".join(f"{s.name:>28s}" for s in seeds) + f" {'overall':>10s}")
    print("=" * 78)
    summary = {}
    for gname, subs in groups.items():
        per = []
        for s in seeds:
            if gname in ("G2f", "G2i"):
                per.append(all(cross[x][0] for x in subs if x in cross))
            else:
                per.append(all(per_seed_gates[s.name][x][0] for x in subs))
        overall = all(per)
        summary[gname] = overall
        print(f"{gname:>6s} " + " ".join(f"{('PASS' if p else 'FAIL'):>28s}" for p in per) + f" {('PASS' if overall else 'FAIL'):>10s}")

    failed = [g for g, ok in summary.items() if not ok]
    d_failed = "G2d" in failed
    print("\n" + "=" * 78)
    if not failed:
        verdict = "PASS (numeric gates G2b-G2i; G2a is a separate unit-test gate)"
    elif d_failed or len(failed) >= 2:
        verdict = "FAIL"
    else:
        verdict = "CONDITIONAL PASS"
    print(f"VERDICT (mechanical, per docs/g2_gates.md, gates G2b-G2i only): {verdict}")
    if failed:
        print(f"  failed gates: {', '.join(failed)}")
    if d_failed:
        print("  G2d (clean false-safe rate) failed -- per docs/g2_gates.md this is an "
              "automatic FAIL and the task's section 19 stop condition.")
    print("=" * 78)

    out = runs_dir / "g2_gates.json"
    out.write_text(json.dumps({
        "verdict": verdict,
        "gates": {k: bool(v) for k, v in summary.items()},
        "failed": failed,
        "per_seed": {
            n: {k: {"pass": bool(v[0]), "detail": v[1]} for k, v in g.items()}
            for n, g in per_seed_gates.items()
        },
        "cross_seed": {k: {"pass": bool(v[0]), "detail": v[1]} for k, v in cross.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
