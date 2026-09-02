#!/usr/bin/env python
"""Evaluate the G1 dual-constraint-estimator calibration against its
pre-declared gates.

Usage:
    python scripts/analyze_g1.py --runs-dir results/runs_constraint_mc_g1

Every threshold, window and direction below is transcribed from
`docs/g1_gates.md`, which was written and committed BEFORE any G1 run
existed -- as was this script. Neither chooses criteria; they apply them.
If a number here disagrees with that document, the document is right and
this script is a bug.

Three rules this file exists to enforce mechanically:

  1. Truth is `oracle.jsonl: true_cost_return` and nothing else. Gate
     arithmetic over evaluation quantities reads only `oracle.jsonl`'s
     `episodic_task_return`, `true_cost_return` and
     `episodic_reported_cost_return`, plus `rounds.jsonl`'s
     `mechanism_reported_cost_return` (the value that actually drove the
     dual) and the `constraint_estimators` block (both estimators, from
     the same rollout).
  2. The two estimators are compared only against the SAME oracle
     episodes, so the two-rollout sampling noise and the one-update policy
     offset that contaminate both biases are shared, and their difference
     is interpretable even though neither bias in isolation is pure
     estimator error.
  3. The runs directory is an explicit argument with no default that
     could reach `results/runs` or `results/runs_g0`, both of which hold
     earlier campaigns that must never be merged with this one.
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
LEARN_WINDOW = 20                 # G4: first/last 20 rounds
TAIL_FRACTION = 0.20              # G5b/G6c/G6d: last 20% of rounds
G1A_RELATIVE_MAX = 0.5            # G1a: |bias_mc| <= 0.5 * |bias_gae|
G1B_ABS_MAX = 0.20 * BUDGET       # G1b/G1d: |mean bias| <= 5.0
G1C_PHASES = 5                    # G1c: five 50-round phases
G1C_MIN_PHASES_BETTER = 4         # G1c: MC better in >= 4 of 5
G2A_EXACT_TOL = 1e-9              # G2a: reported vs true, same rollout
G2C_CORR_MIN = 0.70               # G2c: corr(mc, true) >= 0.70
G3A_TOL = 1e-9                    # G3a: residual identity
G3B_SIGN_MIN = 0.99               # G3b: sign agreement in interior cells
G4B_SD_MULTIPLE = 1.0             # G4b: improvement >= 1.0 sd of first window
G6B_LAMBDA_FRACTION = 0.9         # G6b: max lambda < 0.9 * lambda_max
G6C_KL_TAIL_MAX = 0.05            # G6c: mean tail approx_kl
G6C_KL_ABS_MAX = 0.5              # G6c: max approx_kl anywhere
G6D_ENTROPY_MIN = 0.5             # G6d: mean tail entropy
G6E_DLAMBDA_MAX = 0.45            # G6e: mean |delta lambda| (3x G0's 0.145-0.153)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


class Seed:
    """One G1 seed's logs, reduced to per-round arrays.

    Per-agent quantities are (n_rounds, n_agents) with agents in
    `self.agent_ids` order throughout, so a column index means the same
    agent in every array and a row index means the same round in both
    logs.
    """

    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.name = run_dir.name
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

        def estimator(key):
            return np.array(
                [[rec["constraints"][a]["constraint_estimators"][key] for a in self.agent_ids] for rec in r]
            )

        # --- evaluation quantities (oracle-written) ---
        self.true_cost = per_agent_oracle("true_cost_return")
        self.reported_cost = per_agent_oracle("episodic_reported_cost_return")
        self.task_return = np.array([rec["agents"][self.agent_ids[0]]["episodic_task_return"] for rec in o])
        self.peak_true_cost = per_agent_oracle("peak_true_cost")
        self.violated = per_agent_oracle("violated").astype(bool)

        # --- learner quantities ---
        self.mechanism = per_agent_round(lambda c: c["mechanism_reported_cost_return"])
        self.residual = per_agent_round(lambda c: c["constraint_residual"])
        self.lam = per_agent_round(lambda c: c["lambda_after"])
        self.est_gae = estimator("gae_lambda")
        self.est_mc = estimator("mc_window")
        self.est_mc_ep = estimator("mc_episodic")
        self.active = self.rounds[0]["constraints"][self.agent_ids[0]]["constraint_estimators"]["active"]
        self.kl = per_agent_round(lambda c: c["ppo"]["approx_kl"])
        self.entropy = per_agent_round(lambda c: c["ppo"]["entropy"])
        self.value_loss = per_agent_round(lambda c: c["ppo"]["value_loss"])
        self.cost_value_loss = per_agent_round(lambda c: c["ppo"]["cost_value_loss"])
        self.cost_rate = per_agent_round(lambda c: c["training_diagnostics"]["train_cost_rate_mean"])
        self.reports = {
            sid: per_agent_round(lambda c, s=sid: next(x["value"] for x in c["reports"] if x["source_id"] == s))
            for sid in [x["source_id"] for x in r[0]["constraints"][self.agent_ids[0]]["reports"]]
        }

        # --- derived ---
        self.bias_gae = self.est_gae - self.true_cost
        self.bias_mc = self.est_mc - self.true_cost
        self.bias_mc_ep = self.est_mc_ep - self.true_cost
        self.bias_mech = self.mechanism - self.true_cost
        self.tail = int(self.n * (1 - TAIL_FRACTION))

    # -- helpers ---------------------------------------------------
    def rmse(self, bias: np.ndarray) -> float:
        return float(np.sqrt((bias**2).mean()))

    def corr_per_agent(self, x: np.ndarray, y: np.ndarray) -> float:
        """Correlation across rounds, computed per agent then averaged --
        never on the agent-pooled flattening, which would let cross-agent
        level differences masquerade as temporal tracking."""
        vals = []
        for i in range(x.shape[1]):
            xi, yi = x[:, i], y[:, i]
            if xi.std() < 1e-12 or yi.std() < 1e-12:
                continue
            vals.append(np.corrcoef(xi, yi)[0, 1])
        return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------
# Gates. Each returns (passed, detail-string).
# ---------------------------------------------------------------------

def gate_g1(s: Seed) -> dict:
    mg, mm = abs(s.bias_gae.mean()), abs(s.bias_mc.mean())
    g1a = mm <= G1A_RELATIVE_MAX * mg
    g1b = mm <= G1B_ABS_MAX
    phase = s.n // G1C_PHASES
    better = []
    for p in range(G1C_PHASES):
        sl = slice(p * phase, (p + 1) * phase if p < G1C_PHASES - 1 else s.n)
        better.append(abs(s.bias_mc[sl].mean()) < abs(s.bias_gae[sl].mean()))
    g1c = sum(better) >= G1C_MIN_PHASES_BETTER
    g1d = abs(s.bias_mech.mean()) <= G1B_ABS_MAX
    return {
        "G1a": (g1a, f"|bias_mc|={mm:.3f} <= {G1A_RELATIVE_MAX}*|bias_gae|={G1A_RELATIVE_MAX * mg:.3f}"),
        "G1b": (g1b, f"|bias_mc|={mm:.3f} <= {G1B_ABS_MAX}"),
        "G1c": (g1c, f"MC better in {sum(better)}/{G1C_PHASES} phases {['Y' if b else 'N' for b in better]}"),
        "G1d": (g1d, f"|bias_mechanism|={abs(s.bias_mech.mean()):.3f} <= {G1B_ABS_MAX}"),
    }


def gate_g2(s: Seed) -> dict:
    exact = float(np.abs(s.reported_cost - s.true_cost).max())
    corr = s.corr_per_agent(s.est_mc, s.true_cost)
    return {
        "G2a": (exact <= G2A_EXACT_TOL, f"max|reported-true| (same rollout) = {exact:.3e} <= {G2A_EXACT_TOL:.0e}"),
        "G2c": (corr >= G2C_CORR_MIN, f"corr(mc, true) = {corr:.3f} >= {G2C_CORR_MIN}"),
    }


def gate_g3(s: Seed, eta: float, lam_max: float, W: np.ndarray) -> dict:
    ident = float(np.abs(s.residual - (s.mechanism - BUDGET)).max())
    # Pre-projection dual value, reconstructed from the logged state.
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
        "G3a": (ident <= G3A_TOL, f"max|residual - (mech - d)| = {ident:.3e} <= {G3A_TOL:.0e}"),
        "G3b": (agree >= G3B_SIGN_MIN, f"sign agreement on {int(nz.sum())} interior cells = {agree:.4f} >= {G3B_SIGN_MIN}"),
        "G3c": (resp > 0, f"corr(residual_k, dlambda) = {resp:.3f} > 0"),
    }


def gate_g4(seeds: list[Seed]) -> dict:
    firsts = [s.task_return[:LEARN_WINDOW] for s in seeds]
    lasts = [s.task_return[-LEARN_WINDOW:] for s in seeds]
    improvements = [float(last.mean() - first.mean()) for first, last in zip(firsts, lasts, strict=True)]
    g4a = all(d > 0 for d in improvements)
    pooled_sd = float(np.sqrt(np.mean([f.var(ddof=1) for f in firsts])))
    mean_imp = float(np.mean(improvements))
    g4b = mean_imp >= G4B_SD_MULTIPLE * pooled_sd
    return {
        "G4a": (g4a, f"per-seed improvement {[round(d, 1) for d in improvements]} all > 0"),
        "G4b": (g4b, f"mean improvement {mean_imp:.1f} >= {G4B_SD_MULTIPLE} x pooled sd {pooled_sd:.1f}"),
    }


def gate_g5(s: Seed) -> dict:
    exercised = bool((s.true_cost > BUDGET).any())
    head = float(s.true_cost[: int(s.n * TAIL_FRACTION)].mean())
    tail = float(s.true_cost[s.tail :].mean())
    return {
        "G5a": (exercised, f"true cost exceeded d={BUDGET} in {int((s.true_cost > BUDGET).any(axis=1).sum())} rounds"),
        "G5b": (tail < head, f"true cost first 20% {head:.2f} -> last 20% {tail:.2f}"),
    }


def gate_g6(s: Seed, lam_max: float) -> dict:
    arrays = [s.true_cost, s.mechanism, s.residual, s.lam, s.est_mc, s.est_gae, s.kl, s.entropy,
              s.value_loss, s.cost_value_loss, s.task_return]
    finite = all(bool(np.isfinite(a).all()) for a in arrays)
    lam_bar = G6B_LAMBDA_FRACTION * lam_max
    lam_max_seen = float(s.lam.max())
    saturated = int((s.lam >= lam_bar).sum())
    kl_tail, kl_max = float(s.kl[s.tail :].mean()), float(s.kl.max())
    ent_tail = float(s.entropy[s.tail :].mean())
    dlam = float(np.abs(np.diff(s.lam, axis=0)).mean())
    return {
        "G6a": (finite, "all logged fields finite" if finite else "NON-FINITE VALUE PRESENT"),
        "G6b": (lam_max_seen < lam_bar and saturated == 0,
                f"max lambda {lam_max_seen:.3f} < {lam_bar} ({lam_max_seen / lam_max:.2f}x lambda_max), "
                f"{saturated} saturated cells"),
        "G6c": (kl_tail <= G6C_KL_TAIL_MAX and kl_max <= G6C_KL_ABS_MAX,
                f"tail mean KL {kl_tail:.4f} <= {G6C_KL_TAIL_MAX}, max KL {kl_max:.4f} <= {G6C_KL_ABS_MAX}"),
        "G6d": (ent_tail > G6D_ENTROPY_MIN, f"tail mean entropy {ent_tail:.3f} > {G6D_ENTROPY_MIN}"),
        "G6e": (dlam <= G6E_DLAMBDA_MAX, f"mean |delta lambda| {dlam:.4f} <= {G6E_DLAMBDA_MAX}"),
    }


def gate_g7(seeds: list[Seed]) -> dict:
    complete = all(
        s.meta.get("status") == "complete" and len(s.rounds) == N_ROUNDS and len(s.oracle) == N_ROUNDS
        and (s.dir / "checkpoint.pt").exists()
        for s in seeds
    )
    shas = {s.meta.get("git", {}).get("sha") for s in seeds}
    dirty = {bool(s.meta.get("git", {}).get("dirty")) for s in seeds}
    estimators = {s.meta.get("config_snapshot", {}).get("constraint_estimator") for s in seeds}
    g7b = len(shas) == 1 and dirty == {False} and estimators == {"mc_window"}
    trends = [float(s.true_cost[s.tail :].mean() - s.true_cost[: int(s.n * TAIL_FRACTION)].mean()) for s in seeds]
    improved = [abs(s.bias_mc.mean()) < abs(s.bias_gae.mean()) for s in seeds]
    g7d = len({t < 0 for t in trends}) == 1 and len(set(improved)) == 1
    return {
        "G7a": (complete, f"{sum(1 for s in seeds if s.meta.get('status') == 'complete')}/{len(seeds)} complete, "
                          f"round counts {[len(s.rounds) for s in seeds]}"),
        "G7b": (g7b, f"sha={list(shas)[0] if len(shas) == 1 else shas}, dirty={dirty}, estimator={estimators}"),
        "G7d": (g7d, f"true-cost trend signs {[('down' if t < 0 else 'up') for t in trends]}, "
                     f"MC-improves-bias {improved}"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True, help="e.g. results/runs_constraint_mc_g1")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if runs_dir.name in ("runs", "runs_g0"):
        raise SystemExit(f"refusing to analyze {runs_dir}: that is an earlier campaign, not G1")
    seed_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir() and (d / "rounds.jsonl").exists())
    if not seed_dirs:
        raise SystemExit(f"no runs found under {runs_dir}")
    seeds = [Seed(d) for d in seed_dirs]

    from safelie.consensus.topologies import build_topology

    cfg0 = seeds[0].meta.get("config_snapshot", {})
    eta = cfg0.get("dual", {}).get("eta_lambda", 0.035)
    lam_max = cfg0.get("dual", {}).get("lambda_max", 25.0)
    topo = cfg0.get("topology", {})
    W = build_topology(topo.get("name", "ring"), topo.get("n_agents", 6),
                       p=topo.get("p"), graph_seed=topo.get("graph_seed", 0))

    print("=" * 78)
    print("G1 DUAL-CONSTRAINT-ESTIMATOR CALIBRATION -- pre-declared gates (docs/g1_gates.md)")
    print("=" * 78)

    # ---- per-seed headline table ----
    print("\n## Seed-by-seed results\n")
    hdr = ("seed", "task(f20)", "task(l20)", "trueC(f20)", "trueC(l20)", "mcC", "gaeC", "mech",
           "b_gae", "b_mc", "b_mech", "lam_mu", "lam_max", "lam_act", "viol", "peak")
    print(" ".join(f"{h:>10s}" for h in hdr))
    for s in seeds:
        head_n = int(s.n * TAIL_FRACTION)
        row = (s.name.replace("pilot_A_clean_", ""),
               f"{s.task_return[:LEARN_WINDOW].mean():.1f}", f"{s.task_return[-LEARN_WINDOW:].mean():.1f}",
               f"{s.true_cost[:head_n].mean():.2f}", f"{s.true_cost[s.tail:].mean():.2f}",
               f"{s.est_mc.mean():.2f}", f"{s.est_gae.mean():.2f}", f"{s.mechanism.mean():.2f}",
               f"{s.bias_gae.mean():.2f}", f"{s.bias_mc.mean():.2f}", f"{s.bias_mech.mean():.2f}",
               f"{s.lam.mean():.3f}", f"{s.lam.max():.3f}", f"{(s.lam > 0).mean():.3f}",
               f"{s.violated.mean():.3f}", f"{s.peak_true_cost.max():.2f}")
        print(" ".join(f"{v:>10s}" for v in row))

    # ---- estimator comparison ----
    print("\n## Estimator comparison (same rounds, same oracle episodes)\n")
    print(f"{'seed':>8s} {'bias_gae':>10s} {'bias_mc':>10s} {'bias_mcep':>10s} "
          f"{'rmse_gae':>10s} {'rmse_mc':>10s} {'corr_gae':>10s} {'corr_mc':>10s} {'reduction':>10s}")
    for s in seeds:
        red = 1 - abs(s.bias_mc.mean()) / abs(s.bias_gae.mean()) if s.bias_gae.mean() else float("nan")
        print(f"{s.name.replace('pilot_A_clean_', ''):>8s} "
              f"{s.bias_gae.mean():>10.3f} {s.bias_mc.mean():>10.3f} {s.bias_mc_ep.mean():>10.3f} "
              f"{s.rmse(s.bias_gae):>10.3f} {s.rmse(s.bias_mc):>10.3f} "
              f"{s.corr_per_agent(s.est_gae, s.true_cost):>10.3f} {s.corr_per_agent(s.est_mc, s.true_cost):>10.3f} "
              f"{red:>9.1%}")

    # ---- per-phase bias ----
    print("\n## Bias by training phase (50-round blocks): gae / mc\n")
    for s in seeds:
        phase = s.n // G1C_PHASES
        cells = []
        for p in range(G1C_PHASES):
            sl = slice(p * phase, (p + 1) * phase if p < G1C_PHASES - 1 else s.n)
            cells.append(f"{s.bias_gae[sl].mean():+.1f}/{s.bias_mc[sl].mean():+.1f}")
        print(f"{s.name.replace('pilot_A_clean_', ''):>8s}  " + "  ".join(f"{c:>14s}" for c in cells))

    # ---- variance analysis (section 13) ----
    print("\n## MC variance analysis (reported, gates nothing)\n")
    print(f"{'seed':>8s} {'var_gae':>10s} {'var_mc':>10s} {'var_true':>10s} {'var_mech':>10s} "
          f"{'mean|dlam|':>11s} {'sd_resid':>10s}")
    for s in seeds:
        print(f"{s.name.replace('pilot_A_clean_', ''):>8s} "
              f"{s.est_gae.var():>10.1f} {s.est_mc.var():>10.1f} {s.true_cost.var():>10.1f} "
              f"{s.mechanism.var():>10.1f} {np.abs(np.diff(s.lam, axis=0)).mean():>11.4f} "
              f"{s.residual.std():>10.2f}")

    # ---- per-source behaviour ----
    print("\n## Per-source reports (whole-run mean; peer_critic sources are UNCHANGED by this repair)\n")
    for s in seeds:
        parts = " ".join(f"{sid}={v.mean():.2f}" for sid, v in s.reports.items())
        print(f"  {s.name.replace('pilot_A_clean_', ''):>8s}  {parts}")

    # ---- PPO health ----
    print("\n## PPO health (tail = last 20% of rounds)\n")
    print(f"{'seed':>8s} {'kl_tail':>10s} {'kl_max':>10s} {'ent_f20':>10s} {'ent_tail':>10s} "
          f"{'vloss_tail':>11s} {'cvloss_tail':>12s}")
    for s in seeds:
        print(f"{s.name.replace('pilot_A_clean_', ''):>8s} "
              f"{s.kl[s.tail:].mean():>10.4f} {s.kl.max():>10.4f} "
              f"{s.entropy[:LEARN_WINDOW].mean():>10.3f} {s.entropy[s.tail:].mean():>10.3f} "
              f"{s.value_loss[s.tail:].mean():>11.4f} {s.cost_value_loss[s.tail:].mean():>12.4f}")

    # ---- gate evaluation ----
    print("\n" + "=" * 78)
    print("GATE RESULTS")
    print("=" * 78)
    per_seed_gates: dict[str, dict[str, tuple[bool, str]]] = {}
    for s in seeds:
        g: dict[str, tuple[bool, str]] = {}
        g.update(gate_g1(s))
        g.update(gate_g2(s))
        g.update(gate_g3(s, eta, lam_max, W))
        g.update(gate_g5(s))
        g.update(gate_g6(s, lam_max))
        per_seed_gates[s.name] = g

    cross = {}
    cross.update(gate_g4(seeds))
    cross.update(gate_g7(seeds))

    for s in seeds:
        print(f"\n{s.name}:")
        for k, (ok, detail) in per_seed_gates[s.name].items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {k}: {detail}")
    print("\ncross-seed:")
    for k, (ok, detail) in cross.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}: {detail}")

    # ---- roll up to the seven gates ----
    groups = {"G1": ["G1a", "G1b", "G1c", "G1d"], "G2": ["G2a", "G2c"], "G3": ["G3a", "G3b", "G3c"],
              "G4": ["G4a", "G4b"], "G5": ["G5a", "G5b"], "G6": ["G6a", "G6b", "G6c", "G6d", "G6e"],
              "G7": ["G7a", "G7b", "G7d"]}
    print("\n" + "=" * 78)
    print(f"{'gate':>6s} " + " ".join(f"{s.name.replace('pilot_A_clean_', ''):>10s}" for s in seeds) + f" {'overall':>10s}")
    print("=" * 78)
    summary = {}
    for gname, subs in groups.items():
        per = []
        for s in seeds:
            if gname in ("G4", "G7"):
                per.append(all(cross[x][0] for x in subs if x in cross))
            else:
                per.append(all(per_seed_gates[s.name][x][0] for x in subs))
        overall = all(per)
        summary[gname] = overall
        print(f"{gname:>6s} " + " ".join(f"{('PASS' if p else 'FAIL'):>10s}" for p in per) + f" {('PASS' if overall else 'FAIL'):>10s}")

    failed = [g for g, ok in summary.items() if not ok]
    hard_fail = [x for x in ("G1a", "G2a", "G3a")
                 if any(not per_seed_gates[s.name].get(x, (True, ""))[0] for s in seeds)]
    print("\n" + "=" * 78)
    if not failed:
        verdict = "PASS"
    elif hard_fail or len(failed) >= 2:
        verdict = "FAIL"
    else:
        verdict = "CONDITIONAL PASS"
    print(f"VERDICT (mechanical, per docs/g1_gates.md): {verdict}")
    if failed:
        print(f"  failed gates: {', '.join(failed)}")
    if hard_fail:
        print(f"  estimator-implicating failures (automatic FAIL): {', '.join(hard_fail)}")
    print("=" * 78)

    # numpy comparisons yield np.bool_, which json refuses; coerce every
    # flag to a Python bool on the way out rather than hoping none of the
    # gate expressions ever touches a numpy scalar.
    out = runs_dir / "g1_gates.json"
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
