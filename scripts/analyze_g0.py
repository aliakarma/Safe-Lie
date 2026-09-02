#!/usr/bin/env python
"""Evaluate the G0 clean-baseline validation against its pre-declared gates.

Usage:
    python scripts/analyze_g0.py --runs-dir results/runs_g0

Every threshold, window and direction below is transcribed from
`docs/g0_gates.md`, which was written and committed BEFORE any G0 run
existed. This script does not choose criteria; it applies them. If a
number here disagrees with that document, the document is right and this
script is a bug.

Two rules this file exists to enforce mechanically:

  1. Gate arithmetic reads ONLY the evaluation quantities
     (`episodic_task_return`, `true_cost_return`,
     `episodic_reported_cost_return` from `oracle.jsonl`, and
     `mechanism_reported_cost_return` from `rounds.jsonl`). The learner's
     GAE(lambda) training targets `ret_r[0]`/`ret_c[0]` -- logged as
     `task_return`/`reported_cost_return` in `rounds.jsonl` -- are
     printed only in the clearly-labelled diagnostics block and are
     never substituted into a gate.
  2. The runs directory is an explicit argument with no default that
     could reach `results/runs`, which still holds the pre-P0-repair
     matrix under the very same run-directory names.
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
LEARN_WINDOW = 20          # Gate L: first/last 20 rounds
TAIL_FRACTION = 0.20       # Gates C/K/D: last 20% of rounds
GATE_L2_SD_MULTIPLE = 1.0  # Gate L2: improvement >= 1.0 sd of the first window
GATE_K_BIAS_MAX = 0.20 * BUDGET   # Gate K1/K2: |bias| <= 5.0
GATE_D_ACTIVATION_MIN = 0.25      # Gate D1: lambda > 0 in >= 25% of cells


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


class Seed:
    """One G0 seed's logs, reduced to per-round arrays.

    Shapes are (n_rounds,) for run-level scalars and (n_rounds, n_agents)
    for per-agent quantities, with agents in `self.agent_ids` order
    throughout so a row index means the same agent in every array.
    """

    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.name = run_dir.name
        self.seed = int(run_dir.name.rsplit("seed", 1)[1])
        rounds = _read_jsonl(run_dir / "rounds.jsonl")
        oracle = _read_jsonl(run_dir / "oracle.jsonl")
        meta_path = run_dir / "run_metadata.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        # A run sampled between the learner's write and the orchestrator's
        # is legitimately one record ahead; truncate to the common prefix
        # so every array below indexes the same rounds.
        n = min(len(rounds), len(oracle))
        self.n_rounds_logged = (len(rounds), len(oracle))
        rounds, oracle = rounds[:n], oracle[:n]
        self.n = n
        if n == 0:
            self.agent_ids = []
            return
        self.agent_ids = sorted(rounds[0]["constraints"].keys())
        A = self.agent_ids

        def per_agent(recs, block, fn):
            return np.array([[fn(r[block][a]) for a in A] for r in recs], dtype=float)

        # --- Evaluation quantities (oracle.jsonl) -- gate-admissible.
        self.episodic_task_return = np.array(
            [r["agents"][A[0]]["episodic_task_return"] for r in oracle], dtype=float
        )  # shared scalar reward, identical across agents
        self.true_cost = per_agent(oracle, "agents", lambda d: d["true_cost_return"])
        self.episodic_reported_cost = per_agent(
            oracle, "agents", lambda d: d["episodic_reported_cost_return"]
        )
        self.violated = per_agent(oracle, "agents", lambda d: float(bool(d["violated"])))
        self.peak_true_cost = per_agent(oracle, "agents", lambda d: d["peak_true_cost"])

        # --- Mechanism / dual quantities (rounds.jsonl) -- gate-admissible.
        self.mechanism_cost = per_agent(
            rounds, "constraints", lambda d: d["mechanism_reported_cost_return"]
        )
        self.own_critic_cost = per_agent(
            rounds, "constraints", lambda d: d["reported_cost_return"]
        )
        self.lam = per_agent(rounds, "constraints", lambda d: d["lambda_after"])
        self.residual = per_agent(rounds, "constraints", lambda d: d["constraint_residual"])

        # --- Learner training diagnostics -- NEVER gate-admissible.
        self.train_task_return = per_agent(rounds, "constraints", lambda d: d["task_return"])
        self.train_reward = per_agent(
            rounds, "constraints", lambda d: d["training_diagnostics"]["train_reward_mean"]
        )
        self.train_cost_rate = per_agent(
            rounds, "constraints", lambda d: d["training_diagnostics"]["train_cost_rate_mean"]
        )
        for key in ("policy_loss", "value_loss", "cost_value_loss", "entropy", "approx_kl"):
            setattr(self, key, per_agent(rounds, "constraints", lambda d, k=key: d["ppo"][k]))

        # --- Per-source reports, for the calibration/independence view.
        self.source_ids = [s["source_id"] for s in rounds[0]["constraints"][A[0]]["reports"]]
        self.source_values = np.array(
            [[[s["value"] for s in r["constraints"][a]["reports"]] for a in A] for r in rounds],
            dtype=float,
        )  # (n_rounds, n_agents, M)

    # -- windows -------------------------------------------------------
    @property
    def tail(self) -> slice:
        return slice(self.n - int(round(TAIL_FRACTION * self.n)), self.n)

    @property
    def head(self) -> slice:
        return slice(0, int(round(TAIL_FRACTION * self.n)))


def gate_L(seeds: list[Seed]) -> dict:
    """Learning: last-20-round vs first-20-round mean episodic task return."""
    per_seed = {}
    for s in seeds:
        first = s.episodic_task_return[:LEARN_WINDOW]
        last = s.episodic_task_return[s.n - LEARN_WINDOW : s.n]
        per_seed[s.seed] = {
            "first20_mean": float(first.mean()),
            "first20_sd": float(first.std(ddof=1)),
            "last20_mean": float(last.mean()),
            "improvement": float(last.mean() - first.mean()),
        }
    l1 = all(v["improvement"] > 0 for v in per_seed.values())
    pooled_sd = float(np.sqrt(np.mean([v["first20_sd"] ** 2 for v in per_seed.values()])))
    mean_improvement = float(np.mean([v["improvement"] for v in per_seed.values()]))
    effect = mean_improvement / pooled_sd if pooled_sd > 0 else float("inf")
    l2 = effect >= GATE_L2_SD_MULTIPLE
    return {
        "per_seed": per_seed, "pooled_first20_sd": pooled_sd,
        "mean_improvement": mean_improvement, "effect_sizes_sd": effect,
        "L1": l1, "L2": l2, "pass": l1 and l2,
    }


def gate_C(seeds: list[Seed]) -> dict:
    """Constraint: exercised at all (C1), and trending toward d (C2)."""
    per_seed = {}
    for s in seeds:
        head = s.true_cost[s.head].mean()
        tail = s.true_cost[s.tail].mean()
        per_seed[s.seed] = {
            "whole_run_mean": float(s.true_cost.mean()),
            "max_any_agent_round": float(s.true_cost.max()),
            "exceeds_d_somewhere": bool((s.true_cost > BUDGET).any()),
            "head20pct_mean": float(head),
            "tail20pct_mean": float(tail),
            "delta": float(tail - head),
            "tail_below_d": bool(tail < BUDGET),
            "violation_rate_whole_run": float(s.violated.mean()),
            "peak_violation": float(s.peak_true_cost.max()),
        }
    c1 = all(v["exceeds_d_somewhere"] for v in per_seed.values())
    deltas = [v["delta"] for v in per_seed.values()]
    c2 = all(d < 0 for d in deltas)
    return {
        "per_seed": per_seed, "C1": c1, "C2": c2, "pass": c1 and c2,
        "trend_signs": [int(np.sign(d)) for d in deltas],
    }


def gate_K(seeds: list[Seed]) -> dict:
    """Cost-critic calibration: bias = estimate - true_cost_return."""
    per_seed = {}
    for s in seeds:
        own_bias = s.own_critic_cost - s.true_cost
        mech_bias = s.mechanism_cost - s.true_cost
        d = {}
        for label, b in (("own", own_bias), ("mechanism", mech_bias)):
            d[f"{label}_head_bias"] = float(b[s.head].mean())
            d[f"{label}_tail_bias"] = float(b[s.tail].mean())
            d[f"{label}_tail_rmse"] = float(np.sqrt((b[s.tail] ** 2).mean()))
            d[f"{label}_improved"] = abs(float(b[s.tail].mean())) < abs(float(b[s.head].mean()))
        per_seed[s.seed] = d
    k1 = all(abs(v["own_tail_bias"]) <= GATE_K_BIAS_MAX for v in per_seed.values())
    k2 = all(abs(v["mechanism_tail_bias"]) <= GATE_K_BIAS_MAX for v in per_seed.values())
    k3 = all(v["own_improved"] and v["mechanism_improved"] for v in per_seed.values())
    return {"per_seed": per_seed, "K1": k1, "K2": k2, "K3": k3, "pass": k1 and k2 and k3}


def gate_D(seeds: list[Seed]) -> dict:
    """Dual: lambda activation over the run, and sustained in the tail."""
    per_seed = {}
    for s in seeds:
        per_seed[s.seed] = {
            "activation_fraction": float((s.lam > 0).mean()),
            "mean_lambda": float(s.lam.mean()),
            "max_lambda": float(s.lam.max()),
            "tail_mean_lambda": float(s.lam[s.tail].mean()),
            "saturated_fraction": float((s.lam >= 24.999).mean()),
            "per_agent_mean": [float(x) for x in s.lam.mean(axis=0)],
        }
    d1 = all(v["activation_fraction"] >= GATE_D_ACTIVATION_MIN for v in per_seed.values())
    d2 = all(v["tail_mean_lambda"] > 0 for v in per_seed.values())
    return {"per_seed": per_seed, "D1": d1, "D2": d2, "pass": d1 and d2}


def gate_R(seeds: list[Seed], c: dict) -> dict:
    """Reproducibility: completion, log integrity, cross-seed consistency."""
    per_seed = {}
    for s in seeds:
        nr, no = s.n_rounds_logged
        per_seed[s.seed] = {
            "status": s.meta.get("status"),
            "rounds_records": nr, "oracle_records": no,
            "complete": s.meta.get("status") == "complete" and nr == N_ROUNDS and no == N_ROUNDS,
            "checkpoint": (s.dir / "checkpoint.pt").exists(),
            "git_sha": (s.meta.get("git") or {}).get("sha"),
            "git_dirty": (s.meta.get("git") or {}).get("dirty"),
            "seed_recorded": s.meta.get("seed"),
        }
    r1 = all(v["complete"] and v["checkpoint"] for v in per_seed.values())
    signs = set(c["trend_signs"])
    r3 = len(signs) == 1
    return {"per_seed": per_seed, "R1": r1, "R3": r3, "pass": r1 and r3,
            "note_R2": "integrity_check is scripts/analyze_matrix.py --runs-dir; run separately"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", required=True,
                    help="Directory holding the G0 run directories (results/runs_g0). "
                         "Required, with no default: results/runs still holds the "
                         "pre-P0-repair matrix under identical run-directory names.")
    ap.add_argument("--prefix", default="pilot_A_clean_seed")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    dirs = sorted(d for d in runs_dir.glob(f"{args.prefix}*") if d.is_dir())
    if not dirs:
        print(f"No runs matching {args.prefix}* in {runs_dir}", file=sys.stderr)
        return 2
    seeds = [Seed(d) for d in dirs]
    seeds = [s for s in seeds if s.n > 0]

    print("=" * 78)
    print(f"G0 CLEAN-BASELINE VALIDATION -- {runs_dir}")
    print("Gates transcribed from docs/g0_gates.md (declared before any G0 run existed).")
    print("=" * 78)
    for s in seeds:
        nr, no = s.n_rounds_logged
        print(f"  {s.name}: rounds={nr} oracle={no} status={s.meta.get('status')} "
              f"agents={len(s.agent_ids)} sources={len(s.source_ids)}")

    L, C, K, D = gate_L(seeds), gate_C(seeds), gate_K(seeds), gate_D(seeds)
    R = gate_R(seeds, C)

    def block(title, g, rows):
        print(f"\n{'-' * 78}\n{title}   ->  {'PASS' if g['pass'] else 'FAIL'}\n{'-' * 78}")
        for s in seeds:
            v = g["per_seed"][s.seed]
            print(f"  seed {s.seed}: " + "  ".join(
                f"{k}={v[k]:.3f}" if isinstance(v[k], float) else f"{k}={v[k]}" for k in rows
            ))

    block("GATE L -- learning (episodic_task_return)", L,
          ["first20_mean", "last20_mean", "improvement", "first20_sd"])
    print(f"  L1 all-seeds improvement>0 : {L['L1']}")
    print(f"  L2 mean improvement {L['mean_improvement']:.2f} / pooled sd "
          f"{L['pooled_first20_sd']:.2f} = {L['effect_sizes_sd']:.2f} sd "
          f"(>= {GATE_L2_SD_MULTIPLE}): {L['L2']}")

    block("GATE C -- constraint (true_cost_return, d=25)", C,
          ["whole_run_mean", "head20pct_mean", "tail20pct_mean", "delta",
           "violation_rate_whole_run", "max_any_agent_round"])
    print(f"  C1 constraint exercised (any round > d) : {C['C1']}")
    print(f"  C2 tail < head in every seed            : {C['C2']}  signs={C['trend_signs']}")
    print("  REPORTED, NOT GATED -- tail mean below d: "
          + ", ".join(f"seed{s.seed}={C['per_seed'][s.seed]['tail_below_d']}" for s in seeds))

    block("GATE K -- cost-critic calibration (bias = estimate - true)", K,
          ["own_head_bias", "own_tail_bias", "mechanism_head_bias",
           "mechanism_tail_bias", "own_tail_rmse", "mechanism_tail_rmse"])
    print(f"  K1 |own tail bias|      <= {GATE_K_BIAS_MAX}: {K['K1']}")
    print(f"  K2 |mechanism tail bias| <= {GATE_K_BIAS_MAX}: {K['K2']}")
    print(f"  K3 tail |bias| < head |bias|, both quantities: {K['K3']}")

    block("GATE D -- dual dynamics (lambda)", D,
          ["activation_fraction", "mean_lambda", "max_lambda",
           "tail_mean_lambda", "saturated_fraction"])
    print(f"  D1 activation >= {GATE_D_ACTIVATION_MIN:.0%} of cells: {D['D1']}")
    print(f"  D2 tail mean lambda > 0                : {D['D2']}")

    block("GATE R -- reproducibility", R,
          ["status", "rounds_records", "oracle_records", "complete", "checkpoint", "git_sha"])
    print(f"  R1 all complete with checkpoint : {R['R1']}")
    print(f"  R3 consistent trend direction   : {R['R3']}")
    print(f"  R2 -- {R['note_R2']}")

    results = {"L": L["pass"], "C": C["pass"], "K": K["pass"], "D": D["pass"], "R": R["pass"]}
    n_fail = sum(1 for v in results.values() if not v)
    verdict = "PASS" if n_fail == 0 else ("CONDITIONAL PASS (pending cause analysis)"
                                          if n_fail == 1 else "FAIL")
    print("\n" + "=" * 78)
    print("GATE SUMMARY: " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items()))
    print(f"MECHANICAL VERDICT: {verdict}")
    print("(CONDITIONAL PASS requires an identified cause that does not implicate the")
    print(" learner itself -- docs/g0_gates.md's verdict rule. This script cannot judge that.)")
    print("=" * 78)

    print("\n--- LEARNER TRAINING DIAGNOSTICS (never gate-admissible) ---")
    for s in seeds:
        print(f"  seed {s.seed}: policy_loss {s.policy_loss[:20].mean():+.4f} -> "
              f"{s.policy_loss[s.tail].mean():+.4f} | value_loss "
              f"{s.value_loss[:20].mean():.4f} -> {s.value_loss[s.tail].mean():.4f} | "
              f"cost_value_loss {s.cost_value_loss[:20].mean():.4f} -> "
              f"{s.cost_value_loss[s.tail].mean():.4f}")
        print(f"          entropy {s.entropy[:20].mean():.3f} -> {s.entropy[s.tail].mean():.3f} | "
              f"approx_kl {s.approx_kl[:20].mean():.4f} -> {s.approx_kl[s.tail].mean():.4f} | "
              f"train_reward {s.train_reward[:20].mean():+.3f} -> "
              f"{s.train_reward[s.tail].mean():+.3f}")
        print(f"          ret_c[0] (TRAINING TARGET, not an eval metric) "
              f"{s.own_critic_cost[:20].mean():.2f} -> {s.own_critic_cost[s.tail].mean():.2f}")

    if args.json_out:
        payload = {"runs_dir": str(runs_dir), "gates": {"L": L, "C": C, "K": K, "D": D, "R": R},
                   "summary": results, "mechanical_verdict": verdict}
        Path(args.json_out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nJSON written to {args.json_out}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
