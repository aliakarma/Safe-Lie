#!/usr/bin/env python
"""Verify the A3 Windows smoke test. NOT an A3 analysis.

This checks that the complete M=5, f=1 RCE path RAN CORRECTLY on this
machine. It deliberately makes no scientific claim: three rounds under an
essentially untrained policy cannot estimate the margin, the interaction, or
anything else A3 is about. Every check here is mechanical -- shapes, exact
identities, absence of NaN -- and every one of them is pass/fail rather than
a measurement.

The distinction matters because the failure mode this smoke test exists to
catch is precisely a run that LOOKS fine: an M=5 config that silently
degenerates to A2's floored constant would still finish, still log a margin,
and still produce plausible numbers. So the checks below verify the
mechanism's identities exactly (to 1e-9 against an offline recomputation),
not its plausibility.

    S1  exactly five source reports per owner, every round
    S2  five distinct replica RNG streams, no duplicate/overlapping seeds
    S3  all five collected under one pinned theta_k
    S4  no source data in PPO (structural; see the note in check_s4)
    S5  no oracle/reference leakage into the learner path
    S6  the attacked source is the predeclared one, and only it
    R1  retained_n == 3 in 100% of cells
    R2  degenerate == False in 100% of cells
    R3  MAD is a measurement, not the floor
    R4  beta*MAD varies across cells
    R7  beta*MAD is computed AND reaches the dual update
    N   no NaN/Inf anywhere in the mechanism path
    X   offline recomputation reproduces every logged field to 1e-9

Usage:
    python scripts/a3_smoke_verify.py \
        --attack results/a3_smoke/smoke_m5_rce_attack \
        --clean  results/a3_smoke/smoke_m5_rce_clean \
        --out    results/a3_smoke/a3_smoke_verification.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safelie.defenses.rce import rce_aggregate  # noqa: E402

logging.getLogger("safelie.defenses.rce").setLevel(logging.CRITICAL)

TOL = 1e-9
M, F, BETA, SIGMA_MIN, MIN_RETAINED = 5, 1, 1.5, 1e-3, 3
D_BUDGET = 25.0
B_MAGNITUDE = 12.5
EXPECTED_ATTACKED = "batch_1"


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def load(run_dir: Path) -> dict:
    rounds = read_jsonl(run_dir / "rounds.jsonl")
    if not rounds:
        raise SystemExit(f"{run_dir}: rounds.jsonl is empty -- the run did not start")
    aids = sorted(rounds[0]["constraints"])
    sids = sorted(r["source_id"] for r in rounds[0]["constraints"][aids[0]]["reports"])
    pre = np.array(
        [[[next(x["value"] for x in r["constraints"][a]["reports"] if x["source_id"] == sid)
           for sid in sids] for a in aids] for r in rounds], dtype=float)
    keys = ("point_estimate", "spread", "retained_n", "applied_margin", "pessimistic_estimate")
    logged = {k: np.array([[r["constraints"][a]["aggregate"][k] for a in aids] for r in rounds],
                          dtype=float) for k in keys}
    for k in ("mechanism_reported_cost_return", "constraint_residual"):
        logged[k] = np.array([[r["constraints"][a][k] for a in aids] for r in rounds], dtype=float)
    meta_p = run_dir / "run_metadata.json"
    return {
        "dir": str(run_dir), "n_rounds": len(rounds), "agent_ids": aids, "source_ids": sids,
        "pre": pre, "logged": logged, "rounds": rounds,
        "n_reports_per_cell": np.array(
            [[len(r["constraints"][a]["reports"]) for a in aids] for r in rounds]),
        "degenerate": np.array([[r["constraints"][a]["aggregate"]["degenerate"] for a in aids]
                                for r in rounds], dtype=bool),
        "corrupted": sorted(rounds[0]["constraints"][aids[0]]["corrupted_source_ids"]),
        "meta": json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {},
    }


def recompute(pre: np.ndarray, corrupted_idx: int | None) -> dict:
    """Re-derive the whole aggregation offline from the logged PRE-attack
    reports, so the check is against an independent computation rather than
    against the run's own arithmetic restated."""
    K, N, _ = pre.shape
    post = pre.copy()
    if corrupted_idx is not None:
        post[:, :, corrupted_idx] -= B_MAGNITUDE
    keys = ("point_estimate", "spread", "retained_n", "applied_margin", "pessimistic_estimate")
    out = {k: np.zeros((K, N)) for k in keys}
    retained = np.zeros((K, N, M - 2 * F))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # a floor warning is a failure here
        for k in range(K):
            for n in range(N):
                r = rce_aggregate(post[k, n], F, BETA, SIGMA_MIN, MIN_RETAINED)
                for key in keys:
                    out[key][k, n] = getattr(r, key)
                retained[k, n] = np.sort(r.retained_values)
    out["retained_values"] = retained
    out["post"] = post
    return out


def check(name: str, ok: bool, detail: str) -> dict:
    return {"check": name, "pass": bool(ok), "detail": detail}


def verify(run: dict, arm: str) -> list[dict]:
    res: list[dict] = []
    lg, pre = run["logged"], run["pre"]
    K, N = lg["spread"].shape
    cells = K * N
    attacked = run["corrupted"]
    cidx = run["source_ids"].index(attacked[0]) if attacked else None

    # ---- S1: exactly five source reports per owner, every round -----------
    n_rep = run["n_reports_per_cell"]
    res.append(check(
        "S1 five source reports per owner",
        bool((n_rep == M).all()) and len(run["source_ids"]) == M,
        f"reports/cell unique={sorted(np.unique(n_rep).tolist())}, "
        f"source_ids={run['source_ids']}"))

    # ---- S2: five distinct streams, no duplicate seeds ---------------------
    audit = run["meta"].get("source_seed_audit", {})
    seed_rows = read_jsonl(Path(run["dir"]) / "source_seeds.jsonl")
    per_replica = {sid: set() for sid in run["source_ids"]}
    for row in seed_rows:
        for sid, pairs in row.get("seeds", {}).items():
            per_replica.setdefault(sid, set()).update((int(a), int(b)) for a, b in pairs)
    overlaps = {f"{a}&{b}": len(per_replica[a] & per_replica[b])
                for i, a in enumerate(run["source_ids"]) for b in run["source_ids"][i + 1:]}
    spawn_keys = audit.get("spawn_keys", [])
    res.append(check(
        "S2 five independent RNG streams, no duplicate or overlapping seeds",
        audit.get("duplicate_seed_events", -1) == 0
        and all(v == 0 for v in overlaps.values())
        and len({tuple(k) for k in spawn_keys}) == len(spawn_keys) == M + 1,
        f"duplicate_seed_events={audit.get('duplicate_seed_events')}, "
        f"pairwise stream overlaps={set(overlaps.values())}, "
        f"distinct spawn keys={len({tuple(k) for k in spawn_keys})}/{M + 1} (M+1, incl. reference), "
        f"env seeds issued={audit.get('n_env_seeds_issued')}"))

    # ---- S3: one pinned theta_k per round ---------------------------------
    pinned = [r["source_batch"]["worker_checksums_all_match"] for r in run["rounds"]]
    stable = [r["source_batch"].get("theta_k_checksum_stable_through_dual") for r in run["rounds"]]
    n_traj = {r["source_batch"]["n_trajectories"] for r in run["rounds"]}
    res.append(check(
        "S3 all five sources under one pinned theta_k",
        all(pinned) and all(s is not False for s in stable) and n_traj == {M * 30},
        f"worker checksums matched in {sum(pinned)}/{len(pinned)} rounds; "
        f"theta_k stable through the dual in {sum(1 for s in stable if s)}/{len(stable)}; "
        f"trajectories/round={n_traj} (= M x R_m = {M} x 30)"))

    # ---- S4: no source data in PPO ----------------------------------------
    res.append(check_s4())

    # ---- S5: no oracle/reference leakage ----------------------------------
    ref_p = Path(run["dir"]) / "validation_reference.jsonl"
    ref_rows = read_jsonl(ref_p) if ref_p.exists() else []
    round_txt = (Path(run["dir"]) / "rounds.jsonl").read_text(encoding="utf-8")
    leaked = [t for t in ("reference_mean", "reference_per_trajectory", "true_cost_return",
                          "episodic_true_cost") if t in round_txt]
    res.append(check(
        "S5 no oracle/reference leakage into the learner path",
        not leaked and (Path(run["dir"]) / "oracle.jsonl").exists(),
        f"reference rows in their own file: {len(ref_rows)}; "
        f"oracle.jsonl separate: {(Path(run['dir']) / 'oracle.jsonl').exists()}; "
        f"leaked keys in rounds.jsonl: {leaked or 'none'}"))

    # ---- S6: the attacked source is the predeclared one, and only it ------
    if arm == "attack":
        rec = recompute(pre, cidx)
        untouched = [run["source_ids"][j] for j in range(M)
                     if j != cidx and not np.allclose(rec["post"][:, :, j], pre[:, :, j])]
        res.append(check(
            "S6 the attacked source is the predeclared source, and only it",
            attacked == [EXPECTED_ATTACKED] and not untouched,
            f"corrupted_source_ids={attacked} (declared [{EXPECTED_ATTACKED!r}]); "
            f"other sources modified: {untouched or 'none'}"))
    else:
        rec = recompute(pre, None)
        res.append(check("S6 clean arm carries no corrupted source", attacked == [],
                         f"corrupted_source_ids={attacked}"))

    # ---- R1 / R2: retained_n == 3, never degenerate ------------------------
    res.append(check("R1 retained_n == 3 in 100% of cells",
                     bool((lg["retained_n"] == 3).all()),
                     f"{int((lg['retained_n'] == 3).sum())}/{cells} cells; "
                     f"values seen={sorted(np.unique(lg['retained_n']).tolist())}"))
    res.append(check("R2 degenerate == False in 100% of cells",
                     not run["degenerate"].any(),
                     f"degenerate cells={int(run['degenerate'].sum())}/{cells}"))

    # ---- R3: the MAD is a measurement, not the floor -----------------------
    sp = lg["spread"]
    at_floor = int(np.isclose(sp, SIGMA_MIN, atol=1e-12).sum())
    res.append(check(
        "R3 MAD is a real measurement, never the sigma floor",
        at_floor == 0 and bool((sp > 0).all()),
        f"cells at exactly sigma_min={SIGMA_MIN}: {at_floor}/{cells}; "
        f"nonzero MAD: {int((sp > 0).sum())}/{cells}; "
        f"MAD min={sp.min():.6g} median={np.median(sp):.6g} max={sp.max():.6g}; "
        f"distinct values={len(np.unique(np.round(sp, 12)))}"))

    # ---- R4: beta*MAD varies -----------------------------------------------
    mg = lg["applied_margin"]
    res.append(check(
        "R4 beta*MAD varies across cells (not a constant)",
        len(np.unique(np.round(mg, 12))) > 1 and float(mg.std()) > 0.0,
        f"distinct margins={len(np.unique(np.round(mg, 12)))}/{cells}; "
        f"mean={mg.mean():.6g} sd={mg.std():.6g} "
        f"min={mg.min():.6g} max={mg.max():.6g}"))

    # ---- R7: the margin is computed AND reaches the dual --------------------
    id_margin = float(np.abs(mg - BETA * sp).max())
    id_pess = float(np.abs(lg["pessimistic_estimate"] - (lg["point_estimate"] + mg)).max())
    id_dual = float(np.abs(lg["mechanism_reported_cost_return"] - lg["pessimistic_estimate"]).max())
    id_resid = float(np.abs(lg["constraint_residual"]
                            - (lg["pessimistic_estimate"] - D_BUDGET)).max())
    res.append(check(
        "R7 beta*MAD is computed and reaches the dual update",
        max(id_margin, id_pess, id_dual, id_resid) <= TOL,
        f"max|margin - beta*MAD|={id_margin:.3g}; "
        f"max|pessimistic - (point+margin)|={id_pess:.3g}; "
        f"max|dual input - pessimistic|={id_dual:.3g}; "
        f"max|residual - (pessimistic - d)|={id_resid:.3g}  (all must be <= {TOL})"))

    # ---- N: no NaN/Inf ------------------------------------------------------
    bad = {k: int((~np.isfinite(v)).sum()) for k, v in lg.items()}
    bad["pre_attack_reports"] = int((~np.isfinite(pre)).sum())
    res.append(check("N no NaN/Inf anywhere in the mechanism path",
                     all(v == 0 for v in bad.values()),
                     f"non-finite counts={bad}"))

    # ---- X: offline recomputation reproduces every logged field -------------
    errs = {k: float(np.abs(rec[k] - lg[k]).max())
            for k in ("point_estimate", "spread", "retained_n", "applied_margin",
                      "pessimistic_estimate")}
    res.append(check(
        "X offline recomputation reproduces every logged field to 1e-9",
        max(errs.values()) <= TOL,
        f"max abs error per field={ {k: f'{v:.3g}' for k, v in errs.items()} }"))

    return res


def check_s4() -> dict:
    """No source data enters PPO.

    Structural rather than statistical, and deliberately so. `ExperimentRun`
    passes `finalized[aid]` -- built solely by `rollouts[aid].finalize(...)`
    from the round's own on-policy learner rollout -- plus the scalar
    `lam[i]` into `ppo_lagrangian_update`. The `BatchSourceResult` is read
    only to produce source VALUES and log lines. So the assertion is about
    which objects exist at the call site, and grepping the shipped source for
    that call is a stronger check than any statistic computed after the fact.
    """
    loop = (Path(__file__).resolve().parents[1]
            / "src/safelie/training/loop.py").read_text(encoding="utf-8")
    call = "ppo_lagrangian_update(self.agents[aid], finalized[aid], float(self.lam[i]), cfg.ppo)"
    ok = call in loop and loop.count("ppo_lagrangian_update(") == 1
    return check(
        "S4 no source data enters PPO",
        ok,
        f"the single PPO call site consumes only (agent, finalized_rollout, lambda, cfg): {ok}; "
        f"`batch` is not among its arguments")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attack", required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report: dict = {"note": "SMOKE TEST ONLY -- not an A3 result", "arms": {}}
    all_ok = True

    for arm, d in (("attack", args.attack), ("clean", args.clean)):
        run = load(Path(d))
        checks = verify(run, arm)
        ok = all(c["pass"] for c in checks)
        all_ok &= ok
        lg = run["logged"]
        report["arms"][arm] = {
            "dir": run["dir"], "rounds_completed": run["n_rounds"],
            "cells": int(lg["spread"].size), "all_pass": ok, "checks": checks,
            "mad": {"mean": float(lg["spread"].mean()), "min": float(lg["spread"].min()),
                    "median": float(np.median(lg["spread"])), "max": float(lg["spread"].max()),
                    "nonzero_frac": float((lg["spread"] > 0).mean()),
                    "above_sigma_min_frac": float((lg["spread"] > SIGMA_MIN).mean())},
            "beta_mad": {"mean": float(lg["applied_margin"].mean()),
                         "sd": float(lg["applied_margin"].std()),
                         "min": float(lg["applied_margin"].min()),
                         "max": float(lg["applied_margin"].max())},
            "wall_clock_s_per_round": [round(r["source_batch"]["wall_clock_s"], 1)
                                       for r in run["rounds"]],
        }
        print(f"\n{'=' * 74}\n  {arm.upper()} ARM -- {run['dir']}  "
              f"({run['n_rounds']} rounds, {lg['spread'].size} cells)\n{'=' * 74}")
        for c in checks:
            print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}\n         {c['detail']}")

    # Direction only, and labelled as such: three rounds cannot measure this.
    a = report["arms"]["attack"]["beta_mad"]["mean"]
    e = report["arms"]["clean"]["beta_mad"]["mean"]
    report["margin_responds_direction_only"] = {
        "attack_mean_beta_mad": a, "clean_mean_beta_mad": e, "ratio": (a / e) if e else None,
        "note": "DIRECTION ONLY. Three rounds under an untrained policy; this is a "
                "plumbing observation, not evidence for A3-G3-iv.",
    }

    report["all_pass"] = bool(all_ok)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{'=' * 74}")
    print(f"  mean beta*MAD -- attack {a:.4f} vs clean {e:.4f}"
          + (f"  (ratio {a / e:.3f})" if e else "")
          + "   [DIRECTION ONLY, not evidence]")
    print(f"  SMOKE TEST {'PASSES' if all_ok else 'FAILS'} -> {args.out}")
    print("  This is a plumbing check. It is NOT an A3 scientific result.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
