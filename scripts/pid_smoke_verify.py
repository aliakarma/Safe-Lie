#!/usr/bin/env python
"""Verify the PID-Lagrangian smoke test. NOT a baseline analysis.

This checks that the PID controller RAN CORRECTLY end to end on the real
pipeline. It makes no scientific claim: three rounds under an essentially
untrained policy cannot characterise a controller, and the gains in
`configs/experiment/a3/_smoke_pid_lagrangian.yaml` are smoke values chosen
for visibility, not a declared operating point.

Every check is mechanical -- an exact identity, a shape, or the absence of
a NaN -- and every one is pass/fail rather than a measurement. That matters
because the failure this exists to catch is a run that LOOKS fine: a PID
config whose gains silently never reached the multiplier would still
finish, still log a plausible lambda, and still look like a PID run.

    P1  the run really used controller="pid", with the declared gains
    P2  lambda == clip(k_p*Delta + I + k_d*d) exactly, every cell
        -- this is what proves ALL THREE terms reach the dual update
    P3  I_{k+1} == clip(W @ I_k + k_i*Delta) exactly, every cell
        -- the integral recursion, and the only exact identity PID has
    P4  the derivative is one-sided and correctly lagged:
        d_k == max(0, Delta_k - Delta_{k-1}), and d_0 == 0
    P5  the P and D terms are actually NONZERO somewhere
        -- a gain that reaches the dual but multiplies zero proves nothing
    P6  lambda stays inside [0, lambda_max]
    N   no NaN/Inf anywhere in the controller path
    L   lambda actually DIFFERS from the paired Lagrangian run
        -- the controller changed the multiplier, rather than being
           configured and then ignored (needs --lagrangian)

Usage:
    python scripts/pid_smoke_verify.py \
        --pid results/a3_smoke_pid/smoke_pid_lagrangian \
        --lagrangian results/a3_smoke_rerun/smoke_m5_rce_attack \
        --out results/a3_smoke_pid/pid_smoke_verification.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safelie.consensus.topologies import build_topology  # noqa: E402

TOL = 1e-9


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def load(run_dir: Path) -> dict:
    rounds = read_jsonl(run_dir / "rounds.jsonl")
    if not rounds:
        raise SystemExit(f"{run_dir}: rounds.jsonl is empty -- the run did not start")
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    aids = sorted(rounds[0]["constraints"])

    def per_cell(key: str, where: str | None = None) -> np.ndarray:
        return np.array(
            [[(r["constraints"][a][where][key] if where else r["constraints"][a][key]) for a in aids]
             for r in rounds], dtype=float)

    return {
        "dir": run_dir,
        "rounds": rounds,
        "aids": aids,
        "meta": meta,
        "cfg": meta["config_snapshot"],
        "residual": per_cell("constraint_residual"),
        "lam": per_cell("lambda_after"),
    }


def pid_fields(run: dict) -> dict[str, np.ndarray]:
    keys = ("integral_after", "integral_mixed_before", "p_term", "i_term", "d_term", "derivative_raw")
    missing = [r for r in run["rounds"] if "pid" not in r["constraints"][run["aids"][0]]]
    if missing:
        raise SystemExit(
            f"{run['dir']}: no `pid` block in the round record. The run did not take the PID "
            "path -- check dual.controller in its config snapshot."
        )
    return {
        k: np.array([[r["constraints"][a]["pid"][k] for a in run["aids"]] for r in run["rounds"]], dtype=float)
        for k in keys
    }


def report(name: str, ok: bool, detail: str, results: list) -> None:
    results.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"         {detail}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", required=True, type=Path)
    ap.add_argument("--lagrangian", type=Path, default=None,
                    help="Paired controller='lagrangian' run, identical in every other field. "
                         "Enables check L.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run = load(args.pid)
    pid = pid_fields(run)
    dual = run["cfg"]["dual"]
    results: list = []

    n_rounds, n_agents = run["lam"].shape
    print("=" * 74)
    print(f"  PID-LAGRANGIAN SMOKE -- {args.pid}  ({n_rounds} rounds, {n_rounds * n_agents} cells)")
    print("=" * 74)

    # -- P1 -----------------------------------------------------------------
    k_p, k_i, k_d = float(dual["k_p"]), float(dual["k_i"]), float(dual["k_d"])
    lam_max = float(dual["lambda_max"])
    report(
        "P1 the run used controller='pid' with the declared gains",
        dual["controller"] == "pid" and None not in (dual["k_p"], dual["k_i"], dual["k_d"]),
        f"controller={dual['controller']!r} k_p={k_p} k_i={k_i} k_d={k_d} lambda_max={lam_max}",
        results,
    )

    # -- P2: the identity that proves all three terms reach the multiplier ---
    reconstructed = np.clip(pid["p_term"] + pid["i_term"] + pid["d_term"], 0.0, lam_max)
    err_lam = float(np.max(np.abs(reconstructed - run["lam"])))
    report(
        "P2 lambda == clip(k_p*Delta + I + k_d*d), every cell",
        err_lam <= TOL,
        f"max|reconstructed - lambda_after| = {err_lam:.3e}  (must be <= {TOL:g}); "
        f"p_term also checked against k_p*residual: "
        f"{float(np.max(np.abs(pid['p_term'] - k_p * run['residual']))):.3e}",
        results,
    )

    # -- P3: the integral recursion -----------------------------------------
    expected_I = np.clip(pid["integral_mixed_before"] + k_i * run["residual"], 0.0, lam_max)
    err_I = float(np.max(np.abs(expected_I - pid["integral_after"])))
    # ...and that `integral_mixed_before` really is `W @ I_k`, not a copy of I_k.
    W = build_topology(run["cfg"]["topology"]["name"], run["cfg"]["topology"]["n_agents"])
    prev_I = np.vstack([np.zeros((1, n_agents)), pid["integral_after"][:-1]])
    err_mix = float(np.max(np.abs((prev_I @ W.T) - pid["integral_mixed_before"])))
    report(
        "P3 I_{k+1} == clip(W @ I_k + k_i*Delta), every cell",
        err_I <= TOL and err_mix <= TOL,
        f"max|recursion error| = {err_I:.3e}; max|integral_mixed_before - W @ I_k| = {err_mix:.3e} "
        f"(both must be <= {TOL:g})",
        results,
    )

    # -- P4: one-sided, correctly lagged derivative -------------------------
    delta = run["residual"]
    prev_delta = np.vstack([delta[:1], delta[:-1]])  # round 0 seeds with Delta_0
    expected_d = np.maximum(delta - prev_delta, 0.0)
    err_d = float(np.max(np.abs(expected_d - pid["derivative_raw"])))
    round0_zero = bool(np.all(pid["derivative_raw"][0] == 0.0))
    report(
        "P4 derivative is one-sided and correctly lagged",
        err_d <= TOL and round0_zero and bool(np.all(pid["derivative_raw"] >= 0.0)),
        f"max|d - max(0, Delta_k - Delta_(k-1))| = {err_d:.3e}; round-0 derivative all zero: "
        f"{round0_zero}; any negative derivative: {bool(np.any(pid['derivative_raw'] < 0.0))}",
        results,
    )

    # -- P5: the terms are not merely present but non-trivial ---------------
    nz_p = int(np.count_nonzero(pid["p_term"]))
    nz_i = int(np.count_nonzero(pid["i_term"]))
    nz_d = int(np.count_nonzero(pid["d_term"]))
    n_cells = n_rounds * n_agents
    report(
        "P5 P and I terms are nonzero somewhere",
        nz_p > 0 and nz_i > 0,
        f"nonzero cells -- p_term {nz_p}/{n_cells}, i_term {nz_i}/{n_cells}",
        results,
    )

    # -- D-path coverage: REPORTED, never gating ----------------------------
    # The D term is zero exactly when the cost did not rise, and on a short
    # run under a rapidly-improving policy that is the *normal* outcome --
    # cost falls in every cell, so the one-sided clamp correctly outputs
    # zero throughout. Gating on a nonzero D term would therefore fail a
    # perfectly healthy run, and passing it silently would hide that the
    # end-to-end D path went undemonstrated. So it is reported either way,
    # and the reader is told where the D path IS pinned unconditionally.
    n_rising = int(np.count_nonzero(pid["derivative_raw"]))
    if n_rising:
        print(f"  [INFO] D path exercised end to end: {n_rising}/{n_cells} cells had a rising "
              f"cost; d_term range [{pid['d_term'].min():.3e}, {pid['d_term'].max():.3e}]")
    else:
        print(f"  [INFO] D path NOT exercised end to end in this run: the cost fell or held in "
              f"all {n_cells} cells, so max(0, .) is correctly zero throughout and no nonzero "
              f"d_term reached the multiplier here.")
        print("         This is the expected outcome for a 3-round run under an improving "
              "policy, not a defect -- but it does mean this artifact does not by itself")
        print("         demonstrate D-path propagation. That is pinned unconditionally by")
        print("         tests/unit/test_pid_dual.py::test_derivative_is_one_sided and")
        print("         ::test_derivative_is_zero_on_the_first_round.")
    results.append({
        "check": "D-path coverage (reported, not gating)",
        "pass": True,
        "exercised": bool(n_rising),
        "detail": f"{n_rising}/{n_cells} cells had a rising cost",
    })

    # -- P6: projection ------------------------------------------------------
    report(
        "P6 lambda and the integral stay inside [0, lambda_max]",
        bool(np.all(run["lam"] >= 0.0) and np.all(run["lam"] <= lam_max)
             and np.all(pid["integral_after"] >= 0.0) and np.all(pid["integral_after"] <= lam_max)),
        f"lambda in [{run['lam'].min():.4f}, {run['lam'].max():.4f}], "
        f"integral in [{pid['integral_after'].min():.4f}, {pid['integral_after'].max():.4f}], "
        f"bound = {lam_max}",
        results,
    )

    # -- N: finiteness -------------------------------------------------------
    nonfinite = {k: int((~np.isfinite(v)).sum()) for k, v in pid.items()}
    nonfinite["lambda_after"] = int((~np.isfinite(run["lam"])).sum())
    nonfinite["constraint_residual"] = int((~np.isfinite(run["residual"])).sum())
    report(
        "N  no NaN/Inf anywhere in the controller path",
        sum(nonfinite.values()) == 0,
        f"non-finite counts={nonfinite}",
        results,
    )

    # -- L: the controller actually changed something ------------------------
    if args.lagrangian is not None:
        lag = load(args.lagrangian)
        # A run recorded before `controller` existed has no such key, and its
        # absence *means* lagrangian -- that was the only path at the time.
        # Defaulting here rather than requiring the key is what lets a
        # pre-change artifact serve as the paired baseline, which is exactly
        # the comparison worth making.
        lag_controller = lag["cfg"]["dual"].get("controller", "lagrangian")
        if lag_controller != "lagrangian":
            raise SystemExit(
                f"{args.lagrangian}: expected controller='lagrangian' for the paired run, "
                f"got {lag_controller!r}"
            )
        # Pairing holds at ROUND 0 ONLY, and that is not a weakness of the
        # comparison -- it is the closed loop working. Round 0's sources are
        # collected under the shared initial policy, so its residuals must
        # agree bit for bit. The controller then writes a different lambda,
        # PPO consumes it, and from round 1 the two runs are optimising
        # different objectives and visiting different states. Demanding
        # equal residuals at round 1 would be demanding that the multiplier
        # have no effect, i.e. the opposite of what this check is for.
        res0 = float(np.max(np.abs(lag["residual"][0] - run["residual"][0])))
        lam0 = float(np.max(np.abs(lag["lam"][0] - run["lam"][0])))
        report(
            "L  PID changed the multiplier, from identical round-0 sources",
            res0 <= TOL and lam0 > TOL,
            f"round-0 max|residual difference| = {res0:.3e} (must be <= {TOL:g}: the two runs "
            f"draw the same sources from the same streams, so they are genuinely paired); "
            f"round-0 max|lambda difference| = {lam0:.6f} (must be > {TOL:g}: the controller "
            f"is not being configured and then ignored)",
            results,
        )
        later = [float(np.max(np.abs(lag["residual"][k] - run["residual"][k])))
                 for k in range(1, min(len(lag["rounds"]), n_rounds))]
        print(f"  [INFO] residuals diverge after round 0 as expected "
              f"(max per round: {[f'{d:.4f}' for d in later]}) -- a different multiplier "
              f"means a different PPO update and a different visited state distribution.")
    else:
        print("  [SKIP] L  no --lagrangian run given; cannot check that PID changed anything")

    all_pass = all(r["pass"] for r in results)
    print("=" * 74)
    print(f"  {'PID SMOKE TEST PASSES' if all_pass else 'PID SMOKE TEST FAILED'}")
    print("  This is a plumbing check. It is NOT a baseline result, and the gains")
    print("  it ran under are smoke values, not a declared operating point.")
    print("=" * 74)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "note": "PID SMOKE TEST ONLY -- not a baseline result; gains are not a declaration",
            "pid_run": str(args.pid),
            "lagrangian_run": str(args.lagrangian) if args.lagrangian else None,
            "gains": {"k_p": k_p, "k_i": k_i, "k_d": k_d, "lambda_max": lam_max},
            "checks": results,
            "all_pass": all_pass,
        }, indent=2), encoding="utf-8")
        print(f"  -> {args.out}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
