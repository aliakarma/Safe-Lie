#!/usr/bin/env python
"""Full per-run verification of one A1 training run.

Runs every structural gate in docs/a1_attack_gates.md §7 plus the
learning-health gate A1-G6, against one finished (or partial) run and its
paired clean control. Intended to be invoked once per seed, the moment
that seed's run completes, so a defect is caught after ~7 hours rather
than after ~39.

It does NOT evaluate the scientific gates A1-G1..G5 -- those are paired
contrasts over all three seeds and belong to `scripts/analyze_a1.py`.
Keeping them out is deliberate: per-seed verification must not become a
back door to peeking at the outcome one seed at a time.

Usage:
    python scripts/a1_verify_run.py --run results/runs_a1/B_seed0
    python scripts/a1_verify_run.py --run results/runs_a1/B_seed0 --allow-partial
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-9
D_BUDGET = 25.0
B_MAGNITUDE = 12.5
M_SOURCES = 3
EXPECTED_ROUNDS = 250
EXPECTED_ENV_SEEDS = 23_100
KL_MEDIAN_MAX = 0.0069        # A1-G6 == G10-C-iii verbatim
KL_P95_MAX = 0.01548
LAMBDA_SAT_MAX = 0.05

ATTACKED = {0: "batch_1", 1: "batch_2", 2: "batch_3"}      # §3
CLEAN_DIR = {
    0: ROOT / "results/runs_constraint_batch_g9/g9_batch_clean",
    1: ROOT / "results/runs_constraint_batch_g10/seed1",
    2: ROOT / "results/runs_constraint_batch_g10/seed2",
}


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def parse_run_name(run: Path) -> tuple[str, int]:
    """'B_seed1' -> ('B', 1)."""
    stem = run.name
    cond, _, tail = stem.partition("_seed")
    return cond, int(tail)


def check(name: str, ok: bool, detail: dict | None = None) -> dict:
    out = {"gate": name, "pass": bool(ok)}
    if detail:
        out.update(detail)
    return out


def verify(run: Path, allow_partial: bool) -> dict:
    cond, seed = parse_run_name(run)
    clean = CLEAN_DIR[seed]
    attacked_sid = ATTACKED[seed]
    expect_shift = cond == "B"

    rounds = read_jsonl(run / "rounds.jsonl")
    K = len(rounds)
    aids = sorted(rounds[0]["constraints"].keys())
    results: list[dict] = []

    # ---- A1-S2/S3: injection point and magnitude, on every cell ----------
    # The §7.1 identity is a CONSTANT only for conditions A (0) and B
    # (+B/M). Condition D adds zero-mean i.i.d. Gaussian noise, so its
    # difference is a fresh draw every (round, owner); asserting a constant
    # there would fail a run behaving exactly as §2 declares. D is checked
    # against its declared DISTRIBUTION instead.
    diffs, src_errs = [], []
    for rec in rounds:
        for aid in aids:
            c = rec["constraints"][aid]
            vals = {r["source_id"]: float(r["value"]) for r in c["reports"]}
            post = float(c["mechanism_reported_cost_return"])
            diffs.append(float(np.mean(list(vals.values()))) - post)
            if expect_shift:
                others = sum(v for s, v in vals.items() if s != attacked_sid)
                src_errs.append(abs((M_SOURCES * post - others - vals[attacked_sid])
                                    + B_MAGNITUDE))
    diffs = np.asarray(diffs)

    if cond in ("A", "B"):
        expected = (B_MAGNITUDE / M_SOURCES) if expect_shift else 0.0
        errs = np.abs(diffs - expected)
        results.append(check(
            "A1-S2/S3 mechanism identity", float(errs.max()) <= TOL,
            {"kind": "constant identity", "n_cells": int(errs.size),
             "max_abs_error": float(errs.max()), "expected_pre_minus_post": expected}))
    else:
        # implied per-source perturbation, from mean aggregation over M
        eps = -M_SOURCES * diffs
        n = eps.size
        sigma = 0.5 * D_BUDGET                      # §2: sigma = budget_ratio * d
        se = sigma / np.sqrt(n)
        mean_ok = abs(float(eps.mean())) <= 4 * se  # ~1-in-16,000 false alarm
        sd_ok = 0.85 <= float(eps.std(ddof=1)) / sigma <= 1.15
        live = bool(np.abs(diffs).min() > TOL)
        results.append(check(
            "A1-S2/S3 perturbation matches its declared distribution",
            mean_ok and sd_ok and live,
            {"kind": "zero-mean perturbation", "n_cells": int(n),
             "implied_epsilon_mean": float(eps.mean()),
             "mean_tolerance": float(4 * se),
             "implied_epsilon_sd": float(eps.std(ddof=1)),
             "declared_sigma": sigma,
             "sd_ratio": float(eps.std(ddof=1)) / sigma,
             "live_on_every_cell": live}))
    if expect_shift:
        results.append(check(
            "A1-S3 named source shifted by -B", max(src_errs) <= TOL,
            {"source": attacked_sid, "n_cells": len(src_errs),
             "max_abs_error": float(max(src_errs))}))

    # ---- §3 mapping ------------------------------------------------------
    logged = sorted(rounds[0]["constraints"][aids[0]]["corrupted_source_ids"])
    results.append(check("§3 attacked-source mapping", logged == [attacked_sid],
                         {"logged": logged, "expected": [attacked_sid]}))

    # ---- A1-S6: policy pinning ------------------------------------------
    all_match = all(r["source_batch"]["worker_checksums_all_match"] for r in rounds)
    n_distinct = len({r["source_batch"]["policy_checksum"] for r in rounds})
    chunks_ok = all(r["source_batch"]["n_chunks"] == 48 for r in rounds)
    traj_ok = all(r["source_batch"]["n_trajectories"] == 90 for r in rounds)
    results.append(check("A1-S6 policy pinning", all_match and chunks_ok and traj_ok,
                         {"worker_checksums_all_match": all_match,
                          "n_chunks_all_48": chunks_ok,
                          "n_trajectories_all_90": traj_ok}))
    results.append(check("A1-S6 distinct theta per round", n_distinct == K,
                         {"distinct_checksums": n_distinct, "rounds": K}))

    # ---- A1-S9: common random numbers vs the clean pair ------------------
    a_seeds = read_jsonl(run / "source_seeds.jsonl")
    c_seeds = read_jsonl(clean / "source_seeds.jsonl")
    n_cmp = min(len(a_seeds), len(c_seeds))
    mismatched, n_pairs = [], 0
    for k in range(n_cmp):
        for sid, pairs in a_seeds[k]["seeds"].items():
            cp = c_seeds[k]["seeds"][sid]
            n_pairs += len(pairs)
            if [list(x) for x in pairs] != [list(x) for x in cp]:
                mismatched.append({"round": k, "source": sid})
        if [list(x) for x in a_seeds[k]["reference_seeds"]] != \
           [list(x) for x in c_seeds[k]["reference_seeds"]]:
            mismatched.append({"round": k, "source": "reference"})
    results.append(check("A1-S9 CRN: source seeds identical to clean pair",
                         not mismatched,
                         {"rounds_compared": n_cmp, "seed_pairs_compared": n_pairs,
                          "clean_pair": str(clean.relative_to(ROOT)),
                          "mismatches": mismatched[:5]}))

    # ---- A1-S5: within-run source independence ---------------------------
    env_seen, tor_seen, dup = set(), set(), 0
    replica_sets: dict[str, set] = {}
    ref_set: set = set()
    for rec in a_seeds:
        for sid, pairs in rec["seeds"].items():
            s = replica_sets.setdefault(sid, set())
            for e, t in pairs:
                if e in env_seen:
                    dup += 1
                if t in tor_seen:
                    dup += 1
                env_seen.add(e)
                tor_seen.add(t)
                s.add(e)
        for e, _t in rec["reference_seeds"]:
            ref_set.add(e)
    sids = sorted(replica_sets)
    pairwise = {f"{a}|{b}": len(replica_sets[a] & replica_sets[b])
                for i, a in enumerate(sids) for b in sids[i + 1:]}
    ref_overlap = {s: len(replica_sets[s] & ref_set) for s in sids}
    results.append(check(
        "A1-S5 source streams disjoint within run",
        dup == 0 and not any(pairwise.values()) and not any(ref_overlap.values()),
        {"duplicate_events": dup, "unique_env_seeds": len(env_seen),
         "replica_pairwise_overlap": pairwise, "replica_vs_reference_overlap": ref_overlap}))

    # ---- A1-S8: reference isolation --------------------------------------
    vref = run / "validation_reference.jsonl"
    if vref.exists() and (recs := read_jsonl(vref)):
        rerr = []
        for r in recs:
            k = r["round_k"]
            if k >= K:
                continue
            for sid, per_owner in r["source_means"].items():
                for aid, v in per_owner.items():
                    lg = next(x["value"] for x in rounds[k]["constraints"][aid]["reports"]
                              if x["source_id"] == sid)
                    rerr.append(abs(v - lg))
        results.append(check("A1-S8 reference never saw the corruption",
                             bool(rerr) and max(rerr) <= TOL,
                             {"rounds": [r["round_k"] for r in recs],
                              "n_cells": len(rerr),
                              "max_abs_error": float(max(rerr)) if rerr else None}))
    else:
        results.append({"gate": "A1-S8 reference never saw the corruption",
                        "pass": None, "note": "no validation round reached yet"})

    # ---- A1-S4 + audit, from run_metadata (only exists once complete) -----
    # `run_metadata.json` is written at launch but its `env_steps`,
    # `source_seed_audit` and `timing` blocks are only populated at
    # completion, so mid-run they are absent rather than wrong. Report them
    # as n/a while partial instead of failing a run that is simply not
    # finished yet -- the equivalent live evidence is already covered by the
    # A1-S5 stream check above, which reads source_seeds.jsonl directly.
    md = run / "run_metadata.json"
    meta = json.loads(md.read_text(encoding="utf-8")) if md.exists() else {}
    steps, aud = meta.get("env_steps"), meta.get("source_seed_audit")

    # A1-S4 is DERIVED from rounds.jsonl and the config, never taken from
    # `env_steps`. A no-op resume rewrites that field with the resuming
    # invocation's counts (observed on D_seed0: ppo=2,000 instead of
    # 500,000) while rounds.jsonl, being append-only, stays correct.
    rollout = int(meta.get("config_snapshot", {}).get("rollout_length", 0)) if meta else 0
    derived_ppo = K * rollout
    derived_src = sum(int((r.get("source_batch") or {}).get("env_steps", 0)) for r in rounds)
    if rollout:
        expected_ppo = EXPECTED_ROUNDS * rollout
        ok_steps = (derived_ppo == expected_ppo) if not allow_partial else True
        results.append({"gate": "A1-S4 PPO steps independent of source",
                        "pass": ok_steps if not allow_partial else None,
                        "derived_ppo_steps": derived_ppo,
                        "derived_source_steps": derived_src,
                        "expected_ppo_steps": expected_ppo,
                        "metadata_env_steps": steps,
                        "note": "derived from rounds.jsonl; metadata env_steps is "
                                "reported only for comparison and is unreliable "
                                "after a resume"})
    else:
        results.append({"gate": "A1-S4 PPO steps independent of source",
                        "pass": None, "note": "no config snapshot available"})
    if aud:
        results.append(check("A1-S5 seed audit (metadata)",
                             aud.get("duplicate_seed_events") == 0
                             and aud.get("n_env_seeds_issued") == EXPECTED_ENV_SEEDS,
                             {"n_env_seeds_issued": aud.get("n_env_seeds_issued"),
                              "duplicate_seed_events": aud.get("duplicate_seed_events")}))
    else:
        results.append({"gate": "A1-S5 seed audit (metadata)",
                        "pass": None if allow_partial else False,
                        "note": "source_seed_audit is written at completion"})
    if t := meta.get("timing"):
        results.append({"gate": "timing (reported)", "pass": None,
                        "wall_clock_h": round(t.get("wall_clock_s", 0) / 3600, 2),
                        "s_per_round": round(t.get("s_per_round", 0), 1),
                        "source_s_per_round": round(t.get("source_s_per_round", 0), 1)})

    # ---- A1-G6: learning health -----------------------------------------
    lam = np.array([[r["constraints"][a]["lambda_after"] for a in aids] for r in rounds])
    mech = np.array([[r["constraints"][a]["mechanism_reported_cost_return"] for a in aids]
                     for r in rounds])
    res = np.array([[r["constraints"][a]["constraint_residual"] for a in aids]
                    for r in rounds])
    finite = bool(np.isfinite(lam).all() and np.isfinite(mech).all()
                  and np.isfinite(res).all())
    sat = float((lam >= 25.0 - 1e-9).mean())
    health = {"finite": finite, "lambda_frac_saturated": sat,
              "lambda_max": float(lam.max()), "lambda_frac_positive": float((lam > 0).mean())}
    ppo0 = rounds[0]["constraints"][aids[0]].get("ppo") or {}
    kl_key = next((k for k in ("approx_kl", "kl") if k in ppo0), None)
    ok = finite and sat < LAMBDA_SAT_MAX
    if kl_key:
        kl = np.array([[r["constraints"][a]["ppo"][kl_key] for a in aids] for r in rounds])
        health["kl_median"] = float(np.median(kl))
        health["kl_p95"] = float(np.percentile(kl, 95))
        ok = ok and health["kl_median"] <= KL_MEDIAN_MAX and health["kl_p95"] <= KL_P95_MAX
    if "entropy" in ppo0 and K >= 20:
        ent = np.array([[r["constraints"][a]["ppo"]["entropy"] for a in aids] for r in rounds])
        health["entropy_first10"] = float(ent[:10].mean())
        health["entropy_last10"] = float(ent[-10:].mean())
    results.append(check("A1-G6 learning health", ok, health))

    # ---- completeness ----------------------------------------------------
    results.append({"gate": "rounds complete", "pass": (K == EXPECTED_ROUNDS)
                    if not allow_partial else None,
                    "rounds": K, "expected": EXPECTED_ROUNDS})

    gating = [r["pass"] for r in results if r["pass"] is not None]
    return {"run": str(run.relative_to(ROOT)), "condition": cond, "seed": seed,
            "attacked_source": attacked_sid, "rounds": K, "partial": allow_partial,
            "checks": results, "pass": bool(all(gating))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    if not run.is_absolute():
        run = ROOT / run
    rep = verify(run, args.allow_partial)

    print(f"\nA1 per-run verification: {rep['condition']}_seed{rep['seed']}  "
          f"({rep['rounds']} rounds, attacked={rep['attacked_source']})")
    for c in rep["checks"]:
        v = c["pass"]
        tag = "PASS" if v is True else ("n/a " if v is None else "FAIL")
        extra = {k: val for k, val in c.items() if k not in ("gate", "pass")}
        print(f"  [{tag}] {c['gate']}")
        if v is False or (v is None and extra):
            for k, val in extra.items():
                print(f"          {k}: {val}")
    print(f"\n  {'RUN VERIFIED' if rep['pass'] else 'RUN FAILED VERIFICATION'}")

    out = Path(args.out) if args.out else run / "a1_run_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"  -> {out}")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
