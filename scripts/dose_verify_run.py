#!/usr/bin/env python
"""Full per-run verification of one DR dose run.

Runs every structural gate in docs/dose_response_gates.md section 7 plus
the learning-health gate DR-G6, against one finished (or partial) dose run,
its paired clean control and its paired A1 condition-B run. Intended to be
invoked the moment a run completes, so a defect is caught after ~8 hours
rather than after ~46.

It does NOT evaluate the scientific gates DR-G1..G5 -- those are paired
contrasts over all three seeds and belong to `scripts/analyze_dose.py`.
Keeping them out is deliberate, and inherited from A1: per-seed
verification must not become a back door to peeking at the outcome one
seed at a time.

**`B` is never a constant in this file.** It is derived from the run's own
`config_snapshot` (`attack.budget_ratio * env.budget`) and then checked
against the dose this campaign declares for that `run_id` (DR-S12). A run
whose config says one thing and whose logs say another fails here rather
than being silently analysed at the wrong magnitude.

Usage:
    python scripts/dose_verify_run.py --run results/runs_dose/B100_seed0
    python scripts/dose_verify_run.py --run results/runs_dose/B025_seed0 --allow-partial
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-9
D_BUDGET = 25.0
M_SOURCES = 3
EXPECTED_ROUNDS = 250
EXPECTED_ENV_SEEDS = 23_100
KL_MEDIAN_MAX = 0.0069        # DR-G6 == A1-G6 == G10-C-iii verbatim
KL_P95_MAX = 0.01548
LAMBDA_SAT_MAX = 0.05

# docs/dose_response_gates.md section 3. Do not tune here.
DOSE = {"025": 0.25, "100": 1.0}
A1_DOSE = 0.5

# A1 section 3's balanced mapping, inherited unchanged (gates doc 3.2).
ATTACKED = {0: "batch_1", 1: "batch_2", 2: "batch_3"}
CLEAN_DIR = {
    0: ROOT / "results/runs_constraint_batch_g9/g9_batch_clean",
    1: ROOT / "results/runs_constraint_batch_g10/seed1",
    2: ROOT / "results/runs_constraint_batch_g10/seed2",
}
A1_B_DIR = {k: ROOT / f"results/runs_a1/B_seed{k}" for k in (0, 1, 2)}


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def parse_run_name(run: Path) -> tuple[str, int]:
    """'B100_seed1' -> ('100', 1)."""
    cond, _, tail = run.name.partition("_seed")
    return cond[1:], int(tail)


def check(name: str, ok: bool, detail: dict | None = None) -> dict:
    out = {"gate": name, "pass": bool(ok)}
    if detail:
        out.update(detail)
    return out


def round0_cells(run_dir: Path) -> tuple[dict[str, dict[str, float]], dict[str, float]] | None:
    """Round-0 per-owner source reports and mechanism aggregates.

    Returns `({owner: {source_id: value}}, {owner: mechanism_reported})`.

    **`reports[].value` is logged PRE-attack**, in every condition. Under
    the CRN design it is therefore bit-IDENTICAL across all four doses and
    the clean run at round 0, corrupted source included -- that equality is
    the CRN integrity check, not the attack check. The injected shift is
    visible only in `mechanism_reported_cost_return`, which is written
    post-hook. Both quantities are returned so section 7.2 can check the
    right one against the right expectation.
    """
    f = run_dir / "rounds.jsonl"
    if not f.exists():
        return None
    with f.open(encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                rec = json.loads(ln)
                vals = {aid: {r["source_id"]: float(r["value"]) for r in c["reports"]}
                        for aid, c in rec["constraints"].items()}
                mech = {aid: float(c["mechanism_reported_cost_return"])
                        for aid, c in rec["constraints"].items()}
                return vals, mech
    return None


def verify(run: Path, allow_partial: bool) -> dict:
    tag, seed = parse_run_name(run)
    if tag not in DOSE:
        raise SystemExit(f"{run.name}: unknown dose tag {tag!r}; expected one of {sorted(DOSE)}")
    clean = CLEAN_DIR[seed]
    a1_b = A1_B_DIR[seed]
    attacked_sid = ATTACKED[seed]

    rounds = read_jsonl(run / "rounds.jsonl")
    K = len(rounds)
    aids = sorted(rounds[0]["constraints"].keys())
    results: list[dict] = []

    md = run / "run_metadata.json"
    meta = json.loads(md.read_text(encoding="utf-8")) if md.exists() else {}
    snap = meta.get("config_snapshot", {})

    # ---- DR-S12: the dose is what this campaign declares -----------------
    # B comes from the run's own snapshot, never from a constant here.
    declared_ratio = DOSE[tag]
    snap_ratio = (snap.get("attack") or {}).get("budget_ratio")
    snap_d = (snap.get("env") or {}).get("budget")
    B = float(snap_ratio) * float(snap_d) if snap_ratio is not None and snap_d else None
    atk = snap.get("attack") or {}
    operator_ok = (
        atk.get("name") == "primary"
        and atk.get("f") == 1
        and atk.get("direction") == "negative"
        and atk.get("support") == "persistent"
        and atk.get("adaptivity") == "static"
        and atk.get("consistency") == "consistent"
    )
    results.append(check(
        "DR-S12 declared dose and unchanged operator",
        snap_ratio == declared_ratio and snap_d == D_BUDGET and operator_ok,
        {"config_budget_ratio": snap_ratio, "expected_budget_ratio": declared_ratio,
         "config_d": snap_d, "derived_B": B,
         "derived_B_over_M": (B / M_SOURCES) if B is not None else None,
         "operator": {k: atk.get(k) for k in
                      ("name", "f", "direction", "support", "adaptivity", "consistency")},
         "operator_unchanged_from_a1": operator_ok}))
    if B is None:
        raise SystemExit(f"{run.name}: cannot derive B from run_metadata config_snapshot")

    # ---- DR-S11: no accidental RCE --------------------------------------
    dfn = snap.get("defense") or {}
    calib_files = list(run.glob("*calibration*.json"))
    results.append(check(
        "DR-S11 no RCE on this line",
        dfn.get("name") == "mean" and dfn.get("f") == 0 and not calib_files,
        {"defense_name": dfn.get("name"), "defense_f": dfn.get("f"),
         "calibration_artifacts": [p.name for p in calib_files]}))

    # ---- DR-S2/S3: injection point and magnitude, on every cell ----------
    # Both dose arms run `primary`, which is deterministic, so the identity
    # is an exact CONSTANT +B/M on every (round, owner) cell -- unlike A1's
    # condition D, whose zero-mean draws required a distributional check.
    diffs, src_errs = [], []
    for rec in rounds:
        for aid in aids:
            c = rec["constraints"][aid]
            vals = {r["source_id"]: float(r["value"]) for r in c["reports"]}
            post = float(c["mechanism_reported_cost_return"])
            diffs.append(float(np.mean(list(vals.values()))) - post)
            others = sum(v for s, v in vals.items() if s != attacked_sid)
            src_errs.append(abs((M_SOURCES * post - others - vals[attacked_sid]) + B))
    diffs = np.asarray(diffs)
    expected = B / M_SOURCES
    errs = np.abs(diffs - expected)
    results.append(check(
        "DR-S2/S3 mechanism identity on 100% of cells", float(errs.max()) <= TOL,
        {"n_cells": int(errs.size), "max_abs_error": float(errs.max()),
         "expected_pre_minus_post": expected, "tolerance": TOL}))
    results.append(check(
        "DR-S3 named source shifted by -B", max(src_errs) <= TOL,
        {"source": attacked_sid, "B": B, "n_cells": len(src_errs),
         "max_abs_error": float(max(src_errs))}))

    # ---- gates doc 3.2 mapping ------------------------------------------
    logged = sorted(rounds[0]["constraints"][aids[0]]["corrupted_source_ids"])
    results.append(check("balanced attacked-source mapping", logged == [attacked_sid],
                         {"logged": logged, "expected": [attacked_sid]}))

    # ---- DR-S9 (7.2): cross-dose round-0 identity ------------------------
    # At round 0 the CRN design makes the underlying draws bit-identical
    # across all four doses, so these are exact, not statistical.
    #
    # TWO distinct quantities, with DIFFERENT expectations -- conflating
    # them is the mistake this comment exists to prevent:
    #
    #   reports[].value   logged PRE-attack  -> delta EXACTLY 0 vs every
    #                     reference, corrupted source included. The CRN
    #                     integrity check.
    #   mechanism_reported  written POST-hook -> carries the whole shift,
    #                     -B/M against clean and -(B - B_a1)/M against A1's
    #                     committed B/d = 0.5 run.
    #
    # This mirrors `scripts/a1_mechanism_validation.cross_run_round0`,
    # which is the committed implementation A1 gated on.
    here = round0_cells(run)
    r0: dict = {}
    for label, ref_dir, expect_mech_delta in (
        ("vs_clean", clean, -B / M_SOURCES),
        ("vs_a1_B", a1_b, -(B - A1_DOSE * D_BUDGET) / M_SOURCES),
    ):
        ref = round0_cells(ref_dir)
        if here is None or ref is None:
            r0[label] = {"pass": None, "note": f"round 0 unavailable in {ref_dir.name}"}
            continue
        (h_vals, h_mech), (r_vals, r_mech) = here, ref
        owners = sorted(set(h_vals) & set(r_vals))
        src_deltas = [abs(h_vals[a][s] - r_vals[a][s]) for a in owners for s in h_vals[a]]
        mech_errs = [abs((h_mech[a] - r_mech[a]) - expect_mech_delta) for a in owners]
        crn_ok = bool(src_deltas) and max(src_deltas) <= TOL
        mech_ok = bool(mech_errs) and max(mech_errs) <= TOL
        r0[label] = {
            "pass": bool(crn_ok and mech_ok),
            "reference": str(ref_dir.relative_to(ROOT)),
            "n_owners": len(owners),
            "pre_attack_source_values_identical": crn_ok,
            "pre_attack_max_abs_delta": max(src_deltas) if src_deltas else None,
            "expected_mechanism_delta": expect_mech_delta,
            "observed_mechanism_delta_mean": (
                float(np.mean([h_mech[a] - r_mech[a] for a in owners])) if owners else None),
            "mechanism_max_abs_error": max(mech_errs) if mech_errs else None,
        }
    results.append({"gate": "DR-S9 (7.2) cross-dose round-0 identity",
                    "pass": (None if any(v["pass"] is None for v in r0.values())
                             else all(v["pass"] for v in r0.values())),
                    **r0})

    # ---- DR-S6: policy pinning ------------------------------------------
    all_match = all(r["source_batch"]["worker_checksums_all_match"] for r in rounds)
    n_distinct = len({r["source_batch"]["policy_checksum"] for r in rounds})
    chunks_ok = all(r["source_batch"]["n_chunks"] == 48 for r in rounds)
    traj_ok = all(r["source_batch"]["n_trajectories"] == 90 for r in rounds)
    results.append(check("DR-S6 policy pinning", all_match and chunks_ok and traj_ok,
                         {"worker_checksums_all_match": all_match,
                          "n_chunks_all_48": chunks_ok,
                          "n_trajectories_all_90": traj_ok}))
    results.append(check("DR-S6 distinct theta per round", n_distinct == K,
                         {"distinct_checksums": n_distinct, "rounds": K}))

    # ---- DR-S9: CRN vs BOTH the clean pair and the A1 B pair -------------
    a_seeds = read_jsonl(run / "source_seeds.jsonl")
    crn = {}
    for label, ref_dir in (("clean", clean), ("a1_B", a1_b)):
        f = ref_dir / "source_seeds.jsonl"
        if not f.exists():
            crn[label] = {"pass": None, "note": f"{f} missing"}
            continue
        c_seeds = read_jsonl(f)
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
        crn[label] = {"pass": not mismatched, "rounds_compared": n_cmp,
                      "seed_pairs_compared": n_pairs,
                      "reference": str(ref_dir.relative_to(ROOT)),
                      "mismatches": mismatched[:5]}
    results.append({"gate": "DR-S9 CRN: source seeds identical to clean and A1 pairs",
                    "pass": (None if any(v["pass"] is None for v in crn.values())
                             else all(v["pass"] for v in crn.values())),
                    **crn})

    # ---- DR-S5: within-run source independence ---------------------------
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
        "DR-S5 source streams disjoint within run",
        dup == 0 and not any(pairwise.values()) and not any(ref_overlap.values()),
        {"duplicate_events": dup, "unique_env_seeds": len(env_seen),
         "replica_pairwise_overlap": pairwise,
         "replica_vs_reference_overlap": ref_overlap}))

    # ---- DR-S8: the withheld reference never saw the corruption ----------
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
        results.append(check("DR-S8 reference never saw the corruption",
                             bool(rerr) and max(rerr) <= TOL,
                             {"rounds": [r["round_k"] for r in recs],
                              "n_cells": len(rerr),
                              "max_abs_error": float(max(rerr)) if rerr else None}))
    else:
        results.append({"gate": "DR-S8 reference never saw the corruption",
                        "pass": None, "note": "no validation round reached yet"})

    # ---- DR-S4 + seed audit ----------------------------------------------
    # DERIVED from rounds.jsonl and the config, never from `env_steps`: a
    # no-op resume rewrites that field with the resuming invocation's counts
    # (observed on A1's D_seed0). See gates doc section 12.
    steps, aud = meta.get("env_steps"), meta.get("source_seed_audit")
    rollout = int(snap.get("rollout_length", 0))
    derived_ppo = K * rollout
    derived_src = sum(int((r.get("source_batch") or {}).get("env_steps", 0)) for r in rounds)
    if rollout:
        expected_ppo = EXPECTED_ROUNDS * rollout
        results.append({"gate": "DR-S4 PPO steps independent of source",
                        "pass": (derived_ppo == expected_ppo) if not allow_partial else None,
                        "derived_ppo_steps": derived_ppo,
                        "derived_source_steps": derived_src,
                        "expected_ppo_steps": expected_ppo,
                        "expected_source_steps": 45_000_000,
                        "metadata_env_steps": steps,
                        "note": "derived from rounds.jsonl; metadata env_steps is "
                                "reported only for comparison and is unreliable "
                                "after a resume (gates doc 12)"})
    else:
        results.append({"gate": "DR-S4 PPO steps independent of source",
                        "pass": None, "note": "no config snapshot available"})
    if aud:
        results.append(check("DR-S5 seed audit (metadata)",
                             aud.get("duplicate_seed_events") == 0
                             and aud.get("n_env_seeds_issued") == EXPECTED_ENV_SEEDS,
                             {"n_env_seeds_issued": aud.get("n_env_seeds_issued"),
                              "duplicate_seed_events": aud.get("duplicate_seed_events")}))
    else:
        results.append({"gate": "DR-S5 seed audit (metadata)",
                        "pass": None if allow_partial else False,
                        "note": "source_seed_audit is written at completion"})
    if t := meta.get("timing"):
        results.append({"gate": "timing (reported)", "pass": None,
                        "wall_clock_h": round(t.get("wall_clock_s", 0) / 3600, 2),
                        "s_per_round": round(t.get("s_per_round", 0), 1),
                        "source_s_per_round": round(t.get("source_s_per_round", 0), 1)})
    if pv := meta.get("provenance"):
        results.append({"gate": "provenance (reported)", "pass": None,
                        "cpu_model": pv.get("cpu_model"),
                        "architecture": pv.get("architecture"),
                        "os": pv.get("os"),
                        "cpu_count_logical": pv.get("cpu_count_logical"),
                        "workers": pv.get("workers"),
                        "libraries": pv.get("libraries"),
                        "config_sha256": pv.get("config_sha256")})

    # ---- DR-G6: learning health -----------------------------------------
    lam = np.array([[r["constraints"][a]["lambda_after"] for a in aids] for r in rounds])
    mech = np.array([[r["constraints"][a]["mechanism_reported_cost_return"] for a in aids]
                     for r in rounds])
    res = np.array([[r["constraints"][a]["constraint_residual"] for a in aids]
                    for r in rounds])
    finite = bool(np.isfinite(lam).all() and np.isfinite(mech).all()
                  and np.isfinite(res).all())
    sat = float((lam >= 25.0 - 1e-9).mean())
    health = {"finite": finite, "lambda_frac_saturated": sat,
              "lambda_max": float(lam.max()),
              "lambda_mean": float(lam.mean()),
              "lambda_frac_positive": float((lam > 0).mean())}
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
    results.append(check("DR-G6 learning health", ok, health))

    # ---- completeness ----------------------------------------------------
    results.append({"gate": "rounds complete",
                    "pass": (K == EXPECTED_ROUNDS) if not allow_partial else None,
                    "rounds": K, "expected": EXPECTED_ROUNDS})

    gating = [r["pass"] for r in results if r["pass"] is not None]
    return {"run": str(run.relative_to(ROOT)), "dose_tag": tag,
            "budget_ratio": declared_ratio, "B": B, "B_over_M": B / M_SOURCES,
            "seed": seed, "attacked_source": attacked_sid, "rounds": K,
            "partial": allow_partial,
            "predeclaration": "docs/dose_response_gates.md",
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
    report = verify(run, args.allow_partial)

    print(f"=== {report['run']}  B/d={report['budget_ratio']}  B={report['B']}  "
          f"B/M={report['B_over_M']:.5f}  rounds={report['rounds']} ===")
    for c in report["checks"]:
        mark = {True: "PASS", False: "FAIL", None: " n/a"}[c["pass"]]
        print(f"  [{mark}] {c['gate']}")
        for k, v in c.items():
            if k not in ("gate", "pass"):
                print(f"          {k}: {v}")
    print(f"\n{'ALL GATES PASS' if report['pass'] else 'FAILURES PRESENT'}")

    out = Path(args.out) if args.out else run / "dose_run_verification.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
