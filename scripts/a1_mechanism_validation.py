#!/usr/bin/env python
"""A1 §7 source-level attack validation — the stop conditions.

Two independent checks, both exact rather than statistical.

**In-run identity (A1-S2/S3, gate A1-G4-i).** `rounds.jsonl` logs the
source values PRE-attack (`constraints.<agent>.reports[].value`) and the
aggregate POST-attack (`mechanism_reported_cost_return`). So on every
round and every owner of an attacked run:

    mean_m(reports[m].value) - mechanism_reported_cost_return == +B/M

This localises the injection point to the one hook between the source
registry and the aggregator, on 100% of cells, without needing a second
run.

**Cross-run identity (§7.1).** With common random numbers (§5.1) the
clean and attacked runs draw identical source seeds, so at round 0 -- the
last round before the policies can diverge -- the underlying source
values are bit-identical and

    J_hat_m^attack  - J_hat_m^clean  == -B      (the attacked source)
    J_hat_m'^attack - J_hat_m'^clean == 0       (the other two)
    mean^attack     - mean^clean     == -B/M

**Reference isolation (A1-S8, §7.2).** `validation_reference.jsonl` is
written by the collector before the attack hook, so an attacked run's
reference must carry no -B offset.

Usage:
    python scripts/a1_mechanism_validation.py \
        --clean results/a1_mechanism_check/mech_clean \
        --attack results/a1_mechanism_check/mech_attack \
        --noise results/a1_mechanism_check/mech_noise \
        --B 12.5 --M 3 --attacked-source batch_1 \
        --out results/a1_mechanism_check/a1_mechanism_report.json

Exit 0 iff every gating check passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TOL = 1e-9


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def agent_ids(rounds: list[dict]) -> list[str]:
    return sorted(rounds[0]["constraints"].keys())


def in_run_identity(run: Path, B: float, M: int, expect_shift: bool) -> dict:
    """mean(pre-attack reports) - post-attack aggregate == B/M on every cell."""
    rounds = read_jsonl(run / "rounds.jsonl")
    aids = agent_ids(rounds)
    expected = (B / M) if expect_shift else 0.0
    errs, cells = [], 0
    for rec in rounds:
        for aid in aids:
            c = rec["constraints"][aid]
            pre = float(np.mean([r["value"] for r in c["reports"]]))
            post = float(c["mechanism_reported_cost_return"])
            errs.append(abs((pre - post) - expected))
            cells += 1
    errs = np.asarray(errs)
    return {
        "n_cells": cells,
        "expected_pre_minus_post": expected,
        "max_abs_error": float(errs.max()),
        "frac_cells_within_tol": float((errs <= TOL).mean()),
        "pass": bool(cells > 0 and errs.max() <= TOL),
    }


def cross_run_round0(clean: Path, attacked: Path, B: float, M: int,
                     attacked_source: str, expect_constant_shift: bool) -> dict:
    """Per-source and aggregate shift at round 0, where CRN makes the
    underlying draws bit-identical."""
    rc = read_jsonl(clean / "rounds.jsonl")[0]
    ra = read_jsonl(attacked / "rounds.jsonl")[0]
    aids = agent_ids([rc])

    per_source: dict[str, list[float]] = {}
    agg_shift: list[float] = []
    for aid in aids:
        cv = {r["source_id"]: float(r["value"]) for r in rc["constraints"][aid]["reports"]}
        av = {r["source_id"]: float(r["value"]) for r in ra["constraints"][aid]["reports"]}
        assert set(cv) == set(av), "source sets differ between conditions"
        for sid in cv:
            per_source.setdefault(sid, []).append(av[sid] - cv[sid])
        agg_shift.append(
            float(ra["constraints"][aid]["mechanism_reported_cost_return"])
            - float(rc["constraints"][aid]["mechanism_reported_cost_return"])
        )

    out: dict = {"per_source_logged_value_shift": {}, "underlying_draws_identical": True}
    for sid, shifts in sorted(per_source.items()):
        arr = np.asarray(shifts)
        out["per_source_logged_value_shift"][sid] = {
            "mean": float(arr.mean()), "max_abs": float(np.abs(arr).max()),
        }
        # `reports[].value` is PRE-attack in both runs, so with CRN it must be
        # identical -- this is the CRN integrity check, not the attack check.
        if np.abs(arr).max() > TOL:
            out["underlying_draws_identical"] = False

    agg = np.asarray(agg_shift)
    expected_agg = -(B / M) if expect_constant_shift else None
    out["aggregate_shift"] = {
        "per_owner": [round(float(x), 9) for x in agg],
        "mean": float(agg.mean()),
        "expected": expected_agg,
    }
    if expect_constant_shift:
        out["aggregate_shift"]["max_abs_error"] = float(np.abs(agg - expected_agg).max())
        out["aggregate_shift"]["pass"] = bool(np.abs(agg - expected_agg).max() <= TOL)
    else:
        # Zero-mean generic perturbation: no constant shift is expected, and
        # asserting one would be wrong. Report the realised draw instead.
        out["aggregate_shift"]["pass"] = None
        out["aggregate_shift"]["note"] = (
            "condition D is zero-mean i.i.d.; the round-0 shift is one "
            "realisation, not a predicted constant"
        )
    out["attacked_source"] = attacked_source
    out["pass"] = bool(
        out["underlying_draws_identical"]
        and (out["aggregate_shift"]["pass"] in (True, None))
    )
    return out


def reference_unshifted(clean: Path, attacked: Path) -> dict:
    """A1-S8: the withheld R_ref reference is collected before the hook and
    must carry no corruption."""
    fc, fa = clean / "validation_reference.jsonl", attacked / "validation_reference.jsonl"
    if not (fc.exists() and fa.exists()):
        return {"pass": None, "note": "no validation round in range for this check"}
    rc, ra = read_jsonl(fc), read_jsonl(fa)
    by_round_c = {r["round_k"]: r for r in rc}
    diffs, rounds_compared = [], []
    for r in ra:
        k = r["round_k"]
        if k not in by_round_c:
            continue
        rounds_compared.append(k)
        for aid, v in r["reference_mean"].items():
            diffs.append(abs(v - by_round_c[k]["reference_mean"][aid]))
    if not diffs:
        return {"pass": None, "note": "no overlapping validation rounds"}
    arr = np.asarray(diffs)
    return {
        "rounds_compared": rounds_compared,
        "max_abs_difference": float(arr.max()),
        "pass": bool(arr.max() <= TOL),
        "note": "identical => the reference never saw the corruption",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--attack", required=True)
    ap.add_argument("--noise", default=None)
    ap.add_argument("--B", type=float, default=12.5)
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--attacked-source", default="batch_1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    clean, attack = Path(args.clean), Path(args.attack)
    report: dict = {"B": args.B, "M": args.M, "B_over_M": args.B / args.M,
                    "attacked_source": args.attacked_source, "tolerance": TOL}

    report["clean_in_run_identity"] = in_run_identity(clean, args.B, args.M, expect_shift=False)
    report["attack_in_run_identity"] = in_run_identity(attack, args.B, args.M, expect_shift=True)
    report["attack_vs_clean_round0"] = cross_run_round0(
        clean, attack, args.B, args.M, args.attacked_source, expect_constant_shift=True)
    report["attack_reference_unshifted"] = reference_unshifted(clean, attack)

    gating = [
        report["clean_in_run_identity"]["pass"],
        report["attack_in_run_identity"]["pass"],
        report["attack_vs_clean_round0"]["pass"],
    ]
    if report["attack_reference_unshifted"]["pass"] is not None:
        gating.append(report["attack_reference_unshifted"]["pass"])

    if args.noise:
        noise = Path(args.noise)
        report["noise_vs_clean_round0"] = cross_run_round0(
            clean, noise, args.B, args.M, args.attacked_source, expect_constant_shift=False)
        report["noise_reference_unshifted"] = reference_unshifted(clean, noise)
        gating.append(report["noise_vs_clean_round0"]["pass"])
        if report["noise_reference_unshifted"]["pass"] is not None:
            gating.append(report["noise_reference_unshifted"]["pass"])
        # The perturbation must actually be applied, and only to the named source.
        rc = read_jsonl(clean / "rounds.jsonl")[0]
        rn = read_jsonl(noise / "rounds.jsonl")[0]
        aids = agent_ids([rc])
        d_agg = np.asarray([
            float(rn["constraints"][a]["mechanism_reported_cost_return"])
            - float(rc["constraints"][a]["mechanism_reported_cost_return"]) for a in aids])
        report["noise_is_live"] = {
            "round0_aggregate_shift_per_owner": [round(float(x), 6) for x in d_agg],
            "all_owners_shifted": bool(np.abs(d_agg).min() > TOL),
            "shifts_differ_across_owners": bool(d_agg.std(ddof=1) > TOL),
            "pass": bool(np.abs(d_agg).min() > TOL and d_agg.std(ddof=1) > TOL),
            "note": "D draws one perturbation per (round, owner) -- §5.3; a "
                    "constant shift across owners would mean it was not i.i.d.",
        }
        gating.append(report["noise_is_live"]["pass"])

    report["pass"] = bool(all(gating))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def show(name: str, blk: dict) -> None:
        v = blk.get("pass")
        tag = "PASS" if v is True else ("n/a " if v is None else "FAIL")
        print(f"  [{tag}] {name}")

    print(f"\nA1 mechanism validation  (B={args.B}, M={args.M}, B/M={args.B/args.M:.5f})")
    for k, v in report.items():
        if isinstance(v, dict) and "pass" in v:
            show(k, v)
    print(f"\n{'MECHANISM VALIDATION PASSES' if report['pass'] else 'MECHANISM VALIDATION FAILS'}"
          f" -> {out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
