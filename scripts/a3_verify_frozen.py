#!/usr/bin/env python
"""A3-G1-i / A3-G2-i: prove the A3 configs change exactly what A3 declares.

Three independent checks, because A3 has two things to hold fixed at once --
the A1/G10/A2 architecture, and the internal pairing that makes CRN work.

  1. WITHIN A SEED, the four conditions must be identical except `run_id`,
     `attack` and `defense`. That equality is what makes A'/B'/C'/E' a
     common-random-number design: same `seed`, same `seed_entropy`, so the
     round-0 source draws are bit-identical and the only difference is the
     mechanism.

  2. AGAINST A2, each config may differ only in the declared A3 changes:
     `run_id`, `output_dir`, `attack`, `defense`, `sources` (3 -> 5 entries),
     and inside `source_collection` ONLY `M` and `seed_entropy`. Every other
     source_collection field -- R_m, workers, chunks_per_worker,
     validation_rounds, R_ref, mode -- must be untouched.

  3. THE OPERATING POINT: M=5 with defense.f=1 gives |T| = M - 2f = 3, which
     is exactly `min_retained`. That equality is the entire point of A3 --
     one less retained value and the MAD floors again, reproducing A2's
     degeneracy. It is asserted here so a later edit cannot quietly undo it.

Usage:  python scripts/a3_verify_frozen.py
Exit 0 iff every config passes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safelie.utils.config import load_experiment_config  # noqa: E402

SEEDS = (0, 1, 2)
CONDS = ("a", "b", "c", "e")
LABEL = {"a": "A'", "b": "B'", "c": "C'", "e": "E'"}

A3 = {(c, s): f"configs/experiment/a3/{c}_seed{s}.yaml" for c in CONDS for s in SEEDS}
A2_ARCH_REF = "configs/experiment/a2/c_seed0.yaml"       # the frozen M=3 architecture

# docs/a3_gates.md section 5
ATTACKED = {0: "batch_1", 1: "batch_3", 2: "batch_5"}
ENTROPY = {0: 25873748826208450093657097358164192435,
           1: 224463726283861287585842441982236519383,
           2: 11901184105024366541245543018062542394}
A2_ENTROPY = {286314957402113664887331205920951063913,
              52021175099534868945312500562751741008,
              335268198726974212397240672597355197200}

# `seed` is permitted HERE because the A2 architecture reference is one
# specific seed's config and A3 spans three. Seed equality is not waived --
# check 1 enforces it where it actually matters, WITHIN a seed, which is what
# the CRN pairing depends on.
PERMITTED_VS_A2 = {"run_id", "output_dir", "attack", "defense", "sources", "seed"}
PERMITTED_SOURCE_COLLECTION = {"M", "seed_entropy"}
PERMITTED_WITHIN_SEED = {"run_id", "attack", "defense"}

EXPECTED_RCE = {"name": "rce", "f": 1, "beta": 1.5, "sigma_min": 0.001,
                "min_retained": 3, "use_reliability_weights": False,
                "calibration_rounds": 20, "calibration_alpha": 0.05}
EXPECTED_MEAN = {"name": "mean", "f": 0, "beta": 1.5, "sigma_min": 0.001,
                 "min_retained": 3, "use_reliability_weights": False,
                 "calibration_rounds": 20, "calibration_alpha": 0.05}


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = json.dumps(v, sort_keys=True, default=str)
    return out


def main() -> int:
    cfgs = {k: load_experiment_config(p).model_dump() for k, p in A3.items()}
    a2 = load_experiment_config(A2_ARCH_REF).model_dump()
    ok = True
    report: dict = {"checks": {}}

    # ---- check 1: within-seed CRN pairing
    print("CHECK 1 -- within a seed, the four conditions differ only in "
          "run_id / attack / defense")
    for s in SEEDS:
        base = cfgs[("a", s)]
        fb = flatten(base)
        for c in ("b", "c", "e"):
            fc = flatten(cfgs[(c, s)])
            differing = sorted(k for k in fb if fb[k] != fc[k])
            illegal = [k for k in differing if k.split(".")[0] not in PERMITTED_WITHIN_SEED]
            entry = {
                "differing_keys": differing, "illegal": illegal,
                "seed_equal": base["seed"] == cfgs[(c, s)]["seed"],
                "entropy_equal": (base["source_collection"]["seed_entropy"]
                                  == cfgs[(c, s)]["source_collection"]["seed_entropy"]),
            }
            entry["pass"] = bool(not illegal and entry["seed_equal"] and entry["entropy_equal"])
            ok &= entry["pass"]
            report["checks"][f"within_seed{s}_{LABEL[c]}_vs_A'"] = entry
            flag = "PASS" if entry["pass"] else "FAIL"
            print(f"  [{flag}] seed {s}: {LABEL[c]} vs A'  differs in {differing}")
            if illegal:
                print(f"          ILLEGAL: {illegal}")
            if not entry["entropy_equal"]:
                print("          ILLEGAL: seed_entropy differs within a seed -> no CRN")

    # ---- check 2: architecture vs the frozen A2 config
    print("\nCHECK 2 -- against the frozen A2 architecture, only the declared "
          "A3 changes")
    fa2 = flatten(a2)
    for (c, s), cfg in sorted(cfgs.items()):
        f3 = flatten(cfg)
        keys = set(fa2) | set(f3)
        differing = sorted(k for k in keys if fa2.get(k) != f3.get(k))
        illegal = []
        for k in differing:
            top = k.split(".")[0]
            if top == "source_collection":
                if k.split(".", 1)[1] not in PERMITTED_SOURCE_COLLECTION:
                    illegal.append(k)
            elif top not in PERMITTED_VS_A2:
                illegal.append(k)
        entry = {"differing_keys": differing, "illegal": illegal,
                 "R_m": cfg["source_collection"]["R_m"],
                 "workers": cfg["source_collection"]["workers"],
                 "validation_rounds": cfg["source_collection"]["validation_rounds"],
                 "R_ref": cfg["source_collection"]["R_ref"],
                 "rollout_length": cfg["rollout_length"],
                 "total_steps": cfg["total_steps"]}
        entry["pass"] = not illegal
        ok &= entry["pass"]
        report["checks"][f"vs_a2_{LABEL[c]}_seed{s}"] = entry
        flag = "PASS" if entry["pass"] else "FAIL"
        print(f"  [{flag}] {LABEL[c]}_seed{s}")
        if illegal:
            print(f"          ILLEGAL: {illegal}")

    # ---- check 3: the operating point, the mapping, and the entropies
    print("\nCHECK 3 -- operating point, attacked-source mapping, entropies")
    for (c, s), cfg in sorted(cfgs.items()):
        M = cfg["source_collection"]["M"]
        f = cfg["defense"]["f"]
        n_sources = len(cfg["sources"]["sources"])
        n_classes = len({x["independence_class"] for x in cfg["sources"]["sources"]})
        retained = M - 2 * f
        e = {
            "M": M, "defense_f": f, "n_sources": n_sources,
            "n_independence_classes": n_classes,
            "retained_set_size": retained,
            "min_retained": cfg["defense"]["min_retained"],
            "margin_is_live": bool(cfg["defense"]["name"] == "rce"
                                   and retained >= cfg["defense"]["min_retained"]),
            "defense_matches_declared": (cfg["defense"] == EXPECTED_RCE
                                         if c in ("c", "e") else
                                         cfg["defense"] == EXPECTED_MEAN),
            "seed_entropy": cfg["source_collection"]["seed_entropy"],
            "entropy_matches_declared": cfg["source_collection"]["seed_entropy"] == ENTROPY[s],
            "entropy_not_reused_from_a2": (cfg["source_collection"]["seed_entropy"]
                                           not in A2_ENTROPY),
            "attacked": cfg["attack"]["corrupted_source_ids"],
        }
        if c in ("b", "c"):
            e["attack_mapping_ok"] = e["attacked"] == [ATTACKED[s]]
            e["attack_is_primary"] = (cfg["attack"]["name"] == "primary"
                                      and cfg["attack"]["budget_ratio"] == 0.5
                                      and cfg["attack"]["direction"] == "negative"
                                      and cfg["attack"]["support"] == "persistent")
        else:
            e["attack_mapping_ok"] = e["attacked"] is None
            e["attack_is_primary"] = cfg["attack"]["name"] == "none" and cfg["attack"]["f"] == 0
        e["pass"] = bool(
            M == 5 and n_sources == 5 and n_classes == 5
            and e["defense_matches_declared"]
            and e["entropy_matches_declared"] and e["entropy_not_reused_from_a2"]
            and e["attack_mapping_ok"] and e["attack_is_primary"]
            and (e["margin_is_live"] if c in ("c", "e") else True)
        )
        ok &= e["pass"]
        report["checks"][f"operating_point_{LABEL[c]}_seed{s}"] = e
        flag = "PASS" if e["pass"] else "FAIL"
        extra = (f"  |T|={retained} >= min_retained={e['min_retained']} -> margin LIVE"
                 if c in ("c", "e") else "  undefended reference")
        print(f"  [{flag}] {LABEL[c]}_seed{s}: M={M} f={f}{extra}")

    # entropies must be pairwise distinct across seeds
    ents = {s: cfgs[("a", s)]["source_collection"]["seed_entropy"] for s in SEEDS}
    distinct = len(set(ents.values())) == len(ents)
    report["entropies_distinct_across_seeds"] = distinct
    ok &= distinct
    print(f"  [{'PASS' if distinct else 'FAIL'}] the three seed entropies are pairwise distinct")

    report["pass"] = bool(ok)
    out = Path("results/runs_a3/a3_frozen_verification.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{'ALL CONFIGS PASS' if ok else 'FAILURES PRESENT'} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
