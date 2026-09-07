#!/usr/bin/env python
"""A2-G1-i / A2-G2-i: prove the A2 configs differ from their frozen
counterparts ONLY in the defense block.

Two independent comparisons per seed, because A2 has two treatments to
isolate and each has its own reference:

    C_seed<s>  vs  a1/b_seed<s>.yaml   -> may differ only in defense.*
    E_seed<s>  vs  the clean config    -> may differ only in defense.*

Anything else that differs is an illegal difference and halts the queue.
`run_id` and `output_dir` are bookkeeping and are permitted; `seed` and
`source_collection.seed_entropy` are required to be EQUAL, not merely
permitted to differ, because that equality is what makes A2's pairing a
common-random-number design (docs/a2_rce_gates.md section 3).

The C-vs-B comparison additionally requires the ENTIRE attack block to be
identical, which is the machine-checkable form of "C and B differ only by
the defense" -- the A2-G2 stop condition.

Usage:  python scripts/a2_verify_frozen.py
Exit 0 iff every config passes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safelie.utils.config import load_experiment_config  # noqa: E402

CLEAN = {
    0: "configs/experiment/g9_batch_clean.yaml",
    1: "configs/experiment/g10_batch_clean_seed1.yaml",
    2: "configs/experiment/g10_batch_clean_seed2.yaml",
}
ATTACK_B = {s: f"configs/experiment/a1/b_seed{s}.yaml" for s in (0, 1, 2)}

# (condition, seed) -> (a2 config, reference config)
PAIRS = {
    **{("C", s): (f"configs/experiment/a2/c_seed{s}.yaml", ATTACK_B[s]) for s in (0, 1, 2)},
    **{("E", s): (f"configs/experiment/a2/e_seed{s}.yaml", CLEAN[s]) for s in (0, 1, 2)},
}

ATTACKED = {0: "batch_1", 1: "batch_2", 2: "batch_3"}   # gates doc section 3

PERMITTED_TOP = {"run_id", "output_dir", "defense"}

# The shipped RCE defense, exactly as pre-declared in section 2 of the
# gates document. Not a suggestion: any drift here is an illegal config.
EXPECTED_DEFENSE = {
    "name": "rce",
    "f": 1,
    "beta": 1.5,
    "sigma_min": 0.001,
    "min_retained": 3,
    "use_reliability_weights": False,
    "calibration_rounds": 20,
    "calibration_alpha": 0.05,
}


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
    ok = True
    report = {}
    for (cond, seed), (path, ref_path) in sorted(PAIRS.items()):
        ref = load_experiment_config(ref_path).model_dump()
        a2 = load_experiment_config(path).model_dump()
        fr, fa = flatten(ref), flatten(a2)

        assert set(fr) == set(fa), "config schemas diverged"
        differing = sorted(k for k in fr if fr[k] != fa[k])
        illegal = [k for k in differing if k.split(".")[0] not in PERMITTED_TOP]

        entry = {
            "config": path,
            "reference": ref_path,
            "differing_keys": differing,
            "illegal_differences": illegal,
            "seed_equal": ref["seed"] == a2["seed"],
            "seed_entropy_equal": (
                ref["source_collection"]["seed_entropy"]
                == a2["source_collection"]["seed_entropy"]
            ),
            "defense": a2["defense"],
            "defense_matches_declared": a2["defense"] == EXPECTED_DEFENSE,
            # M > 2f is what makes the trimmed mean defined at all, and
            # M - 2f == 1 is what makes the audited degeneracy (gates doc
            # section 4) hold. Both are asserted, so a later M change
            # cannot silently invalidate the audit.
            "M": a2["source_collection"]["M"],
            "retained_size_M_minus_2f": a2["source_collection"]["M"] - 2 * a2["defense"]["f"],
        }

        if cond == "C":
            entry["attack_identical_to_B"] = ref["attack"] == a2["attack"]
            entry["attacked_source"] = a2["attack"]["corrupted_source_ids"]
            entry["expected_attacked_source"] = [ATTACKED[seed]]
            entry["budget_ratio"] = a2["attack"]["budget_ratio"]
            attack_ok = (
                entry["attack_identical_to_B"]
                and entry["attacked_source"] == entry["expected_attacked_source"]
                and entry["budget_ratio"] == 0.5
                and a2["attack"]["direction"] == "negative"
                and a2["attack"]["support"] == "persistent"
            )
        else:
            entry["attack_is_none"] = a2["attack"]["name"] == "none" and a2["attack"]["f"] == 0
            entry["attacked_source"] = a2["attack"]["corrupted_source_ids"]
            attack_ok = entry["attack_is_none"] and a2["attack"]["corrupted_source_ids"] is None

        entry["pass"] = bool(
            not illegal
            and entry["seed_equal"]
            and entry["seed_entropy_equal"]
            and entry["defense_matches_declared"]
            and entry["retained_size_M_minus_2f"] == 1
            and attack_ok
        )
        ok &= entry["pass"]
        report[f"{cond}_seed{seed}"] = entry

        flag = "PASS" if entry["pass"] else "FAIL"
        print(f"[{flag}] {cond}_seed{seed}  vs {Path(ref_path).name}")
        print(f"        differs only in: {differing}")
        if illegal:
            print(f"        ILLEGAL DIFFERENCES: {illegal}")
        if not entry["seed_entropy_equal"]:
            print("        ILLEGAL: seed_entropy differs from the pair -> no CRN")
        if not entry["defense_matches_declared"]:
            print(f"        ILLEGAL: defense {entry['defense']} != declared {EXPECTED_DEFENSE}")
        if cond == "C" and not entry.get("attack_identical_to_B"):
            print("        ILLEGAL: attack block differs from B -> C and B differ by more "
                  "than the defense (A2-G2 stop condition)")
        if not attack_ok:
            print(f"        ILLEGAL: attack block wrong for condition {cond}")

    out = Path("results/runs_a2/a2_frozen_verification.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"pass": bool(ok), "configs": report}, indent=2), encoding="utf-8")
    print(f"\n{'ALL CONFIGS PASS' if ok else 'FAILURES PRESENT'} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
