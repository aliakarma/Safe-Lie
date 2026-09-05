#!/usr/bin/env python
"""A1 §22 stop condition: prove the A1 configs differ from their frozen
G10 clean counterparts ONLY in the corruption treatment.

Compares each A1 config's fully-resolved Pydantic model against the clean
config for the same training seed, field by field, and fails if any key
outside the permitted set differs. The permitted set is deliberately tiny:

    run_id        bookkeeping
    output_dir    bookkeeping
    attack.*      THE corruption treatment -- the only new scientific factor

`seed` and `source_collection.seed_entropy` are required to be EQUAL, not
merely permitted to differ: that equality is what makes the A1 pairing a
common-random-number design (docs/a1_attack_gates.md §5).

Usage:  python scripts/a1_verify_frozen.py
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
A1 = {(c, s): f"configs/experiment/a1/{c.lower()}_seed{s}.yaml" for c in "BD" for s in (0, 1, 2)}

# Pre-declared balanced mapping (docs/a1_attack_gates.md §3).
ATTACKED = {0: "batch_1", 1: "batch_2", 2: "batch_3"}

PERMITTED_TOP = {"run_id", "output_dir", "attack"}


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
    for (cond, seed), path in sorted(A1.items()):
        clean = load_experiment_config(CLEAN[seed]).model_dump()
        a1 = load_experiment_config(path).model_dump()
        fc, fa = flatten(clean), flatten(a1)

        assert set(fc) == set(fa), "config schemas diverged"
        differing = sorted(k for k in fc if fc[k] != fa[k])
        illegal = [k for k in differing if k.split(".")[0] not in PERMITTED_TOP]

        entry = {
            "config": path,
            "clean_reference": CLEAN[seed],
            "differing_keys": differing,
            "illegal_differences": illegal,
            "seed_equal": clean["seed"] == a1["seed"],
            "seed_entropy_equal": (
                clean["source_collection"]["seed_entropy"]
                == a1["source_collection"]["seed_entropy"]
            ),
            "attacked_source": a1["attack"]["corrupted_source_ids"],
            "expected_attacked_source": [ATTACKED[seed]],
            "budget_ratio": a1["attack"]["budget_ratio"],
            "defense_is_undefended": a1["defense"]["name"] == "mean" and a1["defense"]["f"] == 0,
        }
        entry["pass"] = (
            not illegal
            and entry["seed_equal"]
            and entry["seed_entropy_equal"]
            and entry["attacked_source"] == entry["expected_attacked_source"]
            and entry["budget_ratio"] == 0.5
            and entry["defense_is_undefended"]
        )
        ok &= entry["pass"]
        report[f"{cond}_seed{seed}"] = entry

        flag = "PASS" if entry["pass"] else "FAIL"
        print(f"[{flag}] {cond}_seed{seed}  vs {Path(CLEAN[seed]).name}")
        print(f"        differs only in: {differing}")
        if illegal:
            print(f"        ILLEGAL DIFFERENCES: {illegal}")
        if not entry["seed_entropy_equal"]:
            print("        ILLEGAL: seed_entropy differs from the clean pair -> no CRN")
        if entry["attacked_source"] != entry["expected_attacked_source"]:
            print(f"        ILLEGAL: attacked {entry['attacked_source']}, "
                  f"mapping says {entry['expected_attacked_source']}")

    out = Path("results/runs_a1/a1_frozen_verification.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"pass": bool(ok), "configs": report}, indent=2), encoding="utf-8")
    print(f"\n{'ALL CONFIGS PASS' if ok else 'FAILURES PRESENT'} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
