#!/usr/bin/env python
"""DR-S10 stop condition: prove the six dose configs differ from A1's
condition-B configs ONLY in `attack.budget_ratio`.

Two comparisons per config, both through the CURRENT Pydantic schema so
that fields added after A1 ran (`dual.controller`, `dual.k_*`) resolve on
both sides and cannot masquerade as a treatment difference
(docs/dose_response_gates.md section 5c):

  1. vs `configs/experiment/a1/b_seed{k}.yaml` -- the direct parent.
     Permitted to differ: run_id, output_dir, attack.budget_ratio.
     Nothing else, including every other attack.* field.

  2. vs the seed's frozen clean config -- the chain back to G10.
     Permitted to differ: run_id, output_dir, attack.*.

`seed` and `source_collection.seed_entropy` are required to be EQUAL on
both comparisons, not merely permitted to differ: that equality is what
makes the four-dose design a common-random-number design (gates doc
section 4.1).

Usage:  python scripts/dose_verify_frozen.py
Exit 0 iff every config passes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safelie.utils.config import load_experiment_config  # noqa: E402

# Chain back to the frozen G10 clean architecture, exactly as
# scripts/a1_verify_frozen.py does.
CLEAN = {
    0: "configs/experiment/g9_batch_clean.yaml",
    1: "configs/experiment/g10_batch_clean_seed1.yaml",
    2: "configs/experiment/g10_batch_clean_seed2.yaml",
}
A1_PARENT = {k: f"configs/experiment/a1/b_seed{k}.yaml" for k in (0, 1, 2)}

# The dose assignment this campaign declares. Read from
# docs/dose_response_gates.md section 3; do not tune here.
DOSE = {"025": 0.25, "100": 1.0}
A1_DOSE = 0.5
D_BUDGET = 25.0
M_SOURCES = 3

# Pre-declared balanced mapping, inherited from A1 section 3 unchanged.
ATTACKED = {0: "batch_1", 1: "batch_2", 2: "batch_3"}

DR = {
    (tag, k): f"configs/experiment/dose/b{tag}_seed{k}.yaml"
    for tag in DOSE
    for k in (0, 1, 2)
}

PERMITTED_VS_A1 = {"run_id", "output_dir", "attack.budget_ratio"}
PERMITTED_VS_CLEAN_TOP = {"run_id", "output_dir", "attack"}


def flatten(d: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = json.dumps(v, sort_keys=True, default=str)
    return out


def main() -> int:
    ok = True
    report: dict = {
        "gate": "DR-S10",
        "predeclaration": "docs/dose_response_gates.md",
        "schema_note": (
            "both sides loaded through the CURRENT ExperimentConfig, so "
            "dual.controller / dual.k_* resolve identically and cannot appear "
            "as a treatment difference (gates doc 5c)"
        ),
        "configs": {},
    }

    for (tag, seed), path in sorted(DR.items()):
        name = f"B{tag}_seed{seed}"
        dr = load_experiment_config(path).model_dump()
        parent = load_experiment_config(A1_PARENT[seed]).model_dump()
        clean = load_experiment_config(CLEAN[seed]).model_dump()

        fd, fp, fc = flatten(dr), flatten(parent), flatten(clean)

        # --- 1. vs the A1 parent -------------------------------------------
        assert set(fd) == set(fp), "config schemas diverged (dose vs A1 parent)"
        diff_parent = sorted(k for k in fd if fd[k] != fp[k])
        illegal_parent = [k for k in diff_parent if k not in PERMITTED_VS_A1]

        # --- 2. vs the frozen clean config ---------------------------------
        # The clean configs predate `attack.corrupted_source_ids`, so their
        # resolved models legitimately lack nothing -- Pydantic supplies the
        # default -- but assert the key sets match rather than assume it.
        assert set(fd) == set(fc), "config schemas diverged (dose vs clean)"
        diff_clean = sorted(k for k in fd if fd[k] != fc[k])
        illegal_clean = [
            k for k in diff_clean if k.split(".")[0] not in PERMITTED_VS_CLEAN_TOP
        ]

        expected_ratio = DOSE[tag]
        B = expected_ratio * D_BUDGET
        entry = {
            "config": path,
            "a1_parent": A1_PARENT[seed],
            "clean_reference": CLEAN[seed],
            "differing_keys_vs_a1_parent": diff_parent,
            "illegal_differences_vs_a1_parent": illegal_parent,
            "differing_keys_vs_clean": diff_clean,
            "illegal_differences_vs_clean": illegal_clean,
            "seed_equal_to_parent": dr["seed"] == parent["seed"],
            "seed_entropy_equal_to_parent": (
                dr["source_collection"]["seed_entropy"]
                == parent["source_collection"]["seed_entropy"]
            ),
            "seed_entropy_equal_to_clean": (
                dr["source_collection"]["seed_entropy"]
                == clean["source_collection"]["seed_entropy"]
            ),
            "budget_ratio": dr["attack"]["budget_ratio"],
            "expected_budget_ratio": expected_ratio,
            "parent_budget_ratio": parent["attack"]["budget_ratio"],
            "B": B,
            "B_over_M": B / M_SOURCES,
            "attacked_source": dr["attack"]["corrupted_source_ids"],
            "expected_attacked_source": [ATTACKED[seed]],
            # DR-S11: no accidental RCE, asserted at config time as well as
            # at run time.
            "defense_is_undefended": (
                dr["defense"]["name"] == "mean" and dr["defense"]["f"] == 0
            ),
            # DR-S12: the operator itself must be untouched.
            "operator_unchanged": all(
                dr["attack"][f] == parent["attack"][f]
                for f in ("name", "f", "direction", "support", "adaptivity",
                          "consistency", "corrupted_source_ids")
            ),
            # The controller must resolve to the Eq. 2 path on both sides.
            "controller_resolves_lagrangian": (
                dr["dual"]["controller"] == "lagrangian"
                and parent["dual"]["controller"] == "lagrangian"
            ),
            "dual_unchanged": dr["dual"] == parent["dual"],
        }
        entry["pass"] = bool(
            not illegal_parent
            and not illegal_clean
            and entry["seed_equal_to_parent"]
            and entry["seed_entropy_equal_to_parent"]
            and entry["seed_entropy_equal_to_clean"]
            and entry["budget_ratio"] == expected_ratio
            and entry["parent_budget_ratio"] == A1_DOSE
            and entry["attacked_source"] == entry["expected_attacked_source"]
            and entry["defense_is_undefended"]
            and entry["operator_unchanged"]
            and entry["controller_resolves_lagrangian"]
            and entry["dual_unchanged"]
        )
        ok &= entry["pass"]
        report["configs"][name] = entry

        flag = "PASS" if entry["pass"] else "FAIL"
        print(f"[{flag}] {name}  B/d={expected_ratio}  B={B}  B/M={B / M_SOURCES:.5f}")
        print(f"        vs {Path(A1_PARENT[seed]).name}: differs only in {diff_parent}")
        print(f"        vs {Path(CLEAN[seed]).name}: differs only in {diff_clean}")
        if illegal_parent:
            print(f"        ILLEGAL vs A1 parent: {illegal_parent}")
        if illegal_clean:
            print(f"        ILLEGAL vs clean: {illegal_clean}")
        if not entry["seed_entropy_equal_to_parent"]:
            print("        ILLEGAL: seed_entropy differs from the A1 parent -> no CRN")
        if not entry["operator_unchanged"]:
            print("        ILLEGAL: the attack OPERATOR changed, not just its magnitude")

    out = ROOT / "results/runs_dose/dose_frozen_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report["pass"] = bool(ok)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{'ALL CONFIGS PASS' if ok else 'FAILURES PRESENT'} -> "
          f"{out.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
