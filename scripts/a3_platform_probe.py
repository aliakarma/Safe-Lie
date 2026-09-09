#!/usr/bin/env python
"""A3 cross-platform probe: does a second machine reproduce this one bitwise?

docs/a3_gates.md section 9. Decides, by measurement rather than assumption,
whether A3 may split its 12 runs by CONDITION (6/6, three seeds, ~3.25 days)
or must split them by whole SEED (4/4, two seeds, ~2.2 days).

The branch is chosen by a property of the HARDWARE, measured before any A3
run exists. It cannot be influenced by an A3 result, which is what keeps the
pre-declaration intact.

Three levels, reported separately, because they fail for different reasons
and the distinction is diagnostic:

  1. SOURCE SEEDS      pure integer arithmetic (SeedSequence -> PCG64 ->
                       integers). Platform-independent by construction. A
                       mismatch here means the configs differ, not the
                       hardware -- stop and fix that first.
  2. POLICY CHECKSUM   torch.manual_seed -> network initialisation. Tests
                       whether PyTorch produces identical weights on this
                       architecture.
  3. SOURCE VALUES     the full MuJoCo + PyTorch float path, 2000 steps per
                       trajectory. This is what actually has to match for a
                       cross-machine paired contrast to be valid.

Usage, on the SECOND machine, after cloning:

    python scripts/train.py --config configs/experiment/a3/_platform_probe.yaml --eval-every 1
    python scripts/a3_platform_probe.py \
        --reference results/a2_mechanism_check/mech_rce_clean \
        --candidate results/a3_platform_probe/probe

Exit 0 whichever branch is selected -- a "not identical" result is an
expected, informative outcome, not a failure.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

TOL = 1e-9


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def load(run_dir: Path) -> dict:
    rounds = read_jsonl(run_dir / "rounds.jsonl")
    seeds = read_jsonl(run_dir / "source_seeds.jsonl")
    aids = sorted(rounds[0]["constraints"].keys())
    sids = sorted(r["source_id"] for r in rounds[0]["constraints"][aids[0]]["reports"])
    values = np.array(
        [[[next(x["value"] for x in r["constraints"][a]["reports"] if x["source_id"] == sid)
           for sid in sids] for a in aids] for r in rounds], dtype=float)   # K x N x S
    meta_path = run_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {
        "dir": str(run_dir), "rounds": len(rounds), "agent_ids": aids, "source_ids": sids,
        "values": values,
        "seeds": [s["seeds"] for s in seeds],
        "policy_checksums": [s["policy_checksum"] for s in seeds],
        "config_snapshot": meta.get("config_snapshot", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True,
                    help="the run produced on the FIRST machine (committed in the repo)")
    ap.add_argument("--candidate", required=True,
                    help="the run just produced on THIS machine")
    ap.add_argument("--out", default="results/a3_platform_probe/a3_platform_probe.json")
    args = ap.parse_args()

    ref, cand = load(Path(args.reference)), load(Path(args.candidate))
    K = min(ref["rounds"], cand["rounds"])

    rep: dict = {
        "reference": ref["dir"], "candidate": cand["dir"],
        "rounds_compared": K,
        "this_machine": {
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "python": platform.python_version(),
        },
    }
    try:
        import torch
        rep["this_machine"]["torch"] = torch.__version__
        rep["this_machine"]["numpy"] = np.__version__
    except Exception:
        pass

    # ---- level 0: the two runs must be the same experiment at all
    same_config = True
    for key in ("seed", "total_steps", "rollout_length"):
        a, b = ref["config_snapshot"].get(key), cand["config_snapshot"].get(key)
        if a != b:
            same_config = False
            rep.setdefault("config_mismatch", {})[key] = {"reference": a, "candidate": b}
    a = (ref["config_snapshot"].get("source_collection") or {}).get("seed_entropy")
    b = (cand["config_snapshot"].get("source_collection") or {}).get("seed_entropy")
    if a != b:
        same_config = False
        rep.setdefault("config_mismatch", {})["seed_entropy"] = {"reference": a, "candidate": b}
    rep["level0_same_experiment"] = {
        "pass": same_config,
        "note": "run_id and output_dir are permitted to differ; nothing else is.",
    }

    # ---- level 1: source seeds (pure integer -- must match anywhere)
    seeds_match = ref["seeds"][:K] == cand["seeds"][:K]
    rep["level1_source_seeds"] = {
        "identical": bool(seeds_match),
        "note": ("Pure integer arithmetic; a mismatch means the configs differ, "
                 "not the hardware."),
    }

    # ---- level 2: policy initialisation (torch)
    ck_match = ref["policy_checksums"][:K] == cand["policy_checksums"][:K]
    round0_ck_match = ref["policy_checksums"][0] == cand["policy_checksums"][0]
    rep["level2_policy_checksum"] = {
        "round0_identical": bool(round0_ck_match),
        "all_rounds_identical": bool(ck_match),
        "reference_round0": ref["policy_checksums"][0],
        "candidate_round0": cand["policy_checksums"][0],
        "note": "torch.manual_seed -> network init. Round 0 is init alone.",
    }

    # ---- level 3: source values (the full float path)
    d = cand["values"][:K] - ref["values"][:K]
    r0 = np.abs(d[0])
    rep["level3_source_values"] = {
        "round0_max_abs_diff": float(r0.max()),
        "round0_mean_abs_diff": float(r0.mean()),
        "round0_identical": bool(r0.max() <= TOL),
        "all_rounds_max_abs_diff": float(np.abs(d).max()),
        "all_rounds_identical": bool(np.abs(d).max() <= TOL),
        "per_round_max_abs_diff": [float(np.abs(d[k]).max()) for k in range(K)],
        "tolerance": TOL,
        # CORRECTED after the first real probe (Mac mini, 2026-09-09). The
        # original note claimed a round-0 difference is "pure numerics rather
        # than accumulated divergence" because both runs are still under the
        # same policy. That reasoning silently assumed level 2 passes. It did
        # NOT: torch initialises different weights on arm64 than on AMD64, so
        # the two runs are under DIFFERENT policies from round 0 and the
        # round-0 value difference confounds the initial weights with the
        # float path. The two cannot be separated from this probe's outputs.
        # The BRANCH decision is unaffected -- either cause invalidates a
        # cross-machine paired contrast, and level 1 passing already rules
        # out a config error -- but the interpretation is not what was
        # written, so it is not left standing.
        "note": ("Round 0 is decisive for the BRANCH decision, but it is only "
                 "attributable to numerics alone when level 2 also passes. If "
                 "level 2 fails, the round-0 difference confounds a different "
                 "initial policy with the float path, and this probe cannot "
                 "separate them."),
        "round0_attributable_to_numerics_alone": bool(round0_ck_match),
    }

    # ---- the pre-declared branch
    bit_identical = bool(seeds_match and round0_ck_match and r0.max() <= TOL)
    rep["bit_identical"] = bit_identical
    rep["a3_branch"] = "BRANCH-A" if bit_identical else "BRANCH-B"
    rep["a3_design"] = (
        {"split": "by condition, 6/6", "seeds": 3, "runs_per_machine": 6,
         "critical_path_days": 3.25,
         "rationale": "Cross-machine contrasts are valid, so the 12 runs balance "
                      "evenly and all three seeds are affordable."}
        if bit_identical else
        {"split": "by whole seed, 4/4", "seeds": 2, "runs_per_machine": 4,
         "critical_path_days": 2.2,
         "rationale": "Cross-machine contrasts are NOT valid, so each machine must "
                      "run all four conditions of the seeds it owns. Two seeds fit "
                      "the budget; a third would cost 2.2 extra days of critical "
                      "path because 12 runs split 8/4."}
    )

    if not same_config:
        rep["a3_branch"] = "INVALID"
        rep["a3_design"] = {"error": "The two runs are not the same experiment; "
                                     "fix the config mismatch and re-run the probe."}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print("=" * 70)
    print("A3 CROSS-PLATFORM PROBE")
    print("=" * 70)
    print(f"  reference : {ref['dir']}")
    print(f"  candidate : {cand['dir']}   ({rep['this_machine']['machine']}, "
          f"{rep['this_machine']['platform']})")
    print(f"  rounds compared: {K}")
    print()
    print(f"  [{'OK ' if same_config else 'BAD'}] level 0  same experiment")
    print(f"  [{'OK ' if seeds_match else 'BAD'}] level 1  source seeds identical")
    print(f"  [{'OK ' if round0_ck_match else '-- '}] level 2  policy init identical at round 0")
    print(f"  [{'OK ' if r0.max() <= TOL else '-- '}] level 3  source values identical at round 0 "
          f"(max |diff| = {r0.max():.3e})")
    print()
    if not same_config:
        print("  VERDICT: INVALID -- the two runs are not the same experiment.")
        print("           Fix the config mismatch and re-run before choosing a branch.")
    elif not seeds_match:
        print("  VERDICT: INVALID -- source seeds differ. That path is pure integer")
        print("           arithmetic, so this is a config problem, not a hardware one.")
    elif bit_identical:
        print("  VERDICT: BIT-IDENTICAL -> BRANCH-A")
        print("           A3 splits by CONDITION, 6 runs per machine, 3 seeds, ~3.25 days.")
        print("           Cross-machine paired contrasts are valid; record this, it")
        print("           also frees future campaigns from the by-seed restriction.")
    else:
        print("  VERDICT: NOT bit-identical -> BRANCH-B")
        print("           A3 splits by whole SEED, 4 runs per machine, 2 seeds, ~2.2 days.")
        print("           Every contrast stays on one machine. This is the expected")
        print("           outcome across x86-64 and ARM64 and is not a failure.")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
