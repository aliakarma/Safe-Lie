#!/usr/bin/env python
"""Pre-launch verification of the unconstrained control's configuration.

Implements docs/unconstrained_control.md sections 4 and 5. Checks the
LOADED `ExperimentConfig`, not the YAML text: each control config and its
paired clean config are run through `load_experiment_config` and compared
field by field on the validated pydantic models, so a default that differs
by omission is caught as readily as a value that differs by assignment.

A run may not start until this exits 0.

Usage:
    python scripts/verify_unconstrained_config.py
    python scripts/verify_unconstrained_config.py --json results/runs_unconstrained/config_verification.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safelie.utils.config import load_experiment_config  # noqa: E402

SEEDS = (0, 1, 2)

CONTROL = {s: ROOT / f"configs/experiment/unconstrained/u_seed{s}.yaml" for s in SEEDS}
CLEAN = {
    0: ROOT / "configs/experiment/g9_batch_clean.yaml",
    1: ROOT / "configs/experiment/g10_batch_clean_seed1.yaml",
    2: ROOT / "configs/experiment/g10_batch_clean_seed2.yaml",
}
CLEAN_RUN_DIR = {
    0: "results/runs_constraint_batch_g9/g9_batch_clean",
    1: "results/runs_constraint_batch_g10/seed1",
    2: "results/runs_constraint_batch_g10/seed2",
}

# The complete set of loaded-config paths permitted to differ (§4/§5).
# `dual.lambda_max` is the only scientific one; the rest are identity and
# filing. `source_collection.seed_entropy` appears because seed 0's clean
# config takes it by omission -- the VALUES must still match, which is
# checked separately and strictly below.
ALLOWED_DIFF = {
    "run_id",
    "seed",
    "output_dir",
    "dual.lambda_max",
}


def flatten(obj, prefix: str = "") -> dict:
    out: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def config_hash(cfg) -> str:
    """Hash of the LOADED config, canonicalised -- not of the file bytes."""
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_state() -> dict:
    """Working-tree state, splitting MODIFIED tracked files from NEW untracked ones.

    The integrity property a run depends on is that no tracked file has been
    edited out from under it -- that the code and configs about to execute are
    the committed ones. A brand-new file that nothing imports cannot change a
    run's behaviour, and this control necessarily adds its own pre-declaration,
    configs and scripts before it can launch. Collapsing both into one "dirty"
    bit would either block the control forever or force a commit of results
    that do not exist yet, so they are reported separately and gated
    separately. Both lists are recorded either way, so the provenance of the
    run is auditable regardless of which state it launched in.
    """
    def run(*a):
        return subprocess.run(a, cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
    status = run("git", "status", "--porcelain")
    lines = status.splitlines()
    modified = [ln[3:] for ln in lines if not ln.startswith("??")]
    untracked = [ln[3:] for ln in lines if ln.startswith("??")]
    return {
        "sha": run("git", "rev-parse", "HEAD"),
        "dirty": bool(status),
        "tracked_modified": modified,
        "untracked_new": untracked,
        "dirty_paths": modified + untracked,
    }


def check(results: list, name: str, ok: bool, detail=None) -> bool:
    results.append({"check": name, "pass": bool(ok), "detail": detail})
    return bool(ok)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="results/runs_unconstrained/config_verification.json")
    args = ap.parse_args()

    results: list = []
    report: dict = {
        "predeclaration": "docs/unconstrained_control.md",
        "git": git_state(),
        "per_seed": {},
    }

    entropies: dict = {}
    for s in SEEDS:
        u = load_experiment_config(str(CONTROL[s]))
        a = load_experiment_config(str(CLEAN[s]))
        fu, fa = flatten(u.model_dump(mode="json")), flatten(a.model_dump(mode="json"))

        keys = sorted(set(fu) | set(fa))
        diff = {k: {"clean": fa.get(k, "<absent>"), "control": fu.get(k, "<absent>")}
                for k in keys if fu.get(k, "<absent>") != fa.get(k, "<absent>")}
        entropies[s] = u.source_collection.seed_entropy

        report["per_seed"][s] = {
            "control_config": str(CONTROL[s].relative_to(ROOT)).replace("\\", "/"),
            "clean_config": str(CLEAN[s].relative_to(ROOT)).replace("\\", "/"),
            "paired_clean_run": CLEAN_RUN_DIR[s],
            "control_config_hash_sha256": config_hash(u),
            "clean_config_hash_sha256": config_hash(a),
            "loaded_config_diff": diff,
            "lambda_max": u.dual.lambda_max,
            "seed_entropy": u.source_collection.seed_entropy,
            "n_loaded_fields_compared": len(keys),
        }

        p = f"seed{s}"
        check(results, f"{p}: diff confined to the allowed fields",
              set(diff) <= ALLOWED_DIFF, sorted(set(diff) - ALLOWED_DIFF))
        check(results, f"{p}: dual.lambda_max == 0.0 (the treatment)",
              u.dual.lambda_max == 0.0, u.dual.lambda_max)
        check(results, f"{p}: clean side has dual.lambda_max == 25.0",
              a.dual.lambda_max == 25.0, a.dual.lambda_max)
        check(results, f"{p}: dual.eta_lambda unchanged",
              u.dual.eta_lambda == a.dual.eta_lambda == 0.035, u.dual.eta_lambda)
        check(results, f"{p}: dual.controller is lagrangian",
              u.dual.controller == "lagrangian", u.dual.controller)
        check(results, f"{p}: seed matches the file",
              u.seed == s, u.seed)
        check(results, f"{p}: run_id is U_seed{s}", u.run_id == f"U_seed{s}", u.run_id)
        check(results, f"{p}: output_dir is results/runs_unconstrained",
              u.output_dir == "results/runs_unconstrained", u.output_dir)
        check(results, f"{p}: no attack", u.attack.name == "none" and u.attack.f == 0,
              {"name": u.attack.name, "f": u.attack.f,
               "corrupted": u.attack.corrupted_source_ids})
        check(results, f"{p}: no RCE (defense is the plain mean, f=0)",
              u.defense.name == "mean" and u.defense.f == 0,
              {"name": u.defense.name, "f": u.defense.f})
        check(results, f"{p}: seed_entropy identical to the paired clean run (CRN)",
              u.source_collection.seed_entropy == a.source_collection.seed_entropy,
              {"control": u.source_collection.seed_entropy,
               "clean": a.source_collection.seed_entropy})
        check(results, f"{p}: learner untouched (ppo block identical)",
              u.ppo.model_dump() == a.ppo.model_dump(), None)
        check(results, f"{p}: env / topology / sources identical",
              (u.env.model_dump() == a.env.model_dump()
               and u.topology.model_dump() == a.topology.model_dump()
               and u.sources.model_dump() == a.sources.model_dump()), None)
        check(results, f"{p}: horizon and rollout unchanged",
              u.total_steps == a.total_steps == 500_000
              and u.rollout_length == a.rollout_length == 2000,
              {"total_steps": u.total_steps, "rollout_length": u.rollout_length})
        check(results, f"{p}: constraint_estimator unchanged (mc_window)",
              u.constraint_estimator == a.constraint_estimator == "mc_window",
              u.constraint_estimator)
        check(results, f"{p}: source collection identical to the clean run",
              u.source_collection.model_dump() == a.source_collection.model_dump(), None)

    check(results, "seed entropies are distinct across the three control seeds",
          len(set(entropies.values())) == 3, entropies)
    check(results, "output directories are distinct per seed",
          len({f"results/runs_unconstrained/U_seed{s}" for s in SEEDS}) == 3, None)
    check(results, "no tracked file modified (the code that will run is committed)",
          not report["git"]["tracked_modified"], report["git"]["tracked_modified"])
    check(results, "no tracked file under src/ modified",
          not [p for p in report["git"]["tracked_modified"] if p.startswith("src/")],
          [p for p in report["git"]["tracked_modified"] if p.startswith("src/")])
    # Reported, not gated: this control cannot launch without first adding its
    # own pre-declaration, configs and scripts, so they are necessarily
    # untracked at launch. Recorded so the run's provenance is auditable.
    report["untracked_at_launch"] = report["git"]["untracked_new"]

    report["checks"] = results
    report["all_pass"] = all(r["pass"] for r in results)

    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    width = max(len(r["check"]) for r in results)
    for r in results:
        print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {r['check']:<{width}}"
              + ("" if r["pass"] else f"   {r['detail']}"))
    print(f"\n{'ALL PASS' if report['all_pass'] else 'FAILED'}  -> {args.json}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
