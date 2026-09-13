#!/usr/bin/env python
"""DR-S0: prove the A1 code path did not move between A1 and HEAD.

docs/dose_response_gates.md section 5(b) argues by READING that the two
source deltas since A1 ran -- the PID controller plus its config fields,
the RCE calibration-clone fix, and the `_provenance()` metadata block --
are inert on the `controller=lagrangian` + `defense=mean` path this
campaign uses. Reading is not evidence. This script is the evidence.

It re-runs `configs/experiment/a1/b_seed0.yaml` at the CURRENT HEAD for
two rounds and requires the result to be BITWISE identical to the
committed `results/runs_a1/B_seed0` rounds 0-1 -- every source report,
aggregate, residual, multiplier, PPO statistic, oracle true cost and
source seed pair, compared as exact equality, not within a tolerance.

Four fields are changed to make a two-round run possible, and each is
argued to be incapable of touching rounds 0-1:

  run_id / output_dir   filing only.
  total_steps           4000 instead of 500000. `total_steps` is read in
                        exactly two places (`loop.py:771`,
                        `experiment.py:179/233`) and both compute the
                        round count. There is no learning-rate anneal, no
                        entropy schedule and no curriculum keyed to it, so
                        it cannot change what round 0 does.
  validation_rounds     [] instead of [25,75,125,175,225]. The config's
                        cross-field validator rejects validation rounds
                        outside 0..n_rounds-1, so a two-round clone cannot
                        keep them. Emptying them is inert here because the
                        reference stream is `SeedSequence(entropy).spawn(
                        M+1)[M]` -- a stream the M replicas never touch --
                        and it is drawn from only when `collect_reference`
                        is True, which at rounds 0-1 it never is under
                        either setting.

Timing fields are excluded from the comparison; nothing else is.

Usage:
    python scripts/dose_s0_regression.py
    python scripts/dose_s0_regression.py --compare-only   # skip retraining
Exit 0 iff the run is bitwise identical.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

PARENT_CFG = ROOT / "configs/experiment/a1/b_seed0.yaml"
REFERENCE = ROOT / "results/runs_a1/B_seed0"
WORK = ROOT / "results/runs_dose/_s0_regress"
CLONE_CFG = WORK / "s0_config.yaml"
REPORT = ROOT / "results/runs_dose/dose_s0_regression.json"
N_ROUNDS = 2
ROLLOUT = 2000

# Wall-clock fields are the only legitimately non-deterministic leaves.
EXCLUDE_LEAVES = {"wall_clock_s", "reference_wall_clock_s"}


def build_clone() -> str:
    text = PARENT_CFG.read_text(encoding="utf-8")

    def sub(old: str, new: str, label: str) -> None:
        nonlocal text
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"DR-S0 clone: {label!r} matched {n} times, expected 1")
        text = text.replace(old, new)

    sub("run_id: B_seed0", "run_id: _s0_regress", "run_id")
    sub("output_dir: results/runs_a1", "output_dir: results/runs_dose", "output_dir")
    sub("total_steps: 500000    # 250 rounds at rollout_length 2000 -- UNCHANGED",
        f"total_steps: {N_ROUNDS * ROLLOUT}    # DR-S0: {N_ROUNDS} rounds only",
        "total_steps")
    sub("  validation_rounds: [25, 75, 125, 175, 225]",
        "  validation_rounds: []    # DR-S0: outside a 2-round range; inert (see docstring)",
        "validation_rounds")
    return text


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def diff_tree(a, b, path: str, out: list[dict]) -> None:
    """Exact structural comparison. Any inequality is recorded, not tolerated."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in EXCLUDE_LEAVES:
                continue
            if k not in a or k not in b:
                out.append({"path": f"{path}.{k}", "reason": "key present on one side only",
                            "in_new": k in a, "in_reference": k in b})
                continue
            diff_tree(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append({"path": path, "reason": "length differs",
                        "new": len(a), "reference": len(b)})
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            diff_tree(x, y, f"{path}[{i}]", out)
    else:
        if a != b:
            entry = {"path": path, "new": a, "reference": b}
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                entry["abs_diff"] = abs(float(a) - float(b))
            out.append(entry)


def compare() -> dict:
    run = WORK
    files = ("rounds.jsonl", "oracle.jsonl", "source_seeds.jsonl")
    result: dict = {
        "gate": "DR-S0",
        "predeclaration": "docs/dose_response_gates.md section 5(b)/7",
        "claim": ("the A1 code path is unchanged between the A1 commits and HEAD: "
                  "a1/b_seed0.yaml re-run at HEAD reproduces committed B_seed0 "
                  "rounds 0-1 bitwise"),
        "new_run": str(run.relative_to(ROOT)),
        "reference_run": str(REFERENCE.relative_to(ROOT)),
        "rounds_compared": N_ROUNDS,
        "comparison": "exact equality; only wall-clock leaves excluded",
        "excluded_leaves": sorted(EXCLUDE_LEAVES),
        "files": {},
    }
    ok = True
    for fname in files:
        new_p, ref_p = run / fname, REFERENCE / fname
        if not new_p.exists():
            result["files"][fname] = {"pass": False, "error": f"{new_p} missing"}
            ok = False
            continue
        new, ref = read_jsonl(new_p)[:N_ROUNDS], read_jsonl(ref_p)[:N_ROUNDS]
        if len(new) < N_ROUNDS:
            result["files"][fname] = {"pass": False,
                                      "error": f"only {len(new)} rounds in the new run"}
            ok = False
            continue
        diffs: list[dict] = []
        diff_tree(new, ref, fname, diffs)
        numeric = [d["abs_diff"] for d in diffs if "abs_diff" in d]
        result["files"][fname] = {
            "pass": not diffs,
            "n_differences": len(diffs),
            "max_abs_numeric_difference": max(numeric) if numeric else 0.0,
            "first_differences": diffs[:10],
        }
        ok &= not diffs

    md = run / "run_metadata.json"
    if md.exists():
        meta = json.loads(md.read_text(encoding="utf-8"))
        result["new_run_provenance"] = {
            "git_sha": (meta.get("git") or {}).get("sha"),
            "git_dirty": (meta.get("git") or {}).get("dirty"),
            "python": meta.get("python"),
            "platform": meta.get("platform"),
            "provenance": meta.get("provenance"),
        }
    ref_md = REFERENCE / "run_metadata.json"
    if ref_md.exists():
        rmeta = json.loads(ref_md.read_text(encoding="utf-8"))
        result["reference_run_provenance"] = {
            "git_sha": (rmeta.get("git") or {}).get("sha"),
            "git_dirty": (rmeta.get("git") or {}).get("dirty"),
            "python": rmeta.get("python"),
            "platform": rmeta.get("platform"),
        }
    result["pass"] = bool(ok)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare-only", action="store_true",
                    help="skip retraining and compare an existing _s0_regress run")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    CLONE_CFG.write_text(build_clone(), encoding="utf-8")
    print(f"DR-S0 clone config -> {CLONE_CFG.relative_to(ROOT)}")

    if not args.compare_only:
        rounds_file = WORK / "rounds.jsonl"
        if rounds_file.exists():
            # Never resume into this: a partial regression run must be redone
            # from scratch, not continued, or the comparison is meaningless.
            for f in ("rounds.jsonl", "oracle.jsonl", "source_seeds.jsonl",
                      "validation_reference.jsonl", "checkpoint.pt",
                      "run_metadata.json"):
                p = WORK / f
                if p.exists():
                    p.unlink()
            print("  cleared a previous regression run (never resumed)")
        log = WORK / "s0_regress.log"
        print(f"  running {N_ROUNDS} rounds at HEAD ... (~4 min; log: "
              f"{log.relative_to(ROOT)})")
        with log.open("w", encoding="utf-8") as fh:
            rc = subprocess.call(
                [PY, str(ROOT / "scripts/train.py"), "--config", str(CLONE_CFG),
                 "--eval-every", "1"],
                stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
        if rc != 0:
            print(f"HALT: training exited rc={rc}; see {log}")
            return 1

    report = compare()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== DR-S0: {report['new_run']} vs {report['reference_run']} ===")
    for fname, f in report["files"].items():
        mark = "PASS" if f.get("pass") else "FAIL"
        print(f"  [{mark}] {fname}: {f.get('n_differences', '?')} differences, "
              f"max |diff| = {f.get('max_abs_numeric_difference', '?')}")
        for d in f.get("first_differences", []):
            print(f"          {d}")
        if "error" in f:
            print(f"          {f['error']}")
    if "new_run_provenance" in report:
        p = report["new_run_provenance"]
        print(f"  new run sha={str(p['git_sha'])[:10]} dirty={p['git_dirty']}")
    if "reference_run_provenance" in report:
        p = report["reference_run_provenance"]
        print(f"  reference sha={str(p['git_sha'])[:10]} dirty={p['git_dirty']}")
    print(f"\n{'DR-S0 PASS -- code path unchanged' if report['pass'] else 'DR-S0 FAIL'}")
    print(f"-> {REPORT.relative_to(ROOT)}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
