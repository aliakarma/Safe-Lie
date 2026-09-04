#!/usr/bin/env python
"""Derive (and audit) the per-seed source-RNG entropy for G10.

WHY THIS SCRIPT EXISTS.  `SourceCollectionConfig.seed_entropy` is a fixed
constant (src/safelie/utils/config.py:72) that does NOT depend on
`cfg.seed`.  The replica and reference RNG streams are spawned from it
alone (src/safelie/training/source_batch.py:331).  So re-running the G9
config with `--seed 1` would give a policy trained from a different torch
seed but source batches drawn from the *identical* seed sequence as seed 0.

That is not acceptable for a three-seed reproducibility study: the source
sampling noise would be common-mode across seeds, and "the aggregate is
calibrated in all three seeds" would then be close to one observation
repeated three times rather than three independent ones.  docs/g10_gates.md
section 2 therefore fixes an explicit, published derivation, and this
script both computes it and proves the resulting streams are disjoint from
seed 0's -- BEFORE any G10 run is launched.

The derivation, fixed here and never re-chosen:

    seed 0 : seed_entropy = 286314957402113664887331205920951063913
             (the G9 constant, verbatim -- seed 0 IS the committed G9 run
             and is not re-run)
    seed k : seed_entropy = int(sha256(b"safelie/g10/source-entropy/seed=<k>")[:16])

Nothing about this changes the source ARCHITECTURE.  It changes which
seeds the architecture draws, which is exactly what an independent
training seed is supposed to change.

Usage:
    python scripts/g10_derive_source_entropy.py
    python scripts/g10_derive_source_entropy.py --out results/g10_entropy_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

G9_ENTROPY = 286_314_957_402_113_664_887_331_205_920_951_063_913
M = 3
R_M = 30
R_REF = 120
N_ROUNDS = 250
VALIDATION_ROUNDS = (25, 75, 125, 175, 225)
SEED_HI = 2**31 - 1


def entropy_for_seed(k: int) -> int:
    if k == 0:
        return G9_ENTROPY
    digest = hashlib.sha256(f"safelie/g10/source-entropy/seed={k}".encode()).digest()
    return int.from_bytes(digest[:16], "big")


def replay_streams(entropy: int) -> dict:
    """Replay exactly the draw order `ParallelBatchSourceCollector.collect`
    uses, without touching an environment.

    Per round the M replica streams each draw R_m (env, torch) pairs; on a
    validation round the reference stream then draws R_ref more.  The draw
    order does not depend on the policy, so the whole 250-round seed
    schedule is knowable offline -- which is what makes a pre-launch
    separation audit possible at all.
    """
    root = np.random.SeedSequence(entropy)
    children = root.spawn(M + 1)
    replicas = [np.random.default_rng(ss) for ss in children[:M]]
    reference = np.random.default_rng(children[M])

    per_replica_env: list[set[int]] = [set() for _ in range(M)]
    ref_env: set[int] = set()
    replica_env_seq: list[int] = []
    all_env: list[int] = []
    all_torch: list[int] = []

    def draw(rng, n):
        raw = rng.integers(0, SEED_HI, size=(n, 2), dtype=np.int64)
        return [(int(a), int(b)) for a, b in raw]

    for k in range(N_ROUNDS):
        for m in range(M):
            for e, t in draw(replicas[m], R_M):
                per_replica_env[m].add(e)
                replica_env_seq.append(e)
                all_env.append(e)
                all_torch.append(t)
        if k in VALIDATION_ROUNDS:
            for e, t in draw(reference, R_REF):
                ref_env.add(e)
                all_env.append(e)
                all_torch.append(t)

    return {
        "entropy": entropy,
        "spawn_keys": [[int(ss.entropy)] + list(map(int, ss.spawn_key)) for ss in children],
        "per_replica_env": per_replica_env,
        "ref_env": ref_env,
        "replica_env_seq": replica_env_seq,
        "all_env": all_env,
        "all_torch": all_torch,
    }


def read_recorded_replica_env_seeds(run_dir: Path) -> list[int]:
    """Replica env seeds in the order `source_seeds.jsonl` logged them.

    The reference stream is logged separately in
    `validation_reference.jsonl`, so this is a replica-only sequence and
    must be compared against the replica-only subsequence of the replay.
    """
    out: list[int] = []
    p = run_dir / "source_seeds.jsonl"
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for rid in sorted(rec["seeds"]):
            if rid == "__reference__":
                continue
            for pair in rec["seeds"][rid]:
                out.append(int(pair[0]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-against", default="results/runs_constraint_batch_g9/g9_batch_clean")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    streams = {k: replay_streams(entropy_for_seed(k)) for k in args.seeds}
    report: dict = {
        "derivation": "seed 0 = the G9 constant; seed k = int(sha256('safelie/g10/source-entropy/seed=<k>')[:16])",
        "seeds": {},
        "cross_seed": {},
        "replay_vs_recorded": {},
    }

    print("=== derived seed_entropy ===")
    for k in args.seeds:
        e = entropy_for_seed(k)
        s = streams[k]
        within_env = len(s["all_env"]) - len(set(s["all_env"]))
        entry = {
            "seed_entropy": e,
            "n_env_seeds": len(s["all_env"]),
            "n_unique_env_seeds": len(set(s["all_env"])),
            "within_run_env_duplicates": within_env,
            "within_run_torch_duplicates": len(s["all_torch"]) - len(set(s["all_torch"])),
            "replica_pairwise_overlap": {
                f"{i}-{j}": len(s["per_replica_env"][i] & s["per_replica_env"][j])
                for i in range(M)
                for j in range(i + 1, M)
            },
            "replica_vs_reference_overlap": {
                str(i): len(s["per_replica_env"][i] & s["ref_env"]) for i in range(M)
            },
        }
        report["seeds"][str(k)] = entry
        print(f"  seed {k}: {e}")
        print(
            f"    {len(s['all_env'])} env seeds, {len(set(s['all_env']))} unique, "
            f"{within_env} within-run duplicates"
        )
        print(
            f"    replica pairwise overlap {entry['replica_pairwise_overlap']}, "
            f"replica-vs-reference {entry['replica_vs_reference_overlap']}"
        )

    print("")
    print("=== cross-seed stream separation ===")
    ks = list(args.seeds)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a = set(streams[ks[i]]["all_env"])
            b = set(streams[ks[j]]["all_env"])
            inter = len(a & b)
            exp = len(a) * len(b) / SEED_HI
            identical = streams[ks[i]]["all_env"] == streams[ks[j]]["all_env"]
            report["cross_seed"][f"{ks[i]}-{ks[j]}"] = {
                "n_shared_env_seeds": inter,
                "expected_by_chance": exp,
                "sequences_identical": identical,
            }
            print(
                f"  seed {ks[i]} vs seed {ks[j]}: {inter} shared env seeds "
                f"(chance expectation {exp:.2f}); identical sequence = {identical}"
            )

    audit_dir = Path(args.audit_against)
    recorded = read_recorded_replica_env_seeds(audit_dir)
    if recorded and 0 in streams:
        replay0 = streams[0]["replica_env_seq"]
        match = replay0[: len(recorded)] == recorded
        report["replay_vs_recorded"] = {
            "run_dir": str(audit_dir),
            "n_recorded_replica_env_seeds": len(recorded),
            "replay_matches_recorded": match,
        }
        print("")
        print("=== replay check against the committed G9 seed-0 log ===")
        print(
            f"  {len(recorded)} recorded replica env seeds; the offline replay reproduces "
            f"them exactly: {match}"
        )
        print(
            "  (this is what licenses trusting the same replay for seeds 1 and 2, "
            "which have not been run yet)"
        )
        for k in ks:
            if k == 0:
                continue
            overlap = len(set(recorded) & set(streams[k]["all_env"]))
            print(f"  seed {k} shares {overlap} env seeds with the ACTUAL G9 seed-0 replica draws")
            report["cross_seed"].setdefault(f"0-{k}", {})["n_shared_with_recorded_g9"] = overlap

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
