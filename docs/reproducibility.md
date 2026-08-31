# Reproducibility

Two different questions get conflated in most "reproducibility"
discussions. This document keeps them separate, as `PROJECT_REPORT.md`
insists throughout.

## Question 1: Is the *software pipeline* reproducible?

**Yes, verified.**

- Same seed + same config → bitwise-identical logs
  (`tests/smoke/test_determinism.py::test_same_seed_same_config_gives_bitwise_identical_logs`).
- A checkpoint-restored run continues bitwise-identically to an
  uninterrupted one
  (`tests/smoke/test_determinism.py::test_checkpoint_restore_continues_bitwise_identically`)
  — this required capturing every source of randomness touched between
  rounds (see [training.md](training.md)'s checkpointing section), not
  just model weights, and was the one genuine bug found while building
  this repository.
- Every experiment config validates deterministically and fails loudly
  on any scientifically-invalid setting before any environment, policy,
  or source is constructed (`tests/unit/test_config.py`).
- The theory suite (`safelie.theory`) reproduces Theorem 1 to machine
  precision (1e-10) across 8 topologies and 4 corruption schedules on
  every run — this is a deterministic numerical fact, not a statistical
  one.

## Question 2: Are the *paper's scientific results* reproduced?

**No — and this repository does not attempt to.**

The source paper's own Section 5 states every number in its Table 3/4 is
`\projected{}`: *"design specifications against which we will validate an
implementation, not measured results, and not to be read as evidence."*
`PROJECT_REPORT.md` reiterates this in its own preamble. This repository:

1. Implements the theory, taxonomy, and defense faithfully (verified —
   see [paper_implementation_mapping.md](paper_implementation_mapping.md)).
2. Provides a CPU-only synthetic environment and demonstrates the attack
   → dual-bias → policy pathway moves through the code correctly (see
   below).
3. **Does not** run the paper's actual environments (Safe MAMuJoCo,
   Safety-Gymnasium), because that requires the MuJoCo physics engine and
   multi-agent wrapper packages this build does not have, and which
   `PROJECT_REPORT.md` itself assigns to a Google Colab GPU stage, not to
   repository construction.

**No number produced by this repository should be compared to
`main_iclr.tex`'s Table 3 or Table 4, or presented as evidence for or
against the paper's hypothesis.**

## What was actually run, and what it shows

Four local demo configs (`configs/experiment/local_demo_{clean,attack,rce,benign}.yaml`)
were run once each (1 seed, 75 rounds, ~45s each) on the synthetic
environment:

| Condition | True cost (mean, last 10 rounds, across 6 agents) |
|---|---|
| Clean (no attack) | 21.39, 20.63, 22.07, 21.78, 24.07, 21.39 |
| Attacked (f=1, undefended) | 21.43, 20.66, 22.11, 21.83, 24.09, 21.42 |
| Attacked, RCE-defended | 21.30, 20.57, 22.02, 21.70, 24.09, 21.33 |
| Benign control | 21.39, 20.63, 22.08, 21.78, 24.08, 21.39 |

**Directionally consistent with the paper's mechanism, at a magnitude
this toy setup does not claim is meaningful**: attacked true cost exceeds
clean for all 6 agents; RCE's true cost is below the attacked baseline's
for 5 of 6 agents (tied on the 6th); the benign control tracks the clean
baseline rather than the attack, unlike the attack itself. This is
exactly what `mean`-aggregation-under-Theorem-1 predicts: a single
corrupted source's effect divides across all M=7 sources
(`~12.5/7 ≈ 1.8` on the residual), so the effect is small but present,
not absent. **This is one seed on a hand-built toy environment, run once
each, and is reported honestly as such — it is evidence the pipeline is
wired correctly, not evidence about the paper's hypothesis.**

Full numbers: [SMOKE_TEST_REPORT.md](../SMOKE_TEST_REPORT.md).

## Completing Stage 2 (the Colab pilot)

`configs/experiment/pilot_A_clean.yaml` through `pilot_E_clean_rce.yaml`
are the literal encoding of `PROJECT_REPORT.md` §R8's compact pilot
matrix (M=7, f=1, β=1.5, η_λ=0.035, λ_max=25, ring topology, 5×10⁵ steps,
seeds [0,1,2]). They **validate** against the config schema today, but
**do not run**: `env.name: manyagent_ant` requires `safelie.envs.mamujoco`,
which raises `NotImplementedError` with instructions. To complete this:

1. On a Colab T4 High-RAM runtime, `pip install mujoco safety-gymnasium`
   plus a Multi-Agent MuJoCo factorization package.
2. Implement the adapter in `safelie/envs/mamujoco.py` against the
   `DualCostEnvWrapper` contract — its docstring lists exactly what is
   needed (three method implementations, following
   `safelie.envs.synthetic`'s isolation pattern).
3. Run the full local smoke suite against the new adapter (`pytest
   tests/`, then `python scripts/smoke_test.py`) — the **GREEN SIGNAL**
   gate — before spending any GPU time.
4. Run `notebooks/colab_full_experiment.ipynb`, whose 12-cell structure
   already matches `PROJECT_REPORT.md` §R7.3 and which currently reports
   exactly where it is blocked (cell 8, training).

No part of this repository's attack, defense, source-accounting, or
oracle-isolation logic needs to change to support a real environment —
everything downstream of `DualCostEnvWrapper` is environment-agnostic by
construction.

## What would be needed for the paper's Stage 3

The full-scale grid (`PROJECT_REPORT.md` §10.2: 300+ runs, 10⁷ steps,
75-300 GPU-days by the report's own corrected estimate) is out of scope
for any single-machine or single-Colab-session build. Stage 3 is
explicitly optional and selective in the report's own plan (§R9) —
expand only the dimensions a completed Stage-2 pilot justifies.
