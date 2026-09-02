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
3. **Runs** the paper's environments as far as they exist. Safe MAMuJoCo
   is implemented (`safelie.envs.mamujoco`) and the `pilot_*` configs
   execute. What it does **not** do is treat the resulting numbers as a
   reproduction: the pilot is 1/20 of the paper's step budget, and three
   deviations forced by the reference implementation (below) mean the
   headline environment is not the paper's environment.

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

## Stage 2 (the compact pilot): what running it establishes

`configs/experiment/pilot_A_clean.yaml` through `pilot_E_clean_rce.yaml`
are the literal encoding of `PROJECT_REPORT.md` §R8's compact pilot
matrix (M=7, f=1, β=1.5, η_λ=0.035, λ_max=25, ring topology, 5×10⁵ steps,
seeds [0,1,2]). They run:

```bash
pip install "safelie[mujoco]"
python scripts/calibrate_cost.py --config configs/experiment/pilot_A_clean.yaml
python scripts/train.py --config configs/experiment/pilot_A_clean.yaml
```

**This is not a GPU workload.** Nothing in `safelie` moves a tensor to
CUDA; the networks are small MLPs stepped one observation at a time in a
Python loop. Measured throughput is ~130 env-steps/s, so a single pilot
run takes roughly two hours on CPU and a T4 does not help. Prefer a
high-CPU runtime; use the checkpoint/resume path across sessions.

### Three deviations that bound what the pilot may claim

Implementing the adapter surfaced three facts about the reference
implementation that no amount of compute resolves. They are recorded in
full in `safelie/envs/mamujoco.py`'s docstring and
`docs/assumptions.md`; in summary:

1. **ManyAgent Ant is not a Safe MAMuJoCo environment.** The reference
   implementation's threshold table has no entry for it and its
   constructor asserts on the name. The paper's primary environment does
   not exist in the implementation the paper cites. It runs here only on
   the `gymnasium_robotics` backend, with a cost function this repository
   supplies. `halfcheetah_6x1` is the one genuinely N=6 configuration the
   reference implementation does support.
2. **Safe MAMuJoCo's cost is shared, not per-agent** — one global speed
   indicator, copied to every agent. The paper's per-agent `C^i` collapses
   to a single shared constraint under it, and `[GAP]` G4 becomes vacuous.
   `cost_mode` selects between the reference behaviour and a per-agent
   variant.
3. **The reference thresholds do not bind at this scale.** They are
   calibrated to a converged agent; the pilot trains for 5×10⁵ steps. At
   Ant's 2.418 the constraint sits ~30x from binding, which would make all
   five conditions coincide for reasons unrelated to the hypothesis. The
   threshold is calibrated by measurement instead — see
   `scripts/calibrate_cost.py`.

No part of this repository's attack, defense, source-accounting, or
oracle-isolation logic changed to support a real environment — everything
downstream of `DualCostEnvWrapper` is environment-agnostic by
construction, as designed. What did change: the learner now sizes its
networks from the constructed environment rather than from
`EnvConfig.obs_dim`/`action_dim`, which defaulted to 8/2 and would have
silently built 8-dim policies for 63-dim observations.

## What would be needed for the paper's Stage 3

The full-scale grid (`PROJECT_REPORT.md` §10.2: 300+ runs, 10⁷ steps,
75-300 GPU-days by the report's own corrected estimate) is out of scope
for any single-machine or single-Colab-session build. Stage 3 is
explicitly optional and selective in the report's own plan (§R9) —
expand only the dimensions a completed Stage-2 pilot justifies.
