"""Safe Multi-Agent MuJoCo / Safety-Gymnasium adapter — NOT IMPLEMENTED.

Report reference: main_iclr.tex §5.1 ("Environments. Safe Multi-Agent
MuJoCo, primarily ManyAgent Ant (N=6) and HalfCheetah 2x3 (N=2), plus the
multi-agent tasks of Safety-Gymnasium"); PROJECT_REPORT.md §R7.1
("All MARL training ... on Colab, T4 GPU + High-RAM ... the local machine
never runs a full-scale training job").

**Classification: (D) Not implementable from the materials available to
this repository build.** MuJoCo and the Safe-MAMuJoCo / Safety-Gymnasium
multi-agent wrapper packages are heavy, platform-sensitive dependencies
(a physics engine plus research-grade multi-agent wrappers not reliably
installable in this build environment) that the source report itself
assigns to the Google Colab GPU stage, not to local CPU verification. No
training run in the paper's actual environments has been attempted here;
attempting to bundle a partial or mocked MuJoCo integration and presenting
it as functional would misrepresent what has been verified.

What IS implemented, and how this gap is closed operationally:

  - The exact `safelie.envs.dual_cost.DualCostEnvWrapper` contract this
    adapter would need to satisfy.
  - `safelie.envs.synthetic.SyntheticConstrainedMarlEnv`, a CPU-only
    environment satisfying that same contract, used for every local
    test and the tiny end-to-end smoke run (S15).
  - The full training/attack/defense/eval pipeline
    (`safelie.training`, `safelie.attacks`, `safelie.defenses`,
    `safelie.eval`) is environment-agnostic: it depends only on the
    `DualCostEnvWrapper` protocol, not on any concrete environment.

To complete this adapter for the Colab (Stage-2) pilot:

  1. ``pip install mujoco safety-gymnasium`` (and, for ManyAgent Ant
     specifically, a Multi-Agent MuJoCo factorization package such as
     the one used by MACPO/HARL) inside the Colab runtime.
  2. Implement a class here satisfying `DualCostEnvWrapper`: `reset` /
     `step` translate the underlying multi-agent env's per-agent
     observations, the shared task reward, and the environment's own
     per-agent cost signal into a `DualCostStep`; `oracle_handle` /
     `_oracle_handle_privileged` follow `safelie.envs.synthetic`'s pattern
     exactly (a sealed public view, a privileged one used only by
     `safelie.eval.oracle.OracleEvaluator`).
  3. Resolve `[GAP]` G4 (peer observability of another agent's cost) by
     restricting cost features to what is measurable from shared/global
     simulator state, per PROJECT_REPORT.md §5, Phase 1's recommendation.
  4. Run the full local smoke suite (`pytest tests/`) against the new
     adapter before spending any Colab GPU time, exactly as prescribed
     by the GREEN SIGNAL gate (PROJECT_REPORT.md §R5).

See docs/reproducibility.md for the full Stage-1 vs. Stage-2 division of
labour and docs/paper_implementation_mapping.md for this component's
status.
"""

from __future__ import annotations


def build_mamujoco_env(*_args, **_kwargs):
    raise NotImplementedError(
        "Safe MAMuJoCo / Safety-Gymnasium integration is not implemented in "
        "this repository. See safelie.envs.mamujoco's module docstring for "
        "why (heavy GPU-stage dependency, out of scope for local CPU "
        "verification per PROJECT_REPORT.md §R7.1) and what is needed to "
        "add it for the Colab pilot."
    )
