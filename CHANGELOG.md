# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — Initial repository build

Reference implementation of the theory, attack taxonomy, and RCE defense
from "When the Safety Signal Lies: Adversarial Corruption of Safety-Cost
Feedback in Constrained Multi-Agent RL", built from `PROJECT_REPORT.md`
and `main_iclr.tex`.

### Added

- `safelie.theory`: numerical verification of Theorem 1 (corruption mass
  conservation), Proposition 1 (spreading/stealth), and a new closed-loop
  synthetic diagnostic addressing the report's W1 finding. CPU-only, no
  RL required.
- `safelie.consensus`: doubly stochastic mixing-matrix construction for
  six topologies (`complete`, `ring`, `star`, `erdos_renyi`, `identity`,
  `shared_constraint`).
- `safelie.sources`: the M-source registry with independence-class
  accounting (`effective_M`), closing the report's W4 finding.
- `safelie.attacks`: the four-axis corruption taxonomy, the primary and
  adaptive-stealth attack operators, the benign falsification control,
  and a ground-truth attack ledger.
- `safelie.defenses`: the aggregator zoo (mean, coordinate median, Krum,
  trimmed mean, RCE) with explicit, tested handling of the degenerate
  operating points the report's Table 4 leaves undefined (W2).
- `safelie.envs`: the `DualCostEnvWrapper` contract, an oracle-isolation
  guard, and a synthetic CPU environment for local verification (Safe
  MAMuJoCo integration is intentionally not implemented — see
  `safelie.envs.mamujoco`).
- `safelie.algos` / `safelie.training`: a from-scratch PyTorch
  MAPPO-Lagrangian implementation (PPO-clip, GAE, per-agent cost critics,
  the unconditional dual update).
- `safelie.eval`: the withheld oracle evaluator, detection-gap and
  violation metrics, and the pre-registered success criterion.
- `safelie.governance`: the `SourceAuditor` (the `M >= 2f+1` deployment
  checklist over independence classes, not raw report count).
- `safelie.analysis`: Welch's t-test with Holm correction (gated to >= 5
  seeds, matching decision D6) and measured-only summary tables.
- Local CPU experiment configs (`configs/experiment/local_demo_*.yaml`,
  `smoke.yaml`) and the literal Stage-2 Colab pilot specs
  (`configs/experiment/pilot_*.yaml`), which validate but are not
  executable without the (unimplemented) MuJoCo adapter.
- Full test suite: unit, property, theory, smoke, and isolation tests —
  see `SMOKE_TEST_REPORT.md` for what was actually executed.

### Known limitations

See `IMPLEMENTATION_STATUS.md` for the full list. Headline items: no
Safe MAMuJoCo / Safety-Gymnasium integration (Colab-stage work); no
Stage-2/Stage-3 experiments have been run; the adaptive/Byzantine attack
axes are implemented but not wired into the default training loop.
