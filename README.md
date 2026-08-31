# SafeLie — Corrupted-Signal Multi-Agent RL

Reference implementation, attack harness, and Robust Constraint Estimation
(RCE) defense for **"When the Safety Signal Lies: Adversarial Corruption
of Safety-Cost Feedback in Constrained Multi-Agent RL"** (ICLR 2026
submission, `main_iclr.tex`).

> **Status, read this first.** The source paper's own empirical section
> is explicitly marked `[PROJECTED]` — no training run backs its Table 3
> or Table 4. This repository does not change that: it is a from-scratch,
> tested implementation of the paper's theory, threat model, and defense,
> plus a CPU-only synthetic environment for local verification. **No
> number in this README, or anywhere else in this repository, reproduces
> the paper's projected results.** See
> [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) and
> [SMOKE_TEST_REPORT.md](SMOKE_TEST_REPORT.md) for exactly what has and
> has not been verified, and how.

## What this is

Safe multi-agent RL enforces safety with a primal-dual loop: each agent
estimates its cost return, compares it to a budget, and pushes the
residual into a Lagrange multiplier that penalizes unsafe behavior. In a
**decentralized** deployment, that cost estimate is a *learned quantity*
transmitted over a network — not observed from the environment. This
project treats that transmission channel as an attack surface, and
provides:

1. A formalization of the constrained Markov game with corrupted cost
   feedback, and a four-axis corruption taxonomy.
2. Three theoretical results, verified here **numerically to machine
   precision**, no RL required (`safelie.theory`):
   - **Corruption mass conservation** (Theorem 1): the total dual-variable
     bias injected by corrupted reports is exactly invariant to the
     consensus network's topology.
   - **Spreading and stealth** (Proposition 1): consensus redistributes a
     concentrated, detectable attack into a diffuse one, invisible to
     median-referenced monitoring.
   - **Multiplicative amplification** (Proposition 2): cost corruption
     enters the policy gradient scaled by the constraint-gradient norm,
     unlike additive reward corruption.
3. **Robust Constraint Estimation (RCE)**: trimmed-mean aggregation with
   an uncertainty margin and an *unconditional* bounded multiplier update
   — provably bounded-violation under `M >= 2f+1` honest sources, and
   provably live under over-reporting attacks that deadlock the naive
   gating defense.

## Repository structure

```text
src/safelie/
├── theory/        # Theorem 1 / Prop. 1 / closed-loop diagnostic — no RL, CPU, seconds
├── consensus/      # doubly stochastic mixing matrices, 6 topologies
├── sources/        # the M-source registry, independence-class accounting (W4)
├── attacks/        # the 4-axis taxonomy as composable operators + ground-truth ledger
├── defenses/        # mean, coordinate median, Krum, trimmed mean, RCE
├── envs/          # DualCostEnvWrapper contract, oracle isolation guard, synthetic env
├── algos/          # PyTorch MAPPO-Lagrangian (policy, reward critic, cost critic)
├── training/        # rollout buffer, PPO update, the unconditional dual update
├── eval/           # withheld oracle evaluator, metrics, pre-registered success criterion
├── governance/       # SourceAuditor: the M>=2f+1 deployment checklist
├── analysis/         # Welch/Holm statistics, measured-only summary tables
└── experiment.py     # the orchestrator tying the learner and the evaluator together
```

See [docs/architecture.md](docs/architecture.md) for how these fit
together, and
[docs/paper_implementation_mapping.md](docs/paper_implementation_mapping.md)
for a section-by-section map from `main_iclr.tex` / `PROJECT_REPORT.md` to
this code.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest tests/ -v                      # 116 tests: unit, property, theory, smoke, isolation, integration
python scripts/smoke_test.py          # the GREEN SIGNAL readiness gate
```

Full instructions, including on Windows / PowerShell: [docs/setup.md](docs/setup.md).

## Smoke test — CPU, seconds, no GPU

```bash
python scripts/train.py --config configs/experiment/smoke.yaml
```

Runs a tiny (3-agent, 4-round) clean training loop on the synthetic
environment end to end and writes `rounds.jsonl` + `oracle.jsonl` to
`results/runs/smoke_clean/`.

## Verify the theory (no RL, seconds)

```bash
jupyter notebook notebooks/theory_validation.ipynb
```

Or directly:

```python
from safelie.consensus.topologies import build_topology
from safelie.theory import verify_mass_conservation
import numpy as np

topologies = {name: build_topology(name, 6) for name in
              ["complete", "ring", "star", "identity", "shared_constraint"]}
delta = np.zeros(6); delta[0] = 1.5
result = verify_mass_conservation(topologies, [delta] * 500, eta=0.035, tol=1e-10)
assert result.passed   # holds to 1e-10 across every topology
```

## Run the attack / defense demo on the synthetic environment

```bash
make demo-clean     # no attack: reference baseline
make demo-attack    # f=1 persistent under-reporting, undefended (mean)
make demo-rce       # same attack, defended (RCE)
make demo-benign    # the falsification control: unbiased noise, not an attacker
python scripts/evaluate.py --run clean=results/runs/local_demo_clean \
                            --run attack=results/runs/local_demo_attack \
                            --run rce=results/runs/local_demo_rce \
                            --run benign=results/runs/local_demo_benign \
                            --budget 5.0
```

**This is not Safe MAMuJoCo and these are not the paper's numbers** — see
the warning at the top of this file and
[docs/reproducibility.md](docs/reproducibility.md). It is a from-scratch,
toy-scale environment used to demonstrate that the attack → dual-bias →
policy pathway is actually wired correctly through this codebase. Ran
once (1 seed, 75 rounds): the attacked condition's true cost was higher
than the clean baseline's for all 6 agents, RCE's true cost was lower
than the attacked baseline's for 5 of 6 agents (tied on the 6th), and the
benign control tracked the clean baseline rather than the attack —
directionally consistent with the paper's mechanism, at a magnitude this
toy setup does not claim to be meaningful. See
[SMOKE_TEST_REPORT.md](SMOKE_TEST_REPORT.md) for the actual numbers.

## Real experiments (Safe MAMuJoCo, Colab GPU) — not runnable here

The paper's actual environments (`ManyAgent Ant`, `HalfCheetah` 2×3,
Safety-Gymnasium) require the MuJoCo physics engine and multi-agent
wrapper packages. `PROJECT_REPORT.md`'s own compact execution plan
assigns this stage to a **Google Colab T4 GPU**, not local CPU
verification — and this repository build has no MuJoCo installed.
`safelie/envs/mamujoco.py` documents exactly what is needed to complete
that adapter; `configs/experiment/pilot_*.yaml` are the literal,
validated Stage-2 pilot configs from the report's §R8, ready to run once
the adapter exists. See [docs/reproducibility.md](docs/reproducibility.md).

## The M >= 2f+1 checklist

The paper's most transferable artifact, counted correctly (over
independence classes, not raw report count — see weakness W4 in
`PROJECT_REPORT.md` §13.4):

```bash
python scripts/audit_sources.py --preset m5_two_agent --assumed-f 2
# FAILS: the paper's own N=2 config nominally has M=5, but 3 of those
# reports are ensemble replicas sharing one process -- effective_M=3.
```

## Reproducibility

- **Software pipeline**: fully reproducible. Same seed + config gives
  bitwise-identical logs (`tests/smoke/test_determinism.py`); a
  checkpoint-restored run continues bitwise-identically to an
  uninterrupted one.
- **The paper's scientific results**: not reproduced, not attempted here
  — see the status warning at the top of this file. Executing that study
  is Stage 2/3 of `PROJECT_REPORT.md`'s plan and requires the MuJoCo
  adapter and Colab GPU time this build does not have.

Full details: [docs/reproducibility.md](docs/reproducibility.md).

## Documentation

| File | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Component-by-component system design |
| [docs/setup.md](docs/setup.md) | From-clean-environment install |
| [docs/training.md](docs/training.md) | Config schema, running experiments, checkpoints |
| [docs/evaluation.md](docs/evaluation.md) | Metrics, the oracle, the success criterion |
| [docs/inference.md](docs/inference.md) | Loading a checkpoint, rolling out a policy |
| [docs/reproducibility.md](docs/reproducibility.md) | What is and is not reproducible, and why |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common failures |
| [docs/paper_implementation_mapping.md](docs/paper_implementation_mapping.md) | Paper section → code, with status |
| [docs/assumptions.md](docs/assumptions.md) | Every `[DECISION]` / `[INFERRED]` choice, with rationale |
| [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) | What's done, partial, or not implemented, and why |
| [SMOKE_TEST_REPORT.md](SMOKE_TEST_REPORT.md) | Every test actually executed, with results |

## Limitations (see IMPLEMENTATION_STATUS.md for the full list)

- No Safe MAMuJoCo / Safety-Gymnasium integration — Colab-GPU-stage work,
  intentionally not attempted here.
- No Stage-2 or Stage-3 experiment has been run; nothing here should be
  cited as evidence for or against the paper's hypothesis.
- The adaptive (stealth) and Byzantine (equivocation) attack axes are
  implemented but not wired into the default training loop (matching the
  report's own compact-study scope, decisions D5/G15).
- Reliability weights (Algorithm 1 lines 2/10) are implemented as an
  opt-in, off-by-default feature — the paper declares them but never
  specifies an update rule (`[GAP]` G1).

## Citation

This repository accompanies an anonymous ICLR 2026 submission
(`paper/main_iclr.tex`). Cite the paper, not this repository, for the
theoretical results; see `paper/references.bib` for the bibliography this
work builds on (Yin et al. 2018 for trimmed-mean analysis, Blanchard et
al. 2017 for Krum — both cited in the paper as prior art, not claimed as
novel here either).

## License

MIT — see [LICENSE](LICENSE).
