# Implementation Status

Built from `PROJECT_REPORT.md` (engineering specification) and
`main_iclr.tex` (research methodology). See
[docs/paper_implementation_mapping.md](docs/paper_implementation_mapping.md)
for the full component-by-component map and
[SMOKE_TEST_REPORT.md](SMOKE_TEST_REPORT.md) for exactly what was
executed to verify each claim below.

## Fully Implemented

- **Theory suite** (`safelie.theory`): Theorem 1 (corruption mass
  conservation) verified to 1e-10 across 8 topologies × 4 corruption
  schedules; Proposition 1 (spreading/stealth) verified, including the
  operational stealth claim (median-referenced deviation); a new
  closed-loop synthetic diagnostic (addressing the report's weakness W1)
  implemented and run, measuring `rho ≈ 0.001` at the tested feedback
  gains — invariance appears robust at this scale, reported as a
  measurement, not a proof.
- **Consensus** (`safelie.consensus`): 6 topologies, all verified doubly
  stochastic to 1e-12; Metropolis-Hastings weight construction.
- **Source registry and independence accounting** (`safelie.sources`):
  `effective_M` over independence classes, not raw report count;
  verified to correctly reduce the paper's own N=2, M=5 configuration
  from nominal 5 to effective 3.
- **Attack module** (`safelie.attacks`): the four-axis taxonomy, the
  primary and benign-control operators wired into training, a
  ground-truth ledger verified to reconstruct injected mass exactly.
- **Defense module** (`safelie.defenses`): all 5 aggregators from Table
  4; RCE's two reported-but-unresolved-in-the-paper degenerate cases
  (`M<=2f`, `|T|<3`) fixed by raising / flooring-and-warning rather than
  silently degrading, per the report's own recommendation.
- **MAPPO-Lagrangian** (`safelie.algos`, `safelie.training`): a real
  PyTorch actor-critic-critic implementation — PPO clip, GAE (same gamma
  for reward and cost, per the paper's own warning), the unconditional
  dual update, checkpointing with full RNG-state capture.
- **Withheld oracle** (`safelie.eval.oracle`, `safelie.envs.guards`):
  structural isolation (no `true_cost` field reaches the learner at all,
  not merely a guarded one), verified by a grep-level AST test that no
  training/algorithm module imports the oracle module.
- **SourceAuditor** (`safelie.governance`): verified to reject the
  paper's own N=2, M=5 configuration at f=2 — the report's own Phase 10
  exit criterion.
- **Config validation** (`safelie.utils.config`): every CRITICAL
  cross-field rule the report names (S2, S5, S12) enforced at
  construction time.
- **Statistics** (`safelie.analysis`): Welch/Holm, gated to reject calls
  with fewer than 5 seeds (the compact pilot's 3-seed reporting uses a
  separate, honest per-seed sign/ordering path instead).

## Partially Implemented

| Component | What's done | What's approximated / deferred | Why |
|---|---|---|---|
| Environment | `DualCostEnvWrapper` contract, full isolation machinery, a working synthetic CPU environment | Not Safe MAMuJoCo / Safety-Gymnasium | Requires MuJoCo, a heavy platform-sensitive dependency the report assigns to a Colab GPU stage, not repository construction |
| Peer critic observability (`[GAP]` G4) | Resolved per the report's recommended reading: peer critics evaluate the constraint owner's own observation | Not validated against a real environment's actual state-sharing semantics | No real environment is integrated yet |
| RCE's Theorem 2 guarantee | The estimator is fully implemented and tested | The probabilistic *guarantee*'s precondition (`beta*sigma >= epsilon(M,f,alpha)`) is not runtime-checkable — this is the paper's own acknowledged weakness (W3), not a gap in this implementation | `epsilon` depends on the unknown sub-Gaussian parameter of honest sources |
| Proposition 2 (multiplicative amplification) | The mechanism it describes (combined advantage `adv_R - lambda*adv_C`) is implemented and drives real training | No standalone test isolates the amplification factor `‖∇_θ J_C‖` the way Theorem 1/Prop. 1 have dedicated numerical tests | Would require logging and correlating a gradient-norm quantity during actual RL training; the report names this as a suggested future test, not a required one |

## Not Implemented

| Component | Why | What's needed |
|---|---|---|
| Safe MAMuJoCo / Safety-Gymnasium adapter | Heavy, platform-sensitive dependency; explicitly a Colab-GPU-stage task in the report's own plan | See `safelie/envs/mamujoco.py`'s docstring — a ~3-method adapter against the existing `DualCostEnvWrapper` contract |
| MACPO, Dec-PDO, PID-Lagrangian, unconstrained MAPPO baselines | Report's own compact-study scope (§R2.1) designates MAPPO-Lagrangian as the sole Stage-2 victim; the rest are Stage-3 | Port from a reference implementation once Stage-2 is running |
| Reliability weights (Algorithm 1 lines 2, 10) | The paper declares and initializes them but no line of its own pseudocode reads them (`[GAP]` G1) | A specified update rule from the paper's authors, or a documented invented one — deliberately not fabricated here |
| Adaptive (stealth) and Byzantine attack axes in the default training loop | Both are implemented as standalone functions (`safelie.attacks.adaptive`, `safelie.attacks.byzantine`) but not wired into `safelie.training.loop`; the compact study's scope (decisions D5, and `[GAP]` G15) explicitly defers both to Stage 3 | Wire `stealth_attack`/`byzantine_attack` into `safelie.attacks.apply_attack`'s dispatch and `ExperimentRun.run_round` |
| Cumulative corruption budget (`Delta`, §3.2) enforcement | Declared in the paper, never used in its own evaluation protocol (`[GAP]` G14) | `AttackLedger.total_mass()` already tracks it; enforcing it as a constraint would be a small addition |
| Stage-2 / Stage-3 experiments | Blocked entirely on the environment adapter above | See `docs/reproducibility.md` |

## Technical Debt

- **Dependency on an unimplemented environment adapter blocks all real
  experiments.** This is the single largest piece of remaining work.
- **The synthetic environment's cost/budget scale required manual
  recalibration** (`d=25` → `d=5` for local demos) to make the
  constraint bind within a laptop-feasible run — see
  [docs/assumptions.md](docs/assumptions.md). A real environment
  (Safe MAMuJoCo) would need its own such calibration check before any
  pilot run, per the report's own §R6.1.
- **No distributed/multi-process execution.** The oracle isolation
  boundary is enforced within one process via structural typing and a
  capability handle, not via actual process separation. This is
  documented as sufficient for a research repository but is explicitly
  *not* what `docs/troubleshooting.md` / `PROJECT_REPORT.md` §15.2 would
  recommend for a production deployment (a separately-deployed,
  attested estimator service).
- **Krum, coordinate median, and plain TrimMean are implemented and
  tested but not exercised by any shipped experiment config** — they
  exist for the Stage-3 aggregator ablation (Table 4) and are currently
  reachable only via `safelie.defenses.aggregate("krum", ...)` directly
  or a hand-written config.

## Recommended Next Steps (priority order)

1. Complete the Safe MAMuJoCo adapter (`safelie/envs/mamujoco.py`) — this
   unblocks everything else.
2. Run the local GREEN SIGNAL gate against the new adapter
   (`pytest tests/`, `python scripts/smoke_test.py`) before any GPU time.
3. Run `notebooks/colab_full_experiment.ipynb`'s throughput probe (cell
   6) to get a real per-run time estimate before committing to the full
   pilot matrix.
4. Execute the Stage-2 compact pilot (`configs/experiment/pilot_*.yaml`)
   per `PROJECT_REPORT.md` §R8, reading condition A first to confirm the
   R6.1 precondition (constraint binding) before interpreting B/C/D.
5. Only then: wire in the stealth/Byzantine attack axes, the remaining
   baselines, and the remaining aggregators, per the report's own
   priority ordering (§R9).

## Research Reproducibility

**Software pipeline: fully reproducible**, verified (bitwise-identical
logs given the same seed; bitwise-identical checkpoint-restore
continuation). **Scientific results: not reproduced, not claimed.** See
[docs/reproducibility.md](docs/reproducibility.md) for the full
treatment of this distinction, including the one honest, small-scale
demonstration this repository does provide (a directionally-consistent,
toy-scale, 1-seed illustration that the attack → dual-bias → policy
pathway is wired correctly — not evidence about the paper's hypothesis).

## Production Readiness Assessment

Scored honestly, not to make the repository look impressive.

| Dimension | Score /10 | Justification |
|---|---|---|
| Architecture | 7 | Clean module boundaries, environment-agnostic pipeline, a genuinely enforced isolation boundary. Held back by the missing real-environment adapter. |
| Code quality | 8 | Type-hinted, `ruff`- and `mypy`-clean, docstrings cite specific paper/report sections rather than restating code. |
| Maintainability | 7 | Small, focused modules; one dispatch point per subsystem (aggregators, attacks). Some duplication across the four `local_demo_*.yaml` configs. |
| Scalability | 3 | Untested beyond a 6-agent toy environment on one CPU core; no distributed execution; MuJoCo throughput unmeasured. |
| Reliability | 6 | 111 automated tests including the trickiest ones (bitwise checkpoint-restore, oracle isolation); no experience running for the durations a real Stage-2 pilot needs (hours, with Colab session interruption). |
| Security | 6 | No secrets, no unsafe deserialization identified; the oracle isolation boundary is a within-process capability system, not a hardened production boundary — documented as such. |
| Reproducibility | 8 for the software pipeline / not applicable for the paper's results — see above. |
| Documentation | 8 | Every doc in this repository describes what the code actually does; no aspirational features documented. |
| **Overall** | **6 / 10** | A well-built, honestly-scoped research harness whose main outstanding item (the environment adapter) is clearly identified and does not require redesigning anything already built. |
