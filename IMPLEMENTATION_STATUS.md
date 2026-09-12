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
- **PID-Lagrangian** (`safelie.training.dual.pid_dual_update`): the
  Stooke et al. (2020) multiplier controller, selected with
  `dual.controller: pid`. Consensus mixes the *integral* term, which is
  what makes `k_p = k_d = 0, k_i = eta_lambda` reduce to Eq. 2 bit for bit
  (pinned against `dual_update` itself in
  `tests/unit/test_pid_dual.py`) and what preserves Theorem 1's
  mass-conservation identity on the integral path. The derivative is taken
  on the cost and one-sided, as in the source. **The gains are
  deliberately undeclared**: they have no defaults and are required on the
  `pid` path, because no pre-declaration has fixed an operating point and
  a gain triple must always be someone's stated choice rather than a
  schema fallback.
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
| Environment | `DualCostEnvWrapper` contract, full isolation machinery, a synthetic CPU environment, **and the Safe MAMuJoCo adapter** (`safelie.envs.mamujoco`, two backends) | The paper's primary environment, ManyAgent Ant, does not exist in the reference Safe MAMuJoCo at all, so its cost function is this repository's, not the paper's; its velocity threshold is calibrated rather than specified | See `safelie/envs/mamujoco.py`'s docstring, Deviations 1-3, and `docs/assumptions.md` |
| Peer critic observability (`[GAP]` G4) | Resolved per the report's recommended reading: peer critics evaluate the constraint owner's own observation. On Safe MAMuJoCo, `cost_mode='per_agent_velocity'` derives each agent's cost from its own torso segment's speed, read from shared simulator state — measurable by a peer in principle | The reference implementation makes the question vacuous by giving every agent an identical cost; only the per-agent variant exercises G4 meaningfully | Recorded as a `[DECISION]` in `docs/assumptions.md`, not as a resolution of the paper's ambiguity |
| RCE's Theorem 2 guarantee | The estimator is fully implemented and tested | The probabilistic *guarantee*'s precondition (`beta*sigma >= epsilon(M,f,alpha)`) is not runtime-checkable — this is the paper's own acknowledged weakness (W3), not a gap in this implementation | `epsilon` depends on the unknown sub-Gaussian parameter of honest sources |
| Proposition 2 (multiplicative amplification) | The mechanism it describes (combined advantage `adv_R - lambda*adv_C`) is implemented and drives real training | No standalone test isolates the amplification factor `‖∇_θ J_C‖` the way Theorem 1/Prop. 1 have dedicated numerical tests | Would require logging and correlating a gradient-norm quantity during actual RL training; the report names this as a suggested future test, not a required one |

## Not Implemented

| Component | Why | What's needed |
|---|---|---|
| Safety-Gymnasium multi-agent *navigation* tasks | Goal-conditioned, with a different agent/observation structure than a MuJoCo factorization | A separate adapter; the Safe MAMuJoCo one does not generalize to them |
| MACPO, Dec-PDO, unconstrained MAPPO baselines | Report's own compact-study scope (§R2.1) designates MAPPO-Lagrangian as the sole Stage-2 victim; the rest are Stage-3 | Port from a reference implementation once Stage-2 is running |
| Reliability weights (Algorithm 1 lines 2, 10) | The paper declares and initializes them but no line of its own pseudocode reads them (`[GAP]` G1) | A specified update rule from the paper's authors, or a documented invented one — deliberately not fabricated here |
| Adaptive (stealth) and Byzantine attack axes in the default training loop | Both are implemented as standalone functions (`safelie.attacks.adaptive`, `safelie.attacks.byzantine`) but not wired into `safelie.training.loop`; the compact study's scope (decisions D5, and `[GAP]` G15) explicitly defers both to Stage 3 | Wire `stealth_attack`/`byzantine_attack` into `safelie.attacks.apply_attack`'s dispatch and `ExperimentRun.run_round` |
| Cumulative corruption budget (`Delta`, §3.2) enforcement | Declared in the paper, never used in its own evaluation protocol (`[GAP]` G14) | `AttackLedger.total_mass()` already tracks it; enforcing it as a constraint would be a small addition |
| Stage-3 full-scale grid | 300+ runs at 10⁷ steps; 75-300 GPU-days by the report's own corrected estimate (§10.2), and this workload is CPU-bound, which makes it worse | Selective expansion of whatever a completed Stage-2 pilot justifies (§R9) |

## Technical Debt

- **The workload is CPU-bound and no part of it uses the GPU.** Nothing
  in `safelie` moves a tensor to CUDA — the only CUDA reference is
  `torch.cuda.manual_seed_all` in `safelie.utils.seeding`. The networks
  are small MLPs stepped one observation at a time inside a Python loop,
  so measured throughput is ~130 env-steps/s regardless of accelerator.
  The report's Colab-T4 framing (§R7.1) does not match what was built;
  a high-CPU runtime is the right target. Batching the per-agent forward
  passes would give perhaps 2-3x, at the cost of the bitwise-determinism
  guarantees in `tests/smoke/test_determinism.py`.
- **Both environments required cost-scale calibration** to make the
  constraint bind — `d=25` → `d=5` for the synthetic local demos, and a
  velocity threshold of 0.75 rather than Safe MAMuJoCo's 2.418 for
  ManySegmentAnt. Note that an initial-policy calibration is necessary but
  not sufficient: 1.0 cleared that bar yet still left `lambda` pinned at
  zero until round ~159 of 250, because what the dual update compares
  against the budget is the learner's estimate, whose convergence time
  (~150 rounds) is 60% of a pilot-scale run. `scripts/calibrate_cost.py` now automates the check.
  Note its two numbers: the true discounted cost can be well above the
  budget while the learner's own estimate is still far below it, and it
  is the estimate that drives `lambda`.
- **The logged `detection_gap` is measured against the wrong quantity.**
  `safelie.experiment` computes it against `reported_cost_return`, which
  `safelie.training.loop` sets to the agent's own cost-critic estimate --
  a quantity the attack never touches, since corruption lands in
  `aggregate`. Measured against the aggregate instead, an attacked seed
  moves +3.2 sd from clean rather than +0.2 sd. `scripts/analyze_matrix.py`
  recomputes the corrected metric from `aggregate.point_estimate`, already
  present in every log, so no re-running was needed. RCE's effective
  estimate is likewise recoverable (`point_estimate + beta * spread`,
  since the logged `spread` is the post-flooring value used for the
  margin), so the corrected metric is exact for every condition. Logging
  `applied_margin` and `pessimistic_estimate` directly would still be
  tidier than reconstructing them. See
  [docs/evaluation.md](docs/evaluation.md).
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

1. Run the compact pilot matrix (5 conditions × seeds [0,1,2]) and read
   condition D (the falsification control) before condition C, per §R8.3.
   Budget ~2 h per run on CPU.
2. Decide whether `manyagent_ant` (this repository's cost function) or
   `halfcheetah_6x1` (the reference cost function, also N=6) is the
   environment the paper should report. They are not interchangeable, and
   the choice is a claim about faithfulness, not a configuration detail.
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
