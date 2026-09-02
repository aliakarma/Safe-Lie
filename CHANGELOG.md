# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### G1: the dual update's constraint-objective estimator (post-G0)

G0 was a CONDITIONAL PASS. The learner learns, the dual responds, true
cost moves toward the budget, three seeds reproduce. One finding blocked
a full pass, and this batch repairs it. Nothing here runs or reinterprets
an attack experiment.

**The defect.** The value the dual update compared against the budget `d`
was `ret_c[0]`, the GAE(lambda) bootstrap target at a round's first step.
That is a valid cost-critic regression target and a valid input to
advantage estimation. It is not a valid direct estimator of
`J_C^i(theta) = E[sum_t gamma^t C_t^i]`: a lambda-return is a geometric
blend of n-step returns whose weight on genuine sampled cost decays as
`(gamma*lambda)^n`, so at gamma=0.99, lambda=0.95 it keeps only about
`1/(1 - gamma*lambda) = 16.8` steps of real Monte-Carlo evidence and
hands the rest of the roughly 100 steps of discounted mass to the cost
critic's bootstrap, inheriting whatever bias the critic carries. The
three G0 clean runs measured exactly that: whole-run own-critic bias of
-6.91 / -8.75 / -8.53 against the withheld oracle's true discounted cost
return, and `corr(estimate, true)` of only 0.50 / 0.48 / 0.43 while an
undiscounted per-step cost rate from the learner's own rollout already
reached 0.84 / 0.86 / 0.82 against the identical oracle series.

**The repair** is a separation of concerns, not a replacement of GAE:

- New `safelie.training.constraint_return`. `discounted_window_return`
  computes `sum_t gamma^t C_t` over the round's window on one global
  discount clock -- the same functional
  `safelie.eval.oracle.OracleEvaluator` accumulates, including the
  requirement that the clock does not reset at an internal auto-reset
  boundary. No critic anywhere in the path. A unit test asserts the two
  accumulations agree to `rel_tol=1e-12`, which is what makes "bias
  against the oracle" a measurement of estimator error rather than of a
  definitional mismatch.
- `discounted_cost_to_go` supplies proper Monte-Carlo regression targets
  for the `ensemble_replica` / `monitor` sources, replacing `ret_c`, and
  `complete_target_count` masks the trailing rows whose target is
  censored by the window edge by more than 1% of its own discounted mass
  (459 rows of 2000 at gamma=0.99). Without that mask a head fit on
  systematically shrunk tail targets predicts low at the query point,
  reintroducing the same downward bias through the regression.
- `episodic_mc_returns` computes the textbook per-episode own-clock
  estimator as a logged **diagnostic**, so the size of the definitional
  difference between the two readings of `sum_t gamma^t C_t` is measured
  rather than assumed. It is not the dual signal: complete-episode
  averaging censors episodes still running at the window edge, and
  excluding them biases the mean toward short episodes.
- `ExperimentConfig.constraint_estimator` selects `mc_window` (the
  default) or `gae_lambda` (the pre-G1 behaviour, retained only so the G0
  artifacts stay reproducible and the two can be compared head to head).
  It is recorded in every run's `run_metadata.json` config snapshot, so
  no artifact is ambiguous about which estimator produced it.

**Explicitly unchanged.** GAE keeps gamma=0.99 and lambda=0.95 for policy
optimisation; `adv_r`, `adv_c` and both critics' regression targets still
come from `safelie.training.gae`, and `safelie.training.ppo` is untouched.
The four `peer_critic` sources remain learned-value-function estimators,
because the cost critic's own target is GAE(lambda). The threat model,
RCE, attack magnitude, the budget d=25 and the statistical methodology
are untouched: the attacker still corrupts the communicated source
residual after estimation and before consensus, and the oracle remains
withheld. `tests/unit/test_dual_estimator_wiring.py` asserts both halves
of that -- the honest pre-attack reports are unchanged when the attack is
enabled, and the value the dual consumes is not.

**Logging.** Both estimators are written every round from the same
rollout, whichever is active (`rounds.jsonl:
constraints.*.constraint_estimators`), and paired against the same oracle
episode with their biases (`oracle.jsonl:
agents.*.constraint_estimators`), so the head-to-head comparison comes
from one run's own logs rather than from two campaigns that also differ
in their trajectories.

`docs/g1_gates.md` pre-declares the G1 acceptance gates, with every
numeric bar justified against a G0 measurement that already existed on
disk at declaration time. `scripts/analyze_g1.py` applies them.


### P0 implementation repair (pre-G0)

Independent audit found the learner/source layer was not yet operating in
the regime the paper's algorithm assumes. This batch fixes the underlying
substrate; it does not run or reinterpret any experiment. See
`docs/assumptions.md`'s "P0 implementation repair" section for the full
before/after evidence (including numbers from a real, previously-executed
pilot run that predates this batch and must not be used for the paper).

**Bug fixes** (implementation errors; the paper's algorithm is unchanged):

- Peer-critic sources (`safelie.training.loop._collect_source_value`)
  mapped `peer_critic_<k>` to the literal agent `agent_<k>` for every
  owner. Under the pilot's own M=7 config (`peer_critic_1..4`, N=6),
  agent_1..agent_4 each received themselves as one of their four "peer"
  sources -- 4 of 6 owners, not an edge case. Now resolved owner-relatively
  as `agent[(owner_index + k) mod N]`, which is structurally incapable of
  self-collision for any offset not divisible by N; a config using a
  self-colliding offset now fails at config-validation time
  (`ExperimentConfig`), not silently at runtime.
- `guarantee_in_force` was vacuous under attack: `clean_run_disagreements`
  only accumulated `if attack.name == "none"`, so an attacked run (that
  condition false for its entire length) calibrated `epsilon_offline` from
  an empty list, which `calibrate_epsilon_offline` silently returns as
  `0.0` -- making `applied_margin >= epsilon_offline` true every round
  regardless of whether the margin was adequate. Fixed by
  `safelie.eval.calibration.run_clean_calibration`: a dedicated,
  attack-disabled rollout (`defense.calibration_rounds`, default 20) run
  once before round 0, for every RCE run (clean or attacked), with every
  component of the resulting `epsilon_offline` written to
  `<run_dir>/guarantee_calibration.json` for traceability.
- `compute_gae` treated a time-limit truncation identically to a genuine
  termination (zero bootstrap, lambda-recursion cut). Correct for
  termination; wrong for truncation, whose standard fix bootstraps from
  the critic's own value at the true final observation. `safelie.envs.
  mamujoco`'s backend truncates at ~1000 steps -- well below the pilot
  configs' `rollout_length=2000` -- so every pre-fix MaMuJoCo-backed round
  silently capped the cost critic's effective horizon partway through,
  compounding the bias below. `terminated`/`truncated` are now threaded
  through `safelie.training.buffer`/`gae` separately, and
  `MaMuJoCoDualCostEnv.step` exposes the pre-reset observation via
  `info["final_observation"]` (Gymnasium's own auto-reset convention) so
  a real bootstrap value is available. The synthetic environment's
  numbers are unaffected (its own truncation always falls on the buffer's
  last index).
- `rounds.jsonl`'s `reported_cost_return`/`task_return` (`ret_c[0]`/
  `ret_r[0]`) were used both as learner-training diagnostics AND, via
  `safelie.experiment`'s `detection_gap`, as the paper's evaluation
  metric -- despite being a GAE(lambda) bootstrap TARGET on the training
  rollout's pre-update policy, not a Monte-Carlo return on the same
  policy/episode the oracle evaluates. `safelie.eval.harness.
  evaluate_true_cost` now also computes `episodic_task_return` and
  `episodic_reported_cost_return` as proper discounted sums over its own
  withheld rollout, alongside the already-correct `true_cost_return`; all
  three, plus `violation_rate`/`peak_violation`, are the paper's
  evaluation quantities, logged only in `oracle.jsonl`.
  `mechanism_reported_cost_return` (`rounds.jsonl`, = the aggregate
  value that actually drove that round's dual update) and
  `detection_gap_vs_aggregate` (`oracle.jsonl`) replace the informal
  post-hoc reconstruction `scripts/analyze_matrix.py` previously needed;
  it now reads both fields directly.

**Methodological corrections** (bringing the implementation in line with
the stated algorithm; no change to what is being estimated):

- `AgentBundle` (`safelie.algos.networks`) trained the policy, reward
  critic, and cost critic through ONE shared Adam optimizer and ONE
  joint `clip_grad_norm_` over all three networks' concatenated
  parameters. A single shared clip means one network's gradient norm sets
  the scaling factor applied to every network's update, including the
  policy's -- a coupling with no basis in Eq. 2, where the three updates
  are independent. Each network now has its own optimizer and its own
  clip (`safelie.training.ppo`); `PPOConfig.critic_lr` (default: same as
  `lr`) exposes a separate critic learning rate without inventing a
  second `[SPEC]` number.
- No observation or return normalization existed anywhere in the
  pipeline. `safelie.algos.normalization.RunningMeanStd` (Chan et al.
  parallel-variance, as in OpenAI Baselines/SB3) now normalizes every
  network's input observation and the two critics' regression targets;
  `AgentBundle.value`/`.cost_value` denormalize back to return scale
  before returning, so every caller outside the training loss itself
  (`safelie.training.loop`, the peer-critic query in `safelie.sources.
  estimators`, `scripts/calibrate_cost.py`) keeps receiving exactly the
  same kind of return-scale quantity as before. GAE's own inputs/outputs
  are unchanged (still return-scale, still `[SPEC]` gamma/lambda).
- `PPOConfig.entropy_coef` (a `[GAP]`, decision-D14-pinned default, not
  `[SPEC]`) revised from 0.0 to 0.001: a small, standard continuous-
  control exploration safeguard, not tuned against this repository's own
  results.

**New diagnostics** (measurement tooling; assert nothing about the paper
on their own):

- `safelie.governance.empirical_diversity` / `scripts/
  audit_source_independence.py`: measures per-source bias/MAE/RMSE/
  correlation against the withheld oracle's true cost, the cross-source
  error-correlation matrix, and a participation-ratio "statistical
  effective M" -- distinct from, and never conflated with, `safelie.
  sources.registry.effective_M`'s declared-independence-class count.
  Warns and forces `assumption_1ii_supported=False` below 30 rounds
  (a Pearson correlation is not informative at very small n in either
  direction).
- `tests/unit/test_return_scale_consistency.py`: return-scale invariants
  across the source/attack/aggregation/dual pipeline (attack magnitude
  independent of `rollout_length`, sources mutually commensurate,
  aggregate within its inputs' range, oracle gamma matches `cfg.ppo.gamma`).

### Added

- `safelie.envs.mamujoco`: the Safe MAMuJoCo adapter, previously a
  documented gap. Two backends satisfy `DualCostEnvWrapper` --
  `safety_gymnasium` (the reference Safe MAMuJoCo of Gu et al.) and
  `gymnasium_robotics` (portable, no native cost signal). They cannot be
  installed together: `safety-gymnasium==1.0.0` pins `gymnasium==0.28.1`,
  `gymnasium-robotics==1.2.2` and `mujoco==2.3.3`. Backend selection is an
  install-time fact, recorded per run in `DualCostStep.info["backend"]`.
- `env.name` accepts `halfcheetah_6x1` and `ant_4x2` alongside
  `manyagent_ant` and `halfcheetah_2x3`. `halfcheetah_6x1` is the only
  genuinely N=6 configuration the reference implementation supports.
- `EnvConfig.cost_mode`, `EnvConfig.velocity_threshold`,
  `EnvConfig.agent_obsk` for the MuJoCo backends.
- `scripts/calibrate_cost.py`: measures whether a config's cost constraint
  actually binds before a run is launched, reporting both the true
  discounted cost and the learner's own estimate (only the latter drives
  `lambda`). Guards the §R6.1 precondition failure.
- `safelie[mujoco]` optional dependency group.
- `scripts/run_matrix.py`: priority-ordered job queue for the condition
  matrix. Emits §R8.3's reading order (falsification control before the
  defense claim) as dispatch order, so an interrupted night leaves the
  most decisive subset complete rather than an arbitrary one. Skips
  finished runs and resumes partial ones.
- `scripts/analyze_matrix.py` reports `detection_gap_vs_aggregate` as the
  primary metric, recomputed from `aggregate.point_estimate` (already in
  every log). The logged `detection_gap` is measured against the agent's
  own cost critic, which the attack never touches; against the aggregate
  the same attacked seed moves +3.2 sd from clean rather than +0.2 sd.
- `scripts/analyze_matrix.py`: matrix summary that selects its statistics
  by seed count rather than by taste -- Welch + Holm at n>=5 ([SPEC]
  §5.1), per-seed sign/ordering below that (decision D6), never working
  around `welch_t_test`'s refusal to run at n<5.
- `scripts/train.py --seed / --run-id / --threads`. `--seed` suffixes the
  run_id so each seed writes its own directory; without it a second seed
  silently resumed the first one's checkpoint. `--threads` defaults to 1,
  verified bitwise-identical to the torch default on this pipeline (the
  networks are far too small for intra-op parallelism to pay), which keeps
  results independent of the host core count and lets several runs share a
  machine.

### Fixed

- **Concurrent runs raced on a shared model-asset file.** MaMuJoCo
  generates ManySegmentAnt's model XML to a fixed filename inside its own
  package directory, loads it, then deletes it
  (`mujoco_multi._create_base_gym_env`). Two processes building the same
  scenario race on that one file and the loser reads an empty XML --
  launching four pilot runs at once killed three at startup. Not a
  startup-only hazard: `safelie.experiment` builds a fresh environment for
  every oracle evaluation episode, so a 250-round run touches that path
  250 times. `safelie.envs.mamujoco` now serializes construction with a
  cross-process lock plus a bounded retry. Regression test in
  `tests/unit/test_env_factory.py`.
- **Networks were sized from the config, not the environment.**
  `ExperimentRun.__init__` read `cfg.env.obs_dim` / `cfg.env.action_dim`,
  which default to 8 and 2 and which the `pilot_*` configs never set. On a
  real MuJoCo factorization (ManySegmentAnt 6x1: 63-dim observations,
  4-dim actions) this silently built 8-dim policies and fed them 63-dim
  observations. `ExperimentRun` now takes both from the constructed
  environment, and `DualCostEnvWrapper` declares them.

### Changed

- The `pilot_*.yaml` configs run. Their headers no longer say otherwise,
  and they carry a calibrated `velocity_threshold` with the measurement
  that justifies it.
- `safety_gym_nav` remains unimplemented, now for a stated reason
  (goal-conditioned tasks are not a MuJoCo factorization) rather than as
  part of a blanket deferral.

### Documented

- Three deviations forced by the reference implementation, recorded in
  `safelie/envs/mamujoco.py`'s docstring, `docs/assumptions.md` and
  `docs/reproducibility.md`: **ManyAgent Ant does not exist in Safe
  MAMuJoCo at all** (no threshold-table entry; its constructor asserts on
  the name), Safe MAMuJoCo's cost is **shared across agents rather than
  per-agent** (making `[GAP]` G4 vacuous), and its thresholds **do not
  bind at the pilot's scale** (~30x from binding at `d=25`).
- **The workload is CPU-bound and uses no GPU.** No tensor is moved to
  CUDA anywhere in `safelie`; measured throughput is ~130 env-steps/s,
  putting a 5x10^5-step pilot run near two hours regardless of
  accelerator. The report's Colab-T4 framing does not match what was
  built; `README.md`, `docs/setup.md` and the notebook now say so.

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
  MAMuJoCo integration was not implemented at 0.1.0 — see the
  Unreleased entry above).
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
