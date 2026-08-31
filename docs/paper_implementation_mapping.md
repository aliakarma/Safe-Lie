# Paper → Repository Mapping

Maps `main_iclr.tex` (primary source for methodology, math, and claims)
and `PROJECT_REPORT.md` (primary engineering specification) to this
repository, component by component, with an honest implementation
status for each.

Status legend: **Full** (implemented and tested against the paper's
specification) · **Partial** (implemented with a documented `[DECISION]`
or approximation) · **Not implemented** (documented gap, with what's
needed to close it).

## Formalization (paper §3, report §6)

| Paper concept | Repository | Status |
|---|---|---|
| Constrained Markov game (Def. 1) | `safelie.envs.dual_cost.DualCostEnvWrapper` | **Full** — contract implemented; `safelie.envs.synthetic` is a from-scratch instance |
| Decentralized primal-dual update (Eq. 2) | `safelie.training.dual.dual_update`, `safelie.training.ppo.ppo_lagrangian_update` | **Full** — unconditional, projected, tested (S9, S23) |
| Corrupted cost feedback (Def. 2) | `safelie.attacks.static.static_attack` | **Full** — injection point is exactly "after critic evaluation, before consensus," verified by `tests/property/test_attacks.py` |
| Four-axis corruption taxonomy (Table 2) | `safelie.attacks.taxonomy.TaxonomyPoint` | **Full** — all four axes are independent config fields |

## Theory (paper §4, report §6.4-6.8)

| Paper result | Repository | Status |
|---|---|---|
| Theorem 1 (corruption mass conservation) | `safelie.theory.mass_conservation` | **Full** — verified to 1e-10 across 8 topologies × 4 delta schedules, `tests/theory/test_mass_conservation.py` |
| Proposition 1 (spreading/stealth) | `safelie.theory.spreading` | **Full** — uniformity, residual bound, and the stealth (median-deviation) claim all verified numerically, `tests/theory/test_spreading.py` |
| Proposition 2 (multiplicative amplification) | Not separately implemented as a standalone module | **Not implemented as code** — this is an algebraic identity about gradients, not a runtime-checkable property with the current toy policy; `safelie.training.ppo`'s combined-advantage construction (`adv_R - lambda * adv_C`) is the mechanism the proposition describes, but no test isolates the amplification factor `‖∇_θ J_C‖` the way Tests A/B isolate Theorem 1/Prop. 1. Logging `‖∇_θ J_C‖` and correlating it with the detection gap (the report's own suggested secondary test, §6.6) is a documented future extension. |
| The closed-loop diagnostic (weakness W1, new in report Revision 2) | `safelie.theory.closed_loop` | **Full** — implemented and run; measured `rho ≈ 0.001` at moderate heterogeneous feedback gains on a 6-agent ring/complete/star comparison (see `notebooks/theory_validation.ipynb`'s output), i.e. invariance appears robust at this scale — reported as a measurement, not a proof, exactly as the report specifies |

## Defense: Robust Constraint Estimation (paper §5, report §7.1, Alg. 1)

| Paper element | Repository | Status |
|---|---|---|
| TrimMean_f (Alg. 1 line 5) | `safelie.defenses.trimmean.trimmed_mean_aggregator` | **Full**, with the degenerate-case fix the report recommends: raises on `M <= 2f` rather than the paper's undefined behavior (W2) |
| MAD margin (Alg. 1 lines 6-7) | `safelie.defenses.rce.rce_aggregate` | **Full**, with the degenerate-`|T|` fix: floors and warns rather than silently returning `spread=0.0` (W2, S8) |
| Unconditional dual update (Alg. 1 line 8) | `safelie.training.dual.dual_update` | **Full** — no branch of any kind on the residual magnitude, verified S9/S23 |
| Reliability weights (Alg. 1 lines 2, 10) | `DefenseConfig.use_reliability_weights` (flag only) | **Not implemented** — `[GAP]` G1: the paper declares these but no line of its own pseudocode reads them. Shipped off by default (decision D12) rather than inventing an unspecified update rule |
| Theorem 2 (bounded violation) | `safelie.eval.margin` | **Partial** — the *estimator* (RCE) is fully implemented and tested; the *guarantee*'s precondition (`beta*sigma >= epsilon(M,f,alpha)`) is not runtime-checkable (the paper's own weakness W3), so `guarantee_in_force` is logged as an estimate against an offline-calibrated reference, never as a proof |
| Proposition 3 (liveness under over-reporting) | `safelie.training.dual.dual_update` + `tests/property/test_dual_update.py` | **Full** — verified directly: 1000 rounds of 1e6-magnitude over-reporting, the dual update executes every round and saturates at `lambda_max` rather than deadlocking |

## Aggregator zoo (Table 4)

| Aggregator | Repository | Status |
|---|---|---|
| Mean | `safelie.defenses.mean` | **Full** |
| Coordinate median | `safelie.defenses.median` | **Full** |
| Krum | `safelie.defenses.krum` | **Full** (scalar adaptation; cited to Blanchard et al. as the paper does, not claimed as novel) |
| TrimMean | `safelie.defenses.trimmean` | **Full** |
| TrimMean + margin (RCE) | `safelie.defenses.rce` | **Full** |

## Source layer / independence accounting (report §R10.3, weakness W4)

| Element | Repository | Status |
|---|---|---|
| M reporting sources per constraint | `safelie.sources.registry.SourceRegistry` | **Full** |
| `effective_M` over independence classes | `safelie.sources.registry.effective_M` | **Full**, and enforced at config-validation time (S12), not merely computed and ignored |
| Ensemble/monitor diversification (`[GAP]` G5) | `safelie.sources.estimators.DiversifiedReplica` | **Full** — independent init + bootstrap-resampled minibatches, refit every round |
| Peer critic observability (`[GAP]` G4) | `safelie.training.loop.ExperimentRun._collect_source_value` | **Partial** — resolved per the report's recommended option (1): a peer's critic is evaluated on the *constraint owner's own observation*, restricting to state any agent's network can read. On the synthetic environment this is a reasonable stand-in; a real MuJoCo adapter would need to state an explicit observability assumption for its own state space |

## Attack module (paper App. B, report §7.2)

| Attack | Repository | Status |
|---|---|---|
| Primary (negative, persistent, static, consistent) | `safelie.attacks.static.static_attack` | **Full**, verified end-to-end |
| Over-reporting (positive direction) | `safelie.attacks.static.static_attack(direction="positive")` | **Full** |
| Stealth (negative, selective, adaptive, consistent) | `safelie.attacks.adaptive.stealth_attack` | **Implemented, not wired into the default training loop** — `kappa`/`nu` have no paper-specified values (`[GAP]` G7); compact-study scope (decision D5) defers this to Stage 3 |
| Byzantine (equivocating) consistency axis | `safelie.attacks.byzantine.byzantine_attack` | **Implemented, not wired in** — the source layer supports per-recipient values as the report recommends designing for, but the training loop does not exercise it (decision matching `[GAP]` G15, "not exercised in the pilot") |
| Benign control (falsification test) | `safelie.attacks.static.benign_control` | **Full**, verified unbiased (S20) |
| Ground-truth ledger | `safelie.attacks.ledger.AttackLedger` | **Full**, verified to reconstruct injected mass exactly (S19) |

## Environments (paper §5.1)

| Environment | Repository | Status |
|---|---|---|
| `ManyAgent Ant` (N=6) | `safelie.envs.mamujoco` | **Not implemented** — requires MuJoCo + a Multi-Agent MuJoCo factorization package, assigned by the report to the Colab GPU stage. See that module's docstring for the completion checklist |
| `HalfCheetah` 2×3 (N=2) | — | **Not implemented**, same reason |
| Safety-Gymnasium multi-agent navigation | — | **Not implemented**, same reason |
| Synthetic CPU stand-in | `safelie.envs.synthetic.SyntheticConstrainedMarlEnv` | **Full** — not a paper environment; a `[DECISION]`-classified approximation used solely for local verification, clearly labeled throughout |

## Algorithms / baselines (report Phase 3)

| Baseline | Repository | Status |
|---|---|---|
| MAPPO-Lagrangian | `safelie.algos.networks`, `safelie.training.ppo` | **Full** — the compact study's designated victim (decision, §R2.1) |
| MACPO | — | **Not implemented** — Stage-3 scope (§R9 item 10) |
| Dec-PDO | — | **Not implemented** — Stage-3 scope |
| PID-Lagrangian | — | **Not implemented** — Stage-3 scope |
| Unconstrained MAPPO | — | **Not implemented** — Stage-3 scope; trivially derivable by setting `lambda_max=0` or `dual.eta_lambda=0` on the existing implementation if needed sooner |

## Evaluation protocol (paper §5.1, report §8)

| Element | Repository | Status |
|---|---|---|
| Task return, reported cost, true cost, violation rate, peak violation, detection gap | `safelie.eval.metrics` | **Full** |
| Withheld oracle, isolated from the learner | `safelie.eval.oracle`, `safelie.envs.guards` | **Full**, isolation verified structurally and by grep-level test (S10) |
| Pre-registered success criterion | `safelie.eval.protocol.attack_succeeded` | **Full** |
| Welch's t-test + Holm correction | `safelie.analysis.stats` | **Full**, gated to ≥5 seeds per decision D6 |
| 3-seed pilot reporting (sign/ordering) | `safelie.eval.protocol.pilot_seed_summary`, `consistent_across_seeds` | **Full** |
| Benign control (falsification test) | `safelie.attacks.static.benign_control` + `configs/experiment/local_demo_benign.yaml` | **Full**; run once locally, see [reproducibility.md](reproducibility.md) |

## Governance (report Phase 10)

| Element | Repository | Status |
|---|---|---|
| `SourceAuditor` (M≥2f+1 checklist over independence classes) | `safelie.governance.auditor` | **Full** — verified to reject the paper's own N=2, M=5 configuration at f=2 (Phase 10's own exit criterion) |
| Source-disagreement / multiplier-saturation alarms | `guarantee_in_force` logging in `RceResult` | **Partial** — the underlying signal is logged; a standalone alarm/dashboard is not implemented (out of scope for a CLI-driven research repo) |

## What was never attempted (out of scope by the report's own plan)

- The full 300+ run, 10⁷-step grid (report §10.2) — reclassified by the
  report itself as Stage 3, expanded only after a Stage-2 pilot.
- Any Stage-2 or Stage-3 experiment — blocked on the MuJoCo adapter.
- Hardware-in-the-loop or any physical deployment — motivational only in
  the paper, never in scope for a software repository.
