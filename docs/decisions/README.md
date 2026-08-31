# Decision log (D1-D14)

`PROJECT_REPORT.md` §R5.2 requires every specification gap it lists as
blocking (G1-G20) to have a recorded decision here before the Colab pilot
proceeds, encoded in config validation, not merely documented. This is
that log, cross-referenced to the report's own D1-D14 numbering
(Appendix B, "Revision 2 — decisions taken"). See
[../assumptions.md](../assumptions.md) for the full rationale behind each
— this file is the compact index the report's format expects.

| ID | Decision | Enforced by |
|---|---|---|
| D1 | Environment `ManyAgent Ant`, N=6, d=25; victim MAPPO-Lagrangian only | `configs/experiment/pilot_*.yaml` (Stage-2 spec; not runnable — see `safelie/envs/mamujoco.py`) |
| D2 | Aggregators: `mean` and `rce` only, for the compact study | `configs/experiment/pilot_*.yaml`; `safelie.defenses` implements all 5 for Stage-3 ablations |
| D3 | `M <= 2f` raises at config-validation time; `|T| < 3` floors the margin, warns, and sets `guarantee_in_force=False` | `safelie.utils.config.ExperimentConfig._cross_field_checks`; `safelie.defenses.rce.rce_aggregate` |
| D4 | Topology fixed to `ring` for the whole compact matrix; sigma_2(W) available via `safelie.consensus.mixing.second_largest_singular_value` | `configs/experiment/pilot_*.yaml` |
| D5 | Attack: persistent static under-reporting at B/d=0.5; adaptive attack deferred | `configs/experiment/pilot_B_attack.yaml`; `safelie.attacks.adaptive` implemented but not wired into the default loop |
| D6 | Seeds `[0,1,2]`; results reported per-seed by sign and ordering, never a t-test at n=3 | `safelie.eval.protocol.pilot_seed_summary`; `safelie.analysis.stats.MIN_SEEDS_FOR_INFERENCE` raises below 5 |
| D7 | 5e5 steps per run; the §R6.1 binding-constraint precondition must be checked on the clean condition before others are interpreted | `configs/experiment/pilot_*.yaml`; documented in `docs/reproducibility.md` (and demonstrated as a real confound on the local synthetic demo before the budget was recalibrated — see `docs/assumptions.md`) |
| D8 | Multiplier channel assumed trusted/authenticated; multiplier corruption is out of the Stage-2 threat model | `safelie.attacks` only ever corrupts the cost-report residual, never `W @ lambda` directly |
| D9 | Margin reported as three separate quantities (`beta*sigma`, `epsilon_offline`, `guarantee_in_force`); no runtime safety guarantee is claimed | `safelie.eval.margin` |
| D10 | `effective_M` counts independence classes; configs satisfying M>=2f+1 only via same-class replicas are rejected | `safelie.sources.registry.effective_M`; `safelie.utils.config.ExperimentConfig`; `safelie.governance.auditor` |
| D11 | Theorem 1 stated as an open-loop result; closed-loop transfer measured, never assumed | `safelie.theory.closed_loop.run_closed_loop_diagnostic`, run in `notebooks/theory_validation.ipynb` |
| D12 | Reliability weights shipped disabled; not used in the pilot | `DefenseConfig.use_reliability_weights` defaults to `False`; no update rule implemented |
| D13 | Notebooks contain orchestration only; CI lint fails any notebook cell defining a `def` or `class` | `scripts/lint_notebooks.py`, run in `.github/workflows/ci.yml` |
| D14 | PPO/network hyperparameters pinned to one documented set, identical across all conditions | `safelie.utils.config.PPOConfig` |
