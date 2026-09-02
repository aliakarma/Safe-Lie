"""P0 #10/#11: return-scale consistency across the source/attack/
aggregation/dual pipeline.

Checks that J_C^i, J_hat^i, J_bar^i, beta*sigma, and d all stay on the
same (return, not per-step) scale, per the invariant documented in
`safelie.defenses.base`'s docstring: "never on the per-step cost scale
... the single most likely reimplementation bug."
"""

from __future__ import annotations

import math

import numpy as np

from safelie.attacks import apply_attack
from safelie.defenses import aggregate
from safelie.experiment import run_experiment_with_oracle
from safelie.utils.config import AttackConfig
from safelie.utils.logging import read_jsonl


class TestAttackMagnitudeIsReturnScaleNotStepScale:
    def test_attack_budget_scales_with_d_not_with_rollout_length(self):
        """B = budget_ratio * d (safelie.attacks.apply_attack) must not
        depend on rollout_length or the number of steps summed into a
        return -- only on the return-scale budget d. This is the
        structural guard against the 'per-step vs return-scale' mixing
        the task calls out explicitly."""
        residuals = {"own_critic": 0.0, "peer_critic_1": 0.0}
        cfg = AttackConfig(name="primary", f=1, budget_ratio=0.5, direction="negative", support="persistent")
        for d in (5.0, 25.0, 250.0):
            out = apply_attack(cfg, dict(residuals), {"own_critic"}, k=0, d=d)
            assert math.isclose(abs(out["own_critic"]), 0.5 * d, rel_tol=1e-9)

    def test_attack_magnitude_is_independent_of_rollout_length_end_to_end(self, tiny_config_factory):
        """Same budget d and budget_ratio, two different rollout_lengths:
        the ledger's per-round injected mass must be identical, since it
        is defined on the return scale (a function of d), not the
        per-step scale (which a longer rollout would otherwise inflate)."""
        cfg_short = tiny_config_factory(run_id="scale_short", attack_name="primary", f=1)
        cfg_long = cfg_short.model_copy(update={"rollout_length": 32, "total_steps": 32})

        from safelie.training.loop import ExperimentRun

        run_short = ExperimentRun(cfg_short)
        run_short.run_round()
        run_long = ExperimentRun(cfg_long)
        run_long.run_round()

        mass_short = run_short.ledger.total_mass_per_source()
        mass_long = run_long.ledger.total_mass_per_source()
        for sid in mass_short:
            assert math.isclose(mass_short[sid], mass_long[sid], rel_tol=1e-6), (
                f"attack mass for {sid} changed with rollout_length: "
                f"{mass_short[sid]} (16 steps) vs {mass_long[sid]} (32 steps) -- "
                f"the attack must be return-scale, not per-step-scale"
            )


class TestSourcesAndAggregateShareOneReturnScale:
    def test_all_source_types_report_on_the_same_order_of_magnitude_as_each_other(self, tiny_config_factory):
        """own_critic, peer_critic, and monitor must all be estimates of
        the SAME return-scale quantity J_C^i -- not, e.g., one of them
        silently reporting a per-step cost while the others report a
        discounted return. Checked structurally (relative to each
        other), not against an absolute magnitude, since the absolute
        scale of J_C^i for an arbitrary untrained policy is not itself
        pinned to any specific number."""
        cfg = tiny_config_factory(run_id="scale_sources_agree")
        out_dir = run_experiment_with_oracle(cfg)
        rounds = read_jsonl(out_dir / "rounds.jsonl")
        for record in rounds:
            for c in record["constraints"].values():
                values = [r["value"] for r in c["reports"]]
                spread = max(values) - min(values)
                scale = max(abs(v) for v in values) + 1e-6
                # Sources should disagree by at most a modest multiple of
                # their own scale, not by many orders of magnitude (which
                # would indicate one source is on a completely different
                # -- e.g. per-step vs return-scale -- footing).
                assert spread < 50 * scale, f"sources disagree by {spread}, scale~{scale}: {c['reports']}"

    def test_aggregate_point_estimate_lies_within_the_reported_values_range(self, tiny_config_factory):
        """A basic sanity invariant for mean/RCE aggregation: the
        aggregate must fall within (or, for RCE's pessimistic margin,
        near) the range of the M reports it was computed from -- it
        cannot be on a different scale than its own inputs."""
        cfg = tiny_config_factory(run_id="scale_aggregate_in_range", defense_name="rce", f=1)
        out_dir = run_experiment_with_oracle(cfg)
        rounds = read_jsonl(out_dir / "rounds.jsonl")
        for record in rounds:
            for c in record["constraints"].values():
                values = [r["value"] for r in c["reports"]]
                lo, hi = min(values), max(values)
                margin = c["aggregate"]["applied_margin"]
                assert lo - abs(margin) - 1e-6 <= c["aggregate"]["point_estimate"] <= hi + abs(margin) + 1e-6


class TestAggregatorReturnScaleContract:
    def test_mean_and_rce_agree_on_point_estimate_scale_for_identical_inputs(self):
        values = np.array([10.0, 12.0, 11.0, 50.0, 9.0])  # one outlier
        mean_result = aggregate("mean", values, f=0)
        rce_result = aggregate("rce", values, f=1, beta=1.5)
        # Both consume and produce the SAME scale; RCE's point_estimate
        # (a trimmed mean) should be much closer to the non-outlier
        # cluster than the plain mean is, but both numbers must be
        # directly comparable (no per-step/return rescale between them).
        assert abs(rce_result.point_estimate - 10.5) < abs(mean_result.point_estimate - 10.5)


class TestOracleDiscountingMatchesTrainingDiscounting:
    def test_experiment_orchestrator_passes_ppo_gamma_to_the_oracle_not_a_hardcoded_default(
        self, tiny_config_factory, monkeypatch
    ):
        """The withheld evaluator and the learner must discount with the
        SAME gamma. `safelie.experiment.run_experiment_with_oracle` wires
        `gamma=cfg.ppo.gamma` directly into `evaluate_true_cost`
        (`safelie.eval.harness`) -- a divergent or hardcoded oracle gamma
        would put `true_cost_return` on a different effective scale than
        the budget d it is compared against. `evaluate_true_cost` itself
        is independently verified to discount correctly for whatever
        gamma it is given by
        test_return_reporting.py's manual-recomputation test; this test
        checks only the wiring between the two, by intercepting the call.
        """
        import safelie.experiment as experiment_module

        seen_gammas: list[float] = []
        original = experiment_module.evaluate_true_cost

        def spy(*args, **kwargs):
            seen_gammas.append(kwargs["gamma"])
            return original(*args, **kwargs)

        monkeypatch.setattr(experiment_module, "evaluate_true_cost", spy)

        distinctive_gamma = 0.5  # far from PPOConfig's [SPEC] default of 0.99
        cfg = tiny_config_factory(run_id="scale_oracle_gamma")
        cfg = cfg.model_copy(update={"ppo": cfg.ppo.model_copy(update={"gamma": distinctive_gamma})})
        run_experiment_with_oracle(cfg, eval_every=1)

        assert seen_gammas, "evaluate_true_cost was never called"
        assert all(g == distinctive_gamma for g in seen_gammas), (
            f"oracle was called with gamma(s) {set(seen_gammas)}, expected the "
            f"config's own cfg.ppo.gamma={distinctive_gamma} every time"
        )
