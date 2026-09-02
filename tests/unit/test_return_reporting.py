"""P0 #6 regression: evaluation quantities vs. learner training quantities
must be separately available, correctly defined, and never mixed.

Bug fixed: `rounds.jsonl`'s `reported_cost_return`/`task_return` are
GAE(lambda) training targets (`ret_c[0]`/`ret_r[0]`) computed on the
learner's own pre-update rollout -- not proper episodic Monte-Carlo
returns, and not computed on the same policy snapshot as the oracle's
`true_cost_return`. `safelie.experiment`'s logged `detection_gap` was
also computed against the agent's own cost-critic estimate, a quantity
the attack never touches (corruption lands in the aggregate that the dual
update actually consumes).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from safelie.algos.networks import AgentBundle
from safelie.envs.synthetic import SyntheticConstrainedMarlEnv
from safelie.eval.harness import evaluate_true_cost
from safelie.experiment import run_experiment_with_oracle
from safelie.utils.logging import read_jsonl


class TestOracleEvaluationQuantitiesAreProperMonteCarloSums:
    def test_episodic_returns_match_a_manual_discounted_recomputation(self, tmp_path):
        """Direct unit check of safelie.eval.harness.evaluate_true_cost:
        re-simulate the same deterministic (zero-action) rollout by hand
        and compare bit-for-bit against the harness's own accounting."""
        gamma = 0.9
        n_agents = 2
        agents = {
            f"agent_{i}": AgentBundle(obs_dim=6, action_dim=2, hidden_dim=8, lr=1e-3) for i in range(n_agents)
        }
        # Zero out the policy mean net so actions are deterministic
        # (~0, modulo the sampled Gaussian noise) -- instead, force
        # near-zero action by zeroing the mean net's final layer and
        # using a very small std, so the manual recomputation need only
        # replay the SAME env with the SAME actions the harness used.
        for bundle in agents.values():
            with torch.no_grad():
                bundle.policy.mean_net[-1].weight.zero_()
                bundle.policy.mean_net[-1].bias.zero_()
                bundle.policy.log_std.fill_(-20.0)  # std ~ 0

        def env_factory():
            return SyntheticConstrainedMarlEnv(n_agents=n_agents, budget=25.0, obs_dim=6, action_dim=2, horizon=10)

        result = evaluate_true_cost(
            env_factory=env_factory, agents=agents, gamma=gamma, budget=25.0, rollout_length=10, seed=123,
        )

        # Manual replay: same env, same seed, actions forced to exactly
        # zero (matching what the near-deterministic policy above samples
        # to within float precision at std~0).
        env = env_factory()
        step = env.reset(seed=123)
        expected_task_return = 0.0
        expected_reported = dict.fromkeys(env.agent_ids, 0.0)
        expected_true = dict.fromkeys(env.agent_ids, 0.0)
        for t in range(10):
            step = env.step({aid: np.zeros(2, dtype=np.float32) for aid in env.agent_ids})
            discount = gamma**t
            expected_task_return += discount * step.reward
            for aid in env.agent_ids:
                expected_reported[aid] += discount * step.reported_cost[aid]
                expected_true[aid] += discount * step.reported_cost[aid]  # faithful env: true == reported

        assert math.isclose(result.episodic_task_return, expected_task_return, rel_tol=1e-6, abs_tol=1e-9)
        for aid in env.agent_ids:
            assert math.isclose(result.episodic_reported_cost_return[aid], expected_reported[aid], rel_tol=1e-6, abs_tol=1e-9)
            assert math.isclose(result.true_cost_return[aid], expected_true[aid], rel_tol=1e-6, abs_tol=1e-9)

    def test_reported_cost_return_equals_true_cost_return_for_the_faithful_environment(self, tmp_path, tiny_config_factory):
        """Invariant this repository documents explicitly (dual_cost.py's
        docstring): the environment's `reported_cost` is faithful --
        numerically identical, per step, to its true-cost stream.
        Corruption is injected downstream, never at the sensor. If this
        ever stops holding for a given environment, `detection_gap_vs_*`
        formulas need revisiting -- this test is the tripwire."""
        cfg = tiny_config_factory(run_id="return_reporting_faithful")
        out_dir = run_experiment_with_oracle(cfg)
        oracle_records = read_jsonl(out_dir / "oracle.jsonl")
        assert oracle_records
        for record in oracle_records:
            for a in record["agents"].values():
                assert math.isclose(a["episodic_reported_cost_return"], a["true_cost_return"], rel_tol=1e-9, abs_tol=1e-9)


class TestTrainingAndEvaluationQuantitiesAreSeparatelyAvailable:
    def test_all_five_evaluation_quantities_present_and_finite(self, tiny_config_factory):
        cfg = tiny_config_factory(run_id="return_reporting_eval_quantities")
        out_dir = run_experiment_with_oracle(cfg)
        oracle_records = read_jsonl(out_dir / "oracle.jsonl")
        assert oracle_records
        for record in oracle_records:
            for a in record["agents"].values():
                assert math.isfinite(a["episodic_task_return"])
                assert math.isfinite(a["true_cost_return"])
                assert math.isfinite(a["episodic_reported_cost_return"])
                assert math.isfinite(a["violation_rate_so_far"])
                assert math.isfinite(a["peak_violation_so_far"])

    def test_three_training_quantities_present_finite_and_distinctly_labeled(self, tiny_config_factory):
        """critic prediction, value target, advantage -- kept separate,
        never mixed with the evaluation quantities above."""
        cfg = tiny_config_factory(run_id="return_reporting_training_quantities")
        out_dir = run_experiment_with_oracle(cfg)
        rounds = read_jsonl(out_dir / "rounds.jsonl")
        assert rounds
        for record in rounds:
            for c in record["constraints"].values():
                d = c["training_diagnostics"]
                assert math.isfinite(d["cost_critic_prediction_t0"])
                assert math.isfinite(d["cost_value_target_t0"])
                assert math.isfinite(d["cost_advantage_mean"])
                assert math.isfinite(d["task_critic_prediction_t0"])
                assert math.isfinite(d["task_value_target_t0"])
                assert math.isfinite(d["task_advantage_mean"])
                # value target at t0 is exactly ret_c[0] / ret_r[0],
                # which round_record also exposes at top level for
                # backward compatibility -- must never silently diverge.
                assert d["cost_value_target_t0"] == c["reported_cost_return"]
                assert d["task_value_target_t0"] == c["task_return"]

    def test_detection_gap_formulas_match_their_documented_definitions(self, tiny_config_factory):
        """Regression for the P0 #6 fix itself: detection_gap (vs the
        agent's own critic) and detection_gap_vs_aggregate (vs the
        mechanism/aggregate that actually drove the dual update) must
        equal true_cost_return minus the correspondingly-named reported
        quantity -- exactly, not approximately reconstructed."""
        cfg = tiny_config_factory(run_id="return_reporting_gap_formulas", defense_name="rce", f=1)
        out_dir = run_experiment_with_oracle(cfg)
        rounds = read_jsonl(out_dir / "rounds.jsonl")
        oracle_records = read_jsonl(out_dir / "oracle.jsonl")
        rounds_by_k = {r["round_k"]: r for r in rounds}
        for orec in oracle_records:
            k = orec["round_k"]
            rrec = rounds_by_k[k]
            for aid, a in orec["agents"].items():
                own_critic = rrec["constraints"][aid]["reported_cost_return"]
                mechanism = rrec["constraints"][aid]["mechanism_reported_cost_return"]
                assert math.isclose(a["detection_gap"], a["true_cost_return"] - own_critic, rel_tol=1e-9, abs_tol=1e-9)
                assert math.isclose(
                    a["detection_gap_vs_aggregate"], a["true_cost_return"] - mechanism, rel_tol=1e-9, abs_tol=1e-9
                )

    def test_mechanism_reported_cost_return_is_what_the_dual_update_actually_used(self, tiny_config_factory):
        """mechanism_reported_cost_return must equal point_estimate (or
        pessimistic_estimate for RCE) -- the exact value
        `safelie.training.loop.run_round` subtracts the budget from to
        form the dual update's residual. Not an approximation of it."""
        cfg = tiny_config_factory(run_id="return_reporting_mechanism_rce", defense_name="rce", f=1)
        out_dir = run_experiment_with_oracle(cfg)
        rounds = read_jsonl(out_dir / "rounds.jsonl")
        for record in rounds:
            for c in record["constraints"].values():
                agg = c["aggregate"]
                assert math.isclose(c["mechanism_reported_cost_return"], agg["pessimistic_estimate"], rel_tol=1e-9, abs_tol=1e-9)
                assert math.isclose(agg["pessimistic_estimate"], agg["point_estimate"] + agg["applied_margin"], rel_tol=1e-9, abs_tol=1e-9)
