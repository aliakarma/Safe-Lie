"""G1: the dual update's constraint-objective estimator.

The G0 clean baseline passed conditionally on one finding: the value the
dual update compared against the budget `d` was `ret_c[0]`, a GAE(lambda)
bootstrap target, which is a valid cost-critic regression target but not a
valid estimator of `J_C^i(theta) = E[sum_t gamma^t C_t^i]`. These tests
pin the replacement -- `safelie.training.constraint_return` -- against the
four things it must be, and against the five things section 4 of the G1
brief explicitly forbids it from silently becoming (an undiscounted sum, a
per-step average, a GAE advantage, a normalized return, or a critic
prediction on a different scale).

The load-bearing test in this file is
`TestScaleMatchesTheOracleExactly::test_matches_the_oracle_accumulation_step_for_step`:
the estimator and the withheld oracle must compute the SAME functional, or
the G1 bias comparison measures a definitional mismatch instead of
estimator error.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from safelie.training.buffer import AgentRollout
from safelie.training.constraint_return import (
    complete_target_count,
    complete_target_horizon,
    discounted_cost_to_go,
    discounted_window_return,
    episodic_mc_returns,
)
from safelie.training.gae import compute_gae


def _oracle_style_accumulation(costs, gamma: float) -> float:
    """A transcription of `safelie.eval.oracle.OracleEvaluator.record`'s
    accumulation, written out longhand and independently of the module
    under test: `_discounted_true_cost += (gamma ** _t) * c`, with `_t`
    starting at 0 on reset and incrementing every step, never resetting at
    an internal episode boundary."""
    total = 0.0
    t = 0
    for c in costs:
        total += (gamma**t) * float(c)
        t += 1
    return total


class TestTheDiscountedWindowReturnIsWhatItClaims:
    def test_constant_cost_matches_the_closed_form_geometric_sum(self):
        gamma, T = 0.99, 500
        got = discounted_window_return(np.ones(T), gamma)
        expected = (1.0 - gamma**T) / (1.0 - gamma)
        assert math.isclose(got, expected, rel_tol=1e-12)

    def test_window_return_is_the_first_element_of_the_cost_to_go(self):
        rng = np.random.default_rng(0)
        costs = rng.random(200)
        ctg = discounted_cost_to_go(costs, 0.99)
        assert math.isclose(discounted_window_return(costs, 0.99), float(ctg[0]), rel_tol=1e-12)

    def test_cost_to_go_satisfies_its_own_recursion(self):
        """G_t = c_t + gamma * G_{t+1}, with G_{T-1} = c_{T-1}."""
        rng = np.random.default_rng(1)
        costs = rng.random(64)
        gamma = 0.97
        g = discounted_cost_to_go(costs, gamma)
        assert math.isclose(g[-1], costs[-1], rel_tol=1e-12)
        for t in range(len(costs) - 1):
            assert math.isclose(g[t], costs[t] + gamma * g[t + 1], rel_tol=1e-12)

    def test_empty_rollout_returns_zero_rather_than_raising(self):
        assert discounted_window_return(np.array([]), 0.99) == 0.0
        assert len(discounted_cost_to_go(np.array([]), 0.99)) == 0


class TestScaleMatchesTheOracleExactly:
    """Gate G2's code-level counterpart. The learner's estimator and the
    withheld oracle must measure the SAME functional of the same cost
    stream; only then does `estimate - true_cost_return` isolate estimator
    error rather than a definitional mismatch."""

    @pytest.mark.parametrize("gamma", [0.9, 0.95, 0.99, 0.999])
    def test_matches_the_oracle_accumulation_step_for_step(self, gamma):
        rng = np.random.default_rng(7)
        costs = rng.random(1000) * (rng.random(1000) > 0.6)  # sparse, like a velocity indicator
        assert math.isclose(
            discounted_window_return(costs, gamma),
            _oracle_style_accumulation(costs, gamma),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

    def test_the_clock_does_not_reset_at_an_internal_episode_boundary(self):
        """The backend auto-resets mid-window; the oracle keeps counting on
        one clock (`OracleEpisodeResult`: "'episode' always denotes this
        fixed window"). An estimator that restarted its discount at the
        boundary would over-read by a factor of the discount it dropped."""
        costs = np.ones(20)
        gamma = 0.9
        window = discounted_window_return(costs, gamma)
        # Restarting the clock at t=10 would give two 10-step geometric
        # sums instead of one 20-step one -- a materially larger number.
        restarted = 2.0 * (1.0 - gamma**10) / (1.0 - gamma)
        assert window < restarted
        assert math.isclose(window, (1.0 - gamma**20) / (1.0 - gamma), rel_tol=1e-12)


class TestItIsNoneOfTheForbiddenAlternatives:
    """Section 4 of the G1 brief names five quantities the estimator must
    not accidentally become. Each gets a test that FAILS if the
    implementation drifts into it."""

    @staticmethod
    def _front_loaded_costs(T=1000):
        """Cost concentrated early in the window -- the shape the G0 runs
        actually exhibit (a late-training round with mean rate 0.105 has a
        true discounted return near 21, not 10.5)."""
        costs = np.zeros(T)
        costs[:150] = 1.0
        return costs

    def test_is_not_an_undiscounted_sum(self):
        costs = self._front_loaded_costs()
        mc = discounted_window_return(costs, 0.99)
        assert not math.isclose(mc, float(costs.sum()), rel_tol=0.05)
        assert mc < costs.sum()

    def test_is_not_a_per_step_average(self):
        costs = self._front_loaded_costs()
        mc = discounted_window_return(costs, 0.99)
        assert mc > 10.0 * float(costs.mean())

    def test_is_not_the_per_step_rate_times_the_effective_horizon(self):
        """`cost_rate_mean * 1/(1-gamma)` is the tempting shortcut and it
        is wrong whenever cost is not uniform in time, which it is not
        here. Keeping this explicit stops the shortcut being introduced as
        a 'simplification'."""
        costs = self._front_loaded_costs()
        mc = discounted_window_return(costs, 0.99)
        rate_proxy = float(costs.mean()) / (1.0 - 0.99)
        assert not math.isclose(mc, rate_proxy, rel_tol=0.1)

    def test_is_not_the_gae_lambda_target_when_the_critic_is_wrong(self):
        """The exact G0 defect, reproduced in miniature. With a cost critic
        stuck at zero -- an under-trained critic, which is what the G0 run
        had -- the GAE(lambda) target at t=0 keeps only ~1/(1-gamma*lambda)
        steps of real cost evidence and hands the rest to that zero
        bootstrap, so it under-reads the true discounted return badly. The
        MC estimator has no critic in the path and is exact."""
        gamma, lam, T = 0.99, 0.95, 2000
        costs = np.ones(T)
        cost_values = np.zeros(T)
        terminated = np.zeros(T, dtype=bool)
        truncated = np.zeros(T, dtype=bool)
        _adv, ret_c = compute_gae(costs, cost_values, terminated, truncated, gamma, lam, last_value=0.0)

        mc = discounted_window_return(costs, gamma)
        truth = (1.0 - gamma**T) / (1.0 - gamma)

        assert math.isclose(mc, truth, rel_tol=1e-12)
        assert float(ret_c[0]) < 0.25 * truth, (
            f"GAE(lambda) target {ret_c[0]:.3f} was expected to under-read the true "
            f"discounted return {truth:.3f} substantially; if this no longer holds the "
            f"premise of the G1 repair has changed and must be re-argued"
        )

    def test_is_raw_return_scale_not_a_normalized_return(self):
        """Doubling every per-step cost must double the estimate. A
        normalized return would be invariant to that rescaling."""
        rng = np.random.default_rng(3)
        costs = rng.random(300)
        a = discounted_window_return(costs, 0.99)
        b = discounted_window_return(2.0 * costs, 0.99)
        assert math.isclose(b, 2.0 * a, rel_tol=1e-12)


class TestTheRegressionTargetMask:
    def test_horizon_is_the_smallest_h_with_gamma_to_the_h_below_tol(self):
        for gamma, tol in [(0.99, 0.01), (0.95, 0.01), (0.9, 0.05)]:
            h = complete_target_horizon(gamma, tol)
            assert gamma**h <= tol
            assert gamma ** (h - 1) > tol

    def test_documented_value_at_the_pilot_settings(self):
        assert complete_target_horizon(0.99, 0.01) == 459
        assert complete_target_count(2000, 0.99) == 1541

    def test_a_window_shorter_than_the_horizon_keeps_every_row(self):
        """No masking is possible there; every target is censored to the
        same degree the window definition of J_C already is."""
        assert complete_target_count(200, 0.99) == 200
        assert complete_target_count(1, 0.99) == 1

    def test_retained_targets_are_complete_to_within_the_tolerance(self):
        """Every retained row's missing discounted mass, relative to a
        hypothetical continuation at the same cost level, is <= tol."""
        gamma, T, tol = 0.99, 2000, 0.01
        n = complete_target_count(T, gamma, tol)
        for t in (0, n // 2, n - 1):
            missing_fraction = gamma ** (T - t)
            assert missing_fraction <= tol

    def test_rejects_out_of_range_arguments(self):
        with pytest.raises(ValueError):
            complete_target_horizon(1.0, 0.01)
        with pytest.raises(ValueError):
            complete_target_horizon(0.99, 1.0)


class TestEpisodicDiagnostic:
    def test_segments_on_terminated_or_truncated_and_restarts_the_clock(self):
        costs = np.ones(6)
        terminated = np.zeros(6, dtype=bool)
        truncated = np.zeros(6, dtype=bool)
        terminated[2] = True
        truncated[5] = True
        res = episodic_mc_returns(costs, terminated, truncated, 0.5)
        assert res.n_complete == 2
        assert res.episode_lengths == (3, 3)
        expected = 1.0 + 0.5 + 0.25
        assert all(math.isclose(r, expected, rel_tol=1e-12) for r in res.episode_returns)
        assert res.censored_length == 0
        assert not res.fell_back

    def test_a_trailing_unfinished_episode_is_censored_not_counted(self):
        costs = np.ones(10)
        terminated = np.zeros(10, dtype=bool)
        truncated = np.zeros(10, dtype=bool)
        terminated[3] = True
        res = episodic_mc_returns(costs, terminated, truncated, 0.9)
        assert res.n_complete == 1
        assert res.censored_length == 6

    def test_falls_back_to_the_window_sum_when_no_episode_completes(self):
        costs = np.ones(10)
        z = np.zeros(10, dtype=bool)
        res = episodic_mc_returns(costs, z, z, 0.9)
        assert res.fell_back
        assert res.n_complete == 0
        assert math.isclose(res.mean_return, discounted_window_return(costs, 0.9), rel_tol=1e-12)

    def test_a_genuine_termination_takes_precedence_over_a_truncation(self):
        """Both flags on the same step must produce one boundary, not two
        (a two-boundary reading would emit a zero-length episode)."""
        costs = np.ones(4)
        terminated = np.array([False, True, False, False])
        truncated = np.array([False, True, False, True])
        res = episodic_mc_returns(costs, terminated, truncated, 0.9)
        assert res.n_complete == 2
        assert res.episode_lengths == (2, 2)


class TestTheRolloutBufferExposesBothEstimators:
    @staticmethod
    def _rollout(T=64, cost=1.0):
        r = AgentRollout()
        for t in range(T):
            r.add(
                obs=np.zeros(3),
                raw_action=np.zeros(2),
                logprob=0.0,
                reward=1.0,
                cost=cost,
                value=0.0,
                cost_value=0.0,
                terminated=False,
                truncated=(t == T - 1),
            )
        return r

    def test_finalize_reports_mc_and_gae_as_separate_fields(self):
        out = self._rollout().finalize(gamma=0.99, gae_lambda=0.95, last_value=0.0, last_cost_value=0.0)
        assert "mc_cost_return" in out
        assert "cost_return_estimate" in out
        assert out["mc_cost_return"] != out["cost_return_estimate"]

    def test_mc_cost_return_equals_the_direct_computation(self):
        out = self._rollout().finalize(gamma=0.99, gae_lambda=0.95, last_value=0.0, last_cost_value=0.0)
        assert math.isclose(
            out["mc_cost_return"], discounted_window_return(np.ones(64), 0.99), rel_tol=1e-12
        )

    def test_gae_returns_are_untouched_by_the_repair(self):
        """The advantage/critic-target path must be bit-for-bit what
        `compute_gae` produces: this repair separates the dual estimator
        from GAE, it does not modify GAE."""
        out = self._rollout().finalize(gamma=0.99, gae_lambda=0.95, last_value=0.0, last_cost_value=0.0)
        costs = np.ones(64)
        terminated = np.zeros(64, dtype=bool)
        truncated = np.zeros(64, dtype=bool)
        truncated[-1] = True
        adv_c, ret_c = compute_gae(
            costs, np.zeros(64), terminated, truncated, 0.99, 0.95, 0.0,
            truncation_bootstrap_values=np.zeros(64),
        )
        np.testing.assert_allclose(out["adv_c"], adv_c, rtol=0, atol=0)
        np.testing.assert_allclose(out["ret_c"], ret_c, rtol=0, atol=0)

    def test_mc_cost_to_go_has_one_entry_per_step(self):
        out = self._rollout(T=64).finalize(gamma=0.99, gae_lambda=0.95, last_value=0.0, last_cost_value=0.0)
        assert len(out["mc_cost_to_go"]) == 64
        assert out["n_mc_targets"] == complete_target_count(64, 0.99)

    def test_task_return_counterpart_is_on_the_same_clock(self):
        out = self._rollout().finalize(gamma=0.99, gae_lambda=0.95, last_value=0.0, last_cost_value=0.0)
        assert math.isclose(
            out["mc_task_return"], discounted_window_return(np.ones(64), 0.99), rel_tol=1e-12
        )
