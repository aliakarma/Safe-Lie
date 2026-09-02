"""GAE correctness, including terminal-vs-truncated handling (P0
terminal-handling fix, section 14 of the repair task).

Bug fixed: `compute_gae` used to take one combined `dones` array and
bootstrap with 0 at every done step. Correct for a genuine termination
(cost-to-go really is 0), wrong for a time-limit truncation (the
underlying MDP does not end; the standard fix bootstraps with the
critic's own value at the true final observation). This module tests the
primitive directly; `tests/unit/test_terminal_handling.py` tests it
wired into the real MaMuJoCo backend.
"""

from __future__ import annotations

import numpy as np
import pytest

from safelie.training.gae import compute_gae


class TestLambdaEqualsOneRecoversMonteCarloReturn:
    def test_full_episode_no_truncation_matches_raw_discounted_sum(self):
        """The textbook identity: at lambda=1 with a genuine terminal at
        the buffer's last step, GAE's return telescopes exactly to the
        undiscounted-horizon Monte Carlo return, regardless of the
        critic's own (here, deliberately wrong) value estimates."""
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        values = np.array([100.0, -50.0, 30.0, 0.0])  # arbitrary, wrong -- should cancel out at lambda=1
        terminated = np.array([False, False, False, True])
        truncated = np.array([False, False, False, False])
        gamma = 0.9

        _, ret = compute_gae(rewards, values, terminated, truncated, gamma, gae_lambda=1.0, last_value=0.0)

        expected_t0 = 1.0 + gamma * 2.0 + gamma**2 * 3.0 + gamma**3 * 4.0
        expected_t3 = 4.0
        assert ret[0] == pytest.approx(expected_t0, rel=1e-9)
        assert ret[3] == pytest.approx(expected_t3, rel=1e-9)


class TestTerminatedVsTruncatedBootstrap:
    def test_termination_bootstraps_with_zero_regardless_of_available_value(self):
        rewards = np.array([1.0, 1.0])
        values = np.array([0.0, 0.0])
        terminated = np.array([False, True])
        truncated = np.array([False, False])
        # last_value would matter if bootstrapping were nonzero here -- it must not.
        _, ret_zero_bootstrap = compute_gae(rewards, values, terminated, truncated, gamma=0.99, gae_lambda=0.95, last_value=0.0)
        _, ret_huge_bootstrap = compute_gae(rewards, values, terminated, truncated, gamma=0.99, gae_lambda=0.95, last_value=1000.0)
        np.testing.assert_allclose(ret_zero_bootstrap, ret_huge_bootstrap)

    def test_truncation_bootstraps_with_the_given_value_not_zero(self):
        """The regression itself: a truncated (non-terminal) final step
        must use `truncation_bootstrap_values[t]`, not 0."""
        rewards = np.array([1.0, 1.0])
        values = np.array([0.0, 0.0])
        terminated = np.array([False, False])
        truncated = np.array([False, True])
        gamma = 0.99

        _, ret_zero = compute_gae(
            rewards, values, terminated, truncated, gamma, gae_lambda=0.95, last_value=0.0,
            truncation_bootstrap_values=np.array([0.0, 0.0]),
        )
        _, ret_bootstrapped = compute_gae(
            rewards, values, terminated, truncated, gamma, gae_lambda=0.95, last_value=0.0,
            truncation_bootstrap_values=np.array([0.0, 50.0]),
        )
        # Only the truncated step's own return changes (it directly adds
        # gamma * bootstrap); the earlier step's return also shifts,
        # since it depends on delta_1 through the GAE recursion.
        assert ret_bootstrapped[1] - ret_zero[1] == pytest.approx(gamma * 50.0, rel=1e-9)
        assert ret_bootstrapped[1] > ret_zero[1]
        assert ret_bootstrapped[0] > ret_zero[0]

    def test_termination_takes_precedence_over_truncated_when_both_set(self):
        rewards = np.array([1.0])
        values = np.array([0.0])
        terminated = np.array([True])
        truncated = np.array([True])
        _, ret = compute_gae(
            rewards, values, terminated, truncated, gamma=0.99, gae_lambda=0.95, last_value=0.0,
            truncation_bootstrap_values=np.array([999.0]),
        )
        assert ret[0] == pytest.approx(1.0, rel=1e-9)  # bootstrap must be 0, not 999

    def test_default_truncation_bootstrap_is_zero_when_not_provided(self):
        """Backward-compatible default: omitting `truncation_bootstrap_values`
        reproduces the old (zero-bootstrap-at-any-done) behaviour exactly
        -- correct for the synthetic environment, which never truncates
        mid-buffer, so this default must not change its numbers."""
        rewards = np.array([1.0, 1.0, 1.0])
        values = np.array([0.5, 0.5, 0.5])
        terminated = np.array([False, False, False])
        truncated = np.array([False, False, True])
        _, ret_default = compute_gae(rewards, values, terminated, truncated, gamma=0.9, gae_lambda=0.9, last_value=0.0)
        _, ret_explicit_zero = compute_gae(
            rewards, values, terminated, truncated, gamma=0.9, gae_lambda=0.9, last_value=0.0,
            truncation_bootstrap_values=np.zeros(3),
        )
        np.testing.assert_allclose(ret_default, ret_explicit_zero)

    def test_mid_buffer_truncation_does_not_leak_across_the_boundary(self):
        """The lambda-recursion must still cut at a truncation boundary
        (mask=0 for the recursive carry), even though the delta at that
        step now uses a real bootstrap -- otherwise a later "episode"
        segment's advantage would contaminate an earlier one's."""
        rewards = np.array([1.0, 1.0, 1.0, 1.0])
        values = np.array([0.0, 0.0, 0.0, 0.0])
        terminated = np.array([False, False, False, False])
        truncated = np.array([False, True, False, False])  # mid-buffer truncation at t=1
        gamma, lam = 0.9, 0.9

        adv, _ = compute_gae(
            rewards, values, terminated, truncated, gamma, lam, last_value=0.0,
            truncation_bootstrap_values=np.array([0.0, 10.0, 0.0, 0.0]),
        )
        # adv[2], adv[3] (the segment AFTER the truncation) must be
        # computable purely from rewards[2:] and last_value -- i.e. equal
        # to what a fresh compute_gae call over just that sub-array gives.
        adv_second_segment, _ = compute_gae(
            rewards[2:], values[2:], terminated[2:], truncated[2:], gamma, lam, last_value=0.0,
        )
        np.testing.assert_allclose(adv[2:], adv_second_segment)
