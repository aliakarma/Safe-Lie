"""P0 #2/#3: RunningMeanStd correctness, independent of how it is wired
into AgentBundle."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from safelie.algos.normalization import RunningMeanStd


class TestRunningMeanStdMatchesBatchNumpy:
    def test_single_update_matches_numpy_mean_var(self):
        rms = RunningMeanStd(shape=(3,), epsilon=1e-8)
        data = np.random.default_rng(0).normal(size=(500, 3)) * [1.0, 5.0, 0.1] + [2.0, -3.0, 0.0]
        rms.update(data)
        np.testing.assert_allclose(rms.mean, data.mean(axis=0), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(rms.var, data.var(axis=0), rtol=1e-2, atol=1e-4)

    def test_sequential_batch_updates_converge_to_the_same_stats_as_one_big_batch(self):
        rng = np.random.default_rng(1)
        data = rng.normal(loc=5.0, scale=2.0, size=(1000, 2))

        one_shot = RunningMeanStd(shape=(2,), epsilon=1e-8)
        one_shot.update(data)

        sequential = RunningMeanStd(shape=(2,), epsilon=1e-8)
        for chunk in np.array_split(data, 10):
            sequential.update(chunk)

        np.testing.assert_allclose(one_shot.mean, sequential.mean, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(one_shot.var, sequential.var, rtol=1e-6, atol=1e-6)

    def test_scalar_shape_tracks_returns_correctly(self):
        rms = RunningMeanStd(shape=(), epsilon=1e-8)
        returns = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        rms.update(returns)
        assert rms.mean == pytest.approx(returns.mean(), rel=1e-6)
        assert rms.var == pytest.approx(returns.var(), rel=1e-2)


class TestNormalizeDenormalizeRoundTrip:
    def test_denormalize_inverts_normalize_for_a_converged_estimate(self):
        rng = np.random.default_rng(2)
        data = rng.normal(loc=25.0, scale=4.0, size=(5000,))
        rms = RunningMeanStd(shape=(), epsilon=1e-8)
        rms.update(data)

        raw = torch.as_tensor([25.0, 29.0, 21.0], dtype=torch.float32)
        normed = rms.normalize(raw, clip=1e9)  # no clipping, so the round trip is exact
        recovered = rms.denormalize(normed)
        torch.testing.assert_close(recovered, raw, rtol=1e-3, atol=1e-3)

    def test_clipping_bounds_extreme_values(self):
        rms = RunningMeanStd(shape=(), epsilon=1e-8)
        rms.update(np.array([0.0, 0.0, 0.0, 0.0]))  # mean=0, var~0 -> std floored by +1e-8
        extreme = torch.as_tensor([1000.0], dtype=torch.float32)
        normed = rms.normalize(extreme, clip=10.0)
        assert torch.all(normed <= 10.0) and torch.all(normed >= -10.0)

    def test_zero_variance_dimension_never_produces_nan(self):
        """A padded (always-zero) observation dimension -- e.g. a
        smaller-obs_dim agent under Safe MAMuJoCo's zero-padding -- must
        normalize to exactly 0, never NaN or Inf, regardless of how long
        the run continues."""
        rms = RunningMeanStd(shape=(2,), epsilon=1e-8)
        for _ in range(50):
            rms.update(np.array([[1.0, 0.0], [2.0, 0.0], [1.5, 0.0]]))
        x = torch.as_tensor([[1.7, 0.0]], dtype=torch.float32)
        normed = rms.normalize(x)
        assert torch.isfinite(normed).all()
        assert normed[0, 1].item() == pytest.approx(0.0, abs=1e-6)


class TestCheckpointRoundTrip:
    def test_state_dict_load_state_dict_round_trip_is_exact(self):
        rms = RunningMeanStd(shape=(4,), epsilon=1e-4)
        rng = np.random.default_rng(3)
        rms.update(rng.normal(size=(37, 4)))
        state = rms.state_dict()

        restored = RunningMeanStd(shape=(4,))
        restored.load_state_dict(state)
        np.testing.assert_array_equal(rms.mean, restored.mean)
        np.testing.assert_array_equal(rms.var, restored.var)
        assert rms.count == restored.count

        # Further updates on the restored tracker must continue exactly
        # as if the original had never stopped.
        more = rng.normal(size=(13, 4))
        rms.update(more)
        restored.update(more)
        np.testing.assert_allclose(rms.mean, restored.mean)
        np.testing.assert_allclose(rms.var, restored.var)
