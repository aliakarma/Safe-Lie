"""Property tests for the aggregator zoo.

Report reference: smoke tests S6, S7, S8, S21, S22, S23, S24.
"""

from __future__ import annotations

import numpy as np
import pytest

from safelie.defenses import aggregate
from safelie.defenses.krum import krum_aggregator
from safelie.defenses.mean import mean_aggregator
from safelie.defenses.median import coordinate_median_aggregator
from safelie.defenses.rce import rce_aggregate
from safelie.defenses.trimmean import trimmed_mean_aggregator

# ---------------------------------------------------------------------------
# S21 — mean has no breakdown point
# ---------------------------------------------------------------------------


def test_mean_shifts_exactly_by_corruption_over_m():
    honest = np.array([24.0, 24.5, 23.8, 24.2, 24.1, 23.9])
    B = 6.0
    corrupted = honest.copy()
    corrupted[0] -= B
    result_honest = mean_aggregator(honest)
    result_corrupted = mean_aggregator(corrupted)
    assert result_corrupted.point_estimate == pytest.approx(
        result_honest.point_estimate - B / len(honest), abs=1e-9
    )


def test_mean_is_arbitrarily_corruptible():
    honest = np.array([24.0, 24.5, 23.8, 24.2, 24.1, 23.9])
    for magnitude in [1e3, 1e6, 1e9]:
        corrupted = honest.copy()
        corrupted[0] += magnitude
        result = mean_aggregator(corrupted)
        assert result.point_estimate > magnitude / len(honest) * 0.5


# ---------------------------------------------------------------------------
# S6 — trimmed mean correctness
# ---------------------------------------------------------------------------


def test_trimmed_mean_matches_reference_on_random_inputs():
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = rng.integers(7, 15)
        f = rng.integers(0, (m - 1) // 2)
        values = rng.normal(24.0, 2.0, size=m)
        result = trimmed_mean_aggregator(values, f)
        reference = np.sort(values)[f : m - f].mean() if f > 0 else values.mean()
        assert result.point_estimate == pytest.approx(reference)
        assert result.retained_n == m - 2 * f


def test_trimmed_mean_stays_within_honest_range_under_unbounded_outliers():
    honest = np.array([24.0, 24.5, 23.8, 24.2, 24.1])
    f = 1
    values = np.concatenate([honest, [1e9], [-1e9]])  # M=7, f=1
    result = trimmed_mean_aggregator(values, f)
    assert honest.min() <= result.point_estimate <= honest.max()
    assert result.retained_n == len(values) - 2 * f


def test_trimmed_mean_raises_on_m_le_2f():
    values = np.arange(7, dtype=float)
    with pytest.raises(ValueError, match="M > 2f"):
        trimmed_mean_aggregator(values, f=4)  # M=7, f=4 -> M - 2f = -1, undefined
    with pytest.raises(ValueError, match="M > 2f"):
        trimmed_mean_aggregator(np.arange(6, dtype=float), f=3)  # M=6, f=3 -> M - 2f = 0


def test_trimmed_mean_f3_of_m7_is_boundary_not_raise():
    """M=7, f=3 leaves |T|=1: legal (M > 2f, since 7 > 6) but degenerate."""
    values = np.array([24.0, 24.5, 23.8, 24.2, 24.1, -1e9, 1e9])
    result = trimmed_mean_aggregator(values, f=3)
    assert result.retained_n == 1
    assert result.spread == 0.0
    assert result.degenerate is True


# ---------------------------------------------------------------------------
# S22 — RCE reduces outlier impact
# ---------------------------------------------------------------------------


def test_rce_output_stays_within_honest_range():
    honest = np.array([24.0, 24.5, 23.8, 24.2, 24.1])
    values = np.concatenate([honest, [-100.0], [100.0]])  # M=7, f=1
    result = rce_aggregate(values, f=1, beta=1.5)
    assert honest.min() <= result.point_estimate <= honest.max()
    # pessimistic estimate is inflated above the point estimate
    assert result.pessimistic_estimate >= result.point_estimate


# ---------------------------------------------------------------------------
# S8 — degenerate retained set floors and flags, never silently zeroes
# ---------------------------------------------------------------------------


def test_rce_floors_and_flags_degenerate_spread():
    values = np.array([24.0, 24.5, 23.8, 24.2, 24.1, -1e9, 1e9])  # M=7, f=3 -> |T|=1
    with pytest.warns(RuntimeWarning):
        result = rce_aggregate(values, f=3, beta=1.5, sigma_min=0.5)
    assert result.degenerate is True
    assert result.spread == pytest.approx(0.5)  # floored, not silently 0.0
    assert result.applied_margin == pytest.approx(1.5 * 0.5)


def test_rce_raises_at_m_le_2f_never_silently_degrades():
    values = np.arange(7, dtype=float)
    with pytest.raises(ValueError, match="M > 2f"):
        rce_aggregate(values, f=4, beta=1.5)


# ---------------------------------------------------------------------------
# S23 — over-reporting does not block: the aggregator always returns
# ---------------------------------------------------------------------------


def test_rce_returns_a_finite_estimate_under_extreme_over_reporting():
    honest = np.array([24.0, 24.5, 23.8, 24.2, 24.1])
    values = np.concatenate([honest, [1e6, 1e6]])  # both corrupted sources over-report
    result = rce_aggregate(values, f=2, beta=1.5)
    assert np.isfinite(result.pessimistic_estimate)
    # the two 1e6 outliers must have been trimmed away, not pulled the mean up
    assert result.point_estimate < 30.0


# ---------------------------------------------------------------------------
# Krum
# ---------------------------------------------------------------------------


def test_krum_selects_a_clustered_honest_value_under_one_outlier():
    honest = np.array([24.0, 24.5, 23.8, 24.2, 24.1])
    values = np.concatenate([honest, [1e6]])  # M=6, f=1 -> needs M>=f+3=4, ok
    result = krum_aggregator(values, f=1)
    assert result.point_estimate in honest


def test_krum_raises_below_minimum_m():
    values = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="M >= f \\+ 3"):
        krum_aggregator(values, f=1)  # needs M >= 4


# ---------------------------------------------------------------------------
# Coordinate median
# ---------------------------------------------------------------------------


def test_coordinate_median_is_robust_to_minority_outliers():
    honest = np.array([24.0, 24.5, 23.8, 24.2, 24.1])
    values = np.concatenate([honest, [1e9]])
    result = coordinate_median_aggregator(values)
    assert honest.min() <= result.point_estimate <= honest.max()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_aggregate_dispatch_covers_all_names():
    values = np.array([24.0, 24.5, 23.8, 24.2, 24.1, 23.9, 24.3])
    for name in ["mean", "coordinate_median", "trimmean"]:
        result = aggregate(name, values, f=1)
        assert np.isfinite(result.point_estimate)
    rce_result = aggregate("rce", values, f=1, beta=1.5)
    assert np.isfinite(rce_result.pessimistic_estimate)
    krum_result = aggregate("krum", values, f=1)
    assert np.isfinite(krum_result.point_estimate)
