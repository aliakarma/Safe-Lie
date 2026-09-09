"""Property tests for the aggregator zoo.

Report reference: smoke tests S6, S7, S8, S21, S22, S23, S24.
"""

from __future__ import annotations

import warnings

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
# A3 — the margin at M=5, f=1, where |T| = M - 2f = 3 = min_retained.
#
# docs/a3_gates.md section 6, A3-G3. A2 ran at M=3, f=1, where the retained
# set holds ONE value, the MAD is identically zero, the floor always fires,
# and the margin is the constant beta*sigma_min = 0.0015. The whole point of
# A3 is that at M=5 the floor becomes UNREACHABLE and `spread` becomes a real
# measurement. These tests fail loudly if that stops being true, because an
# A3 run against a flooring implementation would silently be another A2 while
# every log still said "M=5".
#
# The boundary is exact and therefore fragile. `rce_aggregate` floors on
# `retained_n < min_retained`, so at retained_n == 3 == min_retained the
# comparison is `3 < 3` -> False. Changing that to `<=`, or raising
# min_retained to 4, would re-degenerate the mechanism without touching a
# single field of any config. That is the regression these pin.
# ---------------------------------------------------------------------------


M5_F1 = dict(f=1, beta=1.5, sigma_min=1e-3, min_retained=3)


def test_rce_at_m5_f1_retains_three_and_trims_one_min_and_one_max():
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    result = rce_aggregate(values, **M5_F1)
    assert result.retained_n == 3  # M - 2f
    np.testing.assert_allclose(np.sort(result.retained_values), [20.0, 30.0, 40.0])
    assert 10.0 not in result.retained_values  # the one minimum, trimmed
    assert 50.0 not in result.retained_values  # the one maximum, trimmed


def test_rce_at_m5_f1_is_not_degenerate_and_raises_no_floor_warning():
    """A3-G3-i: `degenerate` is False and no floor warning fires."""
    values = np.array([24.0, 24.5, 23.8, 24.2, 24.1])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # a floor warning fails the test
        result = rce_aggregate(values, **M5_F1)
    assert result.degenerate is False
    assert result.retained_n == 3


def test_rce_at_m5_f1_does_not_floor_even_when_the_true_mad_is_below_sigma_min():
    """The load-bearing case.

    Three honest sources can legitimately land within 1e-7 of each other, and
    when they do the correct behaviour is to report that tiny MAD, not to
    inflate it to `sigma_min`. The floor exists to refuse a confident-looking
    margin over a retained set too small to *have* a dispersion (|T| = 1); it
    is not a minimum-margin policy. If this test fails, `spread` at M=5 is no
    longer a measurement and A3-G3-ii is measuring the floor.
    """
    values = np.array([0.0, 1.0, 1.0000001, 1.0000002, 5.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = rce_aggregate(values, **M5_F1)
    assert result.retained_n == 3
    assert result.degenerate is False
    assert result.spread == pytest.approx(1e-7, rel=1e-6)
    assert result.spread < M5_F1["sigma_min"]  # below the floor, and un-floored
    assert result.applied_margin == pytest.approx(1.5 * result.spread)


def test_rce_at_m5_f1_reports_a_genuinely_zero_mad_without_flooring_it():
    """Three exactly-tied retained values give MAD = 0 honestly.

    This is what A3-G3-ii's ">= 99 % of cells" threshold exists for: three
    honest values may coincide, and a coincidence is not a degeneracy. The
    margin collapses to zero for that cell and `degenerate` stays False,
    because the retained set was large enough to have a dispersion — it just
    happened to have none.
    """
    values = np.array([0.0, 7.0, 7.0, 7.0, 9.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = rce_aggregate(values, **M5_F1)
    assert result.retained_n == 3
    assert result.degenerate is False
    assert result.spread == 0.0
    assert result.applied_margin == 0.0


def test_rce_at_m5_f1_margin_is_exactly_beta_times_the_retained_mad():
    """A3 section 7's counterfactual: Y_RCE - Y_trim == beta * MAD, exactly.

    Checked against `trimmed_mean_aggregator` rather than a reimplementation,
    so this also pins that RCE's point estimate IS the trimmed mean and the
    margin is the only difference between them.
    """
    rng = np.random.default_rng(20260909)
    for _ in range(2000):
        values = rng.normal(25.0, 1.053, size=5)
        trim = trimmed_mean_aggregator(values, f=1)
        rce = rce_aggregate(values, **M5_F1)
        assert rce.point_estimate == pytest.approx(trim.point_estimate, abs=0.0)
        assert rce.spread == pytest.approx(trim.spread, abs=0.0)
        assert rce.applied_margin == pytest.approx(1.5 * trim.spread, abs=0.0)
        assert rce.pessimistic_estimate == pytest.approx(
            trim.point_estimate + 1.5 * trim.spread, abs=0.0
        )


def test_rce_at_m5_f1_mad_is_the_minimum_adjacent_gap_of_the_retained_three():
    """Pins the statistic, not merely its non-degeneracy.

    For a sorted retained triple a <= b <= c the unscaled MAD is
    median{b-a, 0, c-b} = min(b-a, c-b) — the SMALLER of the two adjacent
    gaps. Worth pinning because it is why the M=5 margin is modest: MAD over
    three points is a minimum-gap statistic, not a range, so it reads well
    below the sample's actual spread. A change of MAD convention (e.g.
    adopting the 1.4826 consistency constant, docs/assumptions.md) would
    rescale every A3 margin and must not pass silently.
    """
    rng = np.random.default_rng(7)
    for _ in range(2000):
        values = rng.normal(25.0, 1.053, size=5)
        a, b, c = np.sort(trimmed_mean_aggregator(values, f=1).retained_values)
        assert rce_aggregate(values, **M5_F1).spread == pytest.approx(
            min(b - a, c - b), abs=1e-12
        )


@pytest.mark.parametrize(
    "m,f,retained_n,floors",
    [
        (3, 1, 1, True),   # A2's operating point — the floor MUST fire
        (5, 2, 1, True),   # |T| = 1 by a different route
        (5, 1, 3, False),  # A3's operating point — the floor MUST NOT fire
        (7, 2, 3, False),  # |T| = 3 = min_retained, the same boundary
        (7, 1, 5, False),
    ],
)
def test_rce_floor_fires_only_strictly_below_min_retained(m, f, retained_n, floors):
    """The boundary itself, swept.

    `min_retained` is a strict lower bound: |T| == min_retained is satisfied,
    not violated. Flipping the comparison to `<=` would pass every other test
    in this file and quietly turn A3 back into A2.
    """
    values = np.linspace(0.0, 100.0, m)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = rce_aggregate(values, f=f, beta=1.5, sigma_min=1e-3, min_retained=3)
    fired = any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert result.retained_n == retained_n
    assert fired is floors
    assert result.degenerate is floors
    if floors:
        assert result.spread == pytest.approx(1e-3)
    else:
        assert result.spread > 1e-3  # a real MAD over a linspace, nowhere near the floor


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
