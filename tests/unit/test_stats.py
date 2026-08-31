"""Decision D6 / PROJECT_REPORT.md §R2.3: Welch/Holm require >= 5 seeds."""

from __future__ import annotations

import numpy as np
import pytest

from safelie.analysis.stats import MIN_SEEDS_FOR_INFERENCE, holm_correction, welch_t_test


def test_welch_raises_below_min_seeds():
    a = np.array([1.0, 2.0, 3.0])  # 3 seeds, the compact pilot's count
    b = np.array([4.0, 5.0, 6.0])
    with pytest.raises(ValueError, match="requires >="):
        welch_t_test(a, b)


def test_welch_runs_at_min_seeds_and_detects_a_clear_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(24.0, 1.0, size=MIN_SEEDS_FOR_INFERENCE)
    b = rng.normal(58.0, 1.0, size=MIN_SEEDS_FOR_INFERENCE)
    result = welch_t_test(a, b)
    assert result.n_a == result.n_b == MIN_SEEDS_FOR_INFERENCE
    assert result.p_value < 0.05


def test_holm_correction_is_monotone_and_conservative_vs_uncorrected():
    p_values = [0.01, 0.04, 0.03, 0.20]
    adjusted = holm_correction(p_values)
    assert len(adjusted) == len(p_values)
    for raw, adj in zip(p_values, adjusted, strict=False):
        assert adj >= raw - 1e-12


def test_holm_correction_empty_input():
    assert holm_correction([]) == []
