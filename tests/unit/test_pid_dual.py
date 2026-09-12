"""PID-Lagrangian multiplier controller (Stooke et al., 2020).

The load-bearing test in this file is `test_pid_reduces_to_lagrangian_bitwise`.
Everything else checks a property of the controller; that one checks that
adding the controller did not change the experiment that is already frozen.
A1, A2 and A3's twelve configs all select `controller="lagrangian"` by
omission, so if that reduction ever stops holding bit for bit, a completed
campaign's code path has moved underneath it.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from safelie.consensus.topologies import build_topology
from safelie.training.dual import dual_update, pid_dual_update
from safelie.utils.config import DualConfig

LAM_MAX = 25.0
ETA = 0.035


def _W(n: int = 6) -> np.ndarray:
    return build_topology("ring", n)


def _residual_stream(rounds: int = 200, n: int = 6, seed: int = 0) -> list[np.ndarray]:
    """Residuals that both change sign and stay large enough to clip.

    A stream that never reaches the projection boundary would let a broken
    `np.clip` pass, and a strictly-positive stream would never exercise the
    one-sided derivative's `max(0, .)`.
    """
    rng = np.random.default_rng(seed)
    return [rng.normal(loc=2.0 * np.sin(k / 9.0), scale=6.0, size=n) for k in range(rounds)]


# --------------------------------------------------------------------------
# The reduction. This is the regression guard on A1/A2/A3.
# --------------------------------------------------------------------------

def test_pid_reduces_to_lagrangian_bitwise() -> None:
    """k_p = k_d = 0, k_i = eta_lambda must reproduce `dual_update` exactly.

    Compared against `dual_update` itself, not a transcription of it, so the
    test cannot drift away from the function it is pinning.
    """
    W = _W()
    lam = np.zeros(6)
    integral = np.zeros(6)
    prev = None

    for residual in _residual_stream():
        lam = dual_update(lam, W, ETA, residual, LAM_MAX)
        pid_lam, integral = pid_dual_update(
            integral, residual if prev is None else prev, W, 0.0, ETA, 0.0, residual, LAM_MAX
        )
        prev = residual.copy()
        # Bitwise, not approximate: the two must be the same arithmetic.
        assert np.array_equal(lam, pid_lam)
        assert np.array_equal(integral, pid_lam)


def test_reduction_holds_when_the_projection_clips() -> None:
    """The reduction must survive the boundary, where clip order could differ."""
    W = _W()
    lam = np.zeros(6)
    integral = np.zeros(6)
    # Large positive residuals drive straight into lam_max and pin there.
    for _ in range(400):
        residual = np.full(6, 90.0)
        lam = dual_update(lam, W, ETA, residual, LAM_MAX)
        pid_lam, integral = pid_dual_update(integral, residual, W, 0.0, ETA, 0.0, residual, LAM_MAX)
        assert np.array_equal(lam, pid_lam)
    assert np.allclose(lam, LAM_MAX)

    # ...and at the lower boundary, where the multiplier must not go negative.
    for _ in range(400):
        residual = np.full(6, -90.0)
        lam = dual_update(lam, W, ETA, residual, LAM_MAX)
        pid_lam, integral = pid_dual_update(integral, residual, W, 0.0, ETA, 0.0, residual, LAM_MAX)
        assert np.array_equal(lam, pid_lam)
    assert np.allclose(lam, 0.0)


# --------------------------------------------------------------------------
# Controller properties
# --------------------------------------------------------------------------

def test_derivative_is_one_sided() -> None:
    """A falling cost must not discount the multiplier.

    This is the whole point of Stooke et al.'s `max(0, .)`: a two-sided
    derivative would subtract while the policy is already improving, which
    is when the constraint is least in danger and the multiplier least
    needs relaxing.
    """
    W = _W()
    rising = np.full(6, 5.0)
    falling = np.full(6, -5.0)

    # Cost rising: the derivative term contributes.
    lam_rise, _ = pid_dual_update(np.zeros(6), falling, W, 0.0, 0.0, 1.0, rising, LAM_MAX)
    assert np.all(lam_rise > 0.0)

    # Cost falling by the same magnitude: it contributes exactly nothing,
    # rather than an equal and opposite amount.
    lam_fall, _ = pid_dual_update(np.zeros(6), rising, W, 0.0, 0.0, 1.0, falling, LAM_MAX)
    assert np.array_equal(lam_fall, np.zeros(6))


def test_derivative_is_zero_on_the_first_round() -> None:
    """Seeding `prev_residual` with `Delta_0` makes round 0's derivative 0."""
    W = _W()
    d0 = np.array([7.0, -3.0, 0.0, 12.0, -8.5, 1.0])
    with_d, _ = pid_dual_update(np.zeros(6), d0, W, 0.0, ETA, 50.0, d0, LAM_MAX)
    without_d, _ = pid_dual_update(np.zeros(6), d0, W, 0.0, ETA, 0.0, d0, LAM_MAX)
    assert np.array_equal(with_d, without_d)


def test_proportional_bias_does_not_accumulate_but_integral_does() -> None:
    """The transfer-function claim, made concrete.

    main_iclr.tex sec. 2 keeps PID-Lagrangian as a baseline because it
    "alters the attack's transfer function". Under a persistent bias the
    integral path grows without bound in K (Theorem 1's mass conservation)
    while the proportional path contributes a fixed offset. That difference
    is the reason to run the baseline at all, so it is pinned here.
    """
    W = _W()
    bias = np.full(6, 0.5)

    # Pure proportional: the multiplier is a function of the CURRENT
    # residual only, so it is identical at round 10 and round 500.
    lam_p_early = np.zeros(6)
    integral = np.zeros(6)
    for k in range(500):
        lam_p, integral = pid_dual_update(integral, bias, W, 1.0, 0.0, 0.0, bias, LAM_MAX)
        if k == 10:
            lam_p_early = lam_p
    assert np.array_equal(lam_p_early, lam_p)
    assert np.allclose(integral, 0.0)  # k_i = 0 -> nothing accumulates

    # Pure integral under the same bias: grows linearly in K, without bound.
    # `W` is doubly stochastic and the bias is uniform across agents, so
    # `W @ I_k == I_k` and the recursion is exactly `I_{k+1} = I_k + k_i * b`
    # -- Theorem 1's `eta * delta * K` growth, per agent. Pinning the closed
    # form rather than just "it went up" is what distinguishes accumulation
    # from any other monotone response.
    integral = np.zeros(6)
    seen = []
    for _ in range(500):
        lam_i, integral = pid_dual_update(integral, bias, W, 0.0, ETA, 0.0, bias, LAM_MAX)
        seen.append(lam_i.copy())
    assert seen[10].max() < seen[100].max() < seen[400].max()
    for k in (10, 100, 400, 499):
        assert np.allclose(seen[k], ETA * 0.5 * (k + 1))
    # Still far below the projection at K=500, which is the point: nothing
    # here is bounded by anything except lam_max, eventually.
    assert seen[-1].max() < LAM_MAX


def test_theorem_1_mass_conservation_holds_on_the_integral() -> None:
    """`1^T I_K == k_i * sum_k 1^T Delta_k`, independent of W.

    The proof uses only `1^T W = 1^T`, so putting the mixing on the integral
    is what preserves it.

    The identity is a property of the UNPROJECTED recursion --
    `safelie.theory.mass_conservation` simulates it with no projection at
    all -- so the residual stream below is strictly positive, keeping the
    integral off both bounds for all 300 rounds. The test asserts that
    precondition rather than assuming it: clipping at either end destroys
    the identity, and a stream that quietly started clipping would otherwise
    turn this into a test of nothing.
    """
    for topology in ("ring", "complete", "star", "identity"):
        W = build_topology(topology, 6)
        integral = np.zeros(6)
        total = 0.0
        rng = np.random.default_rng(7)
        for _ in range(300):
            residual = rng.normal(0.5, 0.1, size=6)
            total += float(residual.sum())
            _, integral = pid_dual_update(integral, residual, W, 0.0, ETA, 0.0, residual, 1e9)
            # Projection inactive at both ends, every round.
            assert np.all(integral > 0.0) and np.all(integral < 1e9)
        assert float(integral.sum()) == pytest.approx(ETA * total, rel=1e-9, abs=1e-9)


def test_projection_is_respected_under_every_gain() -> None:
    W = _W()
    rng = np.random.default_rng(3)
    integral = np.zeros(6)
    prev = np.zeros(6)
    for _ in range(500):
        residual = rng.normal(0.0, 40.0, size=6)
        lam, integral = pid_dual_update(integral, prev, W, 2.0, 0.5, 3.0, residual, LAM_MAX)
        prev = residual.copy()
        assert np.all(lam >= 0.0) and np.all(lam <= LAM_MAX)
        assert np.all(integral >= 0.0) and np.all(integral <= LAM_MAX)


def test_no_nan_from_extreme_residuals() -> None:
    W = _W()
    integral = np.zeros(6)
    for residual in (np.full(6, 1e12), np.full(6, -1e12), np.zeros(6)):
        lam, integral = pid_dual_update(integral, np.zeros(6), W, 1.0, 1.0, 1.0, residual, LAM_MAX)
        assert np.all(np.isfinite(lam)) and np.all(np.isfinite(integral))


# --------------------------------------------------------------------------
# Liveness (Proposition 3): structural, same argument as `dual_update`
# --------------------------------------------------------------------------

def test_pid_update_has_no_branches() -> None:
    """No `if` may guard the update -- a gate reintroduces the deadlock.

    Checked on the compiled bytecode rather than the source text, so a
    branch smuggled in through a comprehension or a ternary is caught too.
    """
    import dis

    jumps = {
        i.opname for i in dis.get_instructions(pid_dual_update)
        if "JUMP" in i.opname
    }
    assert jumps == set(), f"pid_dual_update must have exactly one code path; found {jumps}"


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------

def test_default_controller_is_lagrangian() -> None:
    """Omission must select the frozen path. A1/A2/A3 depend on this."""
    assert DualConfig().controller == "lagrangian"
    assert (DualConfig().k_p, DualConfig().k_i, DualConfig().k_d) == (None, None, None)


def test_pid_requires_explicit_gains() -> None:
    with pytest.raises(ValidationError, match="requires explicit gains"):
        DualConfig(controller="pid")
    with pytest.raises(ValidationError, match="k_d"):
        DualConfig(controller="pid", k_p=1.0, k_i=0.035)
    # All three present is accepted.
    cfg = DualConfig(controller="pid", k_p=1.0, k_i=0.035, k_d=2.0)
    assert (cfg.k_p, cfg.k_i, cfg.k_d) == (1.0, 0.035, 2.0)


def test_lagrangian_config_rejects_dead_gains() -> None:
    """A gain nothing reads is a silent lie about what the run did."""
    with pytest.raises(ValidationError, match="does not read"):
        DualConfig(k_p=1.0)


def test_negative_gains_rejected() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        DualConfig(controller="pid", k_p=-1.0, k_i=0.035, k_d=0.0)
