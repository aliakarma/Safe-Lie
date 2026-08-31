"""Smoke test S27 / Theory Test B: Proposition 1, spreading and stealth.

PROJECT_REPORT.md §R3.2: three checks — uniformity bounded as K grows,
the residual bound respecting 1/(1-sigma_2(W)), and the stealth claim
made numerically (median-referenced deviation stays O(1) under consensus,
grows linearly in K under W=I).
"""

from __future__ import annotations

import numpy as np
import pytest
from safelie.consensus.topologies import build_topology
from safelie.theory.spreading import verify_spreading

N = 6
ETA = 0.035
DELTA = 2.0
AGENT = 0


@pytest.mark.parametrize("topology_name", ["complete", "ring", "star", "erdos_renyi"])
@pytest.mark.parametrize("K", [50, 500, 2000])
def test_uniformity_bounded_as_k_grows(topology_name, K):
    """||e_K - (eta*delta*K/N) * 1||_2 must not grow with K for a connected,
    aperiodic topology."""
    kwargs = {"p": 0.5, "graph_seed": 3} if topology_name == "erdos_renyi" else {}
    W = build_topology(topology_name, N, **kwargs)
    result = verify_spreading(W, DELTA, AGENT, N, ETA, K)
    # The residual r_K is bounded by 1/(1-sigma2) uniformly in K, hence the
    # gap to the uniform component must not scale with K.
    assert result.residual_bound_satisfied, (
        f"{topology_name}, K={K}: residual norm {result.residual_norm:.4f} "
        f"exceeds bound {result.residual_bound:.4f}"
    )


@pytest.mark.parametrize("topology_name", ["complete", "ring", "star"])
def test_residual_bound_matches_sigma2(topology_name):
    W = build_topology(topology_name, N)
    result = verify_spreading(W, DELTA, AGENT, N, ETA, K=1000)
    assert 0.0 <= result.sigma2 < 1.0
    assert result.residual_bound == pytest.approx(1.0 / (1.0 - result.sigma2))


def test_identity_topology_diverges_and_concentrates():
    """W = I: sigma_2 = 1, the series does not converge, and the bias
    concentrates entirely on the corrupted agent (Corollary/Prop. 1's
    recovery of the concentrated case)."""
    W = build_topology("identity", N)
    K = 500
    result = verify_spreading(W, DELTA, AGENT, N, ETA, K)
    assert result.sigma2 == pytest.approx(1.0)
    assert result.residual_bound == float("inf")
    expected_concentrated = np.zeros(N)
    expected_concentrated[AGENT] = ETA * DELTA * K
    np.testing.assert_allclose(result.e_K, expected_concentrated, atol=1e-9)


def test_stealth_claim_median_deviation_bounded_under_consensus_but_not_under_identity():
    """The operational content of Proposition 1: under a connected topology
    the median-referenced deviation stays O(1) as K grows; under W=I it
    grows linearly in K. This is the number that decides whether
    median-referenced monitoring is blind."""
    K_small, K_large = 200, 2000

    W_ring = build_topology("ring", N)
    dev_small = verify_spreading(W_ring, DELTA, AGENT, N, ETA, K_small).median_deviation
    dev_large = verify_spreading(W_ring, DELTA, AGENT, N, ETA, K_large).median_deviation
    # O(1): the K=2000 deviation must not be anywhere near 10x the K=200 one.
    assert dev_large < 5 * dev_small + 1e-6, (
        f"ring median deviation grew from {dev_small:.4f} (K={K_small}) to "
        f"{dev_large:.4f} (K={K_large}); Proposition 1 predicts O(1), not growth"
    )

    W_identity = build_topology("identity", N)
    dev_small_id = verify_spreading(W_identity, DELTA, AGENT, N, ETA, K_small).median_deviation
    dev_large_id = verify_spreading(W_identity, DELTA, AGENT, N, ETA, K_large).median_deviation
    ratio = dev_large_id / max(dev_small_id, 1e-9)
    expected_ratio = K_large / K_small
    assert ratio == pytest.approx(expected_ratio, rel=1e-6), (
        "under W=I the concentrated attack's median deviation should grow "
        "linearly in K"
    )

    # The stealth comparison itself: at matched, large K the ring's
    # monitoring-visible deviation must be orders of magnitude below the
    # non-communicating control's.
    assert dev_large < dev_large_id / 10, (
        f"consensus (ring) median deviation {dev_large:.2f} is not "
        f"substantially smaller than the non-communicating control's "
        f"{dev_large_id:.2f}; the stealth mechanism did not manifest"
    )
