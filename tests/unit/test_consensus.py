"""Smoke test S4 / Phase 4 exit criterion: doubly stochastic construction."""

from __future__ import annotations

import numpy as np
import pytest

from safelie.consensus.mixing import assert_doubly_stochastic, second_largest_singular_value
from safelie.consensus.topologies import build_topology, list_topologies


@pytest.mark.parametrize("name", list_topologies())
@pytest.mark.parametrize("n", [2, 6, 12])
def test_every_topology_is_doubly_stochastic(name, n):
    kwargs = {"p": 0.5, "graph_seed": 1} if name == "erdos_renyi" else {}
    W = build_topology(name, n, **kwargs)
    assert_doubly_stochastic(W, tol=1e-12)


def test_identity_has_sigma2_equal_one():
    W = build_topology("identity", 6)
    assert second_largest_singular_value(W) == pytest.approx(1.0)


def test_complete_graph_mixes_fastest():
    """sigma_2 should be smallest for the complete graph among connected
    topologies of the same size — it is the fastest-mixing case."""
    n = 8
    sigma2 = {
        name: second_largest_singular_value(build_topology(name, n))
        for name in ["complete", "ring", "star"]
    }
    assert sigma2["complete"] <= sigma2["ring"]
    assert sigma2["complete"] <= sigma2["star"]


def test_shared_constraint_is_rank_one_averaging():
    n = 5
    W = build_topology("shared_constraint", n)
    np.testing.assert_allclose(W, np.full((n, n), 1.0 / n))


def test_non_doubly_stochastic_matrix_is_rejected():
    """A row-stochastic-but-not-column-stochastic matrix must be caught —
    this is the bug Theorem 1's proof is most sensitive to."""
    W = np.array([[0.5, 0.5], [0.9, 0.1]])  # row-stochastic, not column-stochastic
    with pytest.raises(AssertionError):
        assert_doubly_stochastic(W)


def test_erdos_renyi_is_reproducible_given_seed():
    W1 = build_topology("erdos_renyi", 10, p=0.4, graph_seed=5)
    W2 = build_topology("erdos_renyi", 10, p=0.4, graph_seed=5)
    np.testing.assert_array_equal(W1, W2)
