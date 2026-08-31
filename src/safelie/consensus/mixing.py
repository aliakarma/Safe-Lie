"""Doubly stochastic mixing matrix construction and diagnostics.

Report reference: Phase 4; Theorem 1 (PROJECT_REPORT.md §6.4); smoke test S4.

Theorem 1 requires $W\\mathbf{1} = \\mathbf{1}$ *and* $\\mathbf{1}^\\top W =
\\mathbf{1}^\\top$ (row- and column-stochastic). A row-stochastic-but-not-
column-stochastic matrix silently breaks the theorem's proof (which relies
on $\\mathbf{1}^\\top W^m = \\mathbf{1}^\\top$ for all $m$) while looking like
a perfectly normal consensus matrix. The report calls this the most likely
implementation bug in the entire theory suite, so both conditions are
asserted, never just one.
"""

from __future__ import annotations

import numpy as np


def assert_doubly_stochastic(W: np.ndarray, tol: float = 1e-10) -> None:
    """Raise ``AssertionError`` unless W is doubly stochastic to `tol`.

    Checks row sums, column sums, and non-negativity (a stochastic matrix
    with negative entries can still sum to one and would silently violate
    the probabilistic interpretation Proposition 1's mixing-rate argument
    relies on).
    """
    n = W.shape[0]
    assert W.shape == (n, n), f"W must be square, got {W.shape}"
    assert np.all(W >= -tol), "W has negative entries; not a valid mixing matrix"
    row_sums = W.sum(axis=1)
    col_sums = W.sum(axis=0)
    assert np.allclose(row_sums, 1.0, atol=tol), (
        f"W is not row-stochastic: row sums range [{row_sums.min():.3e}, "
        f"{row_sums.max():.3e}]"
    )
    assert np.allclose(col_sums, 1.0, atol=tol), (
        f"W is not column-stochastic (this is the bug Theorem 1's proof is "
        f"most sensitive to): col sums range [{col_sums.min():.3e}, "
        f"{col_sums.max():.3e}]"
    )


def second_largest_singular_value(W: np.ndarray) -> float:
    """sigma_2(W), the mixing rate governing Proposition 1's residual bound."""
    singular_values = np.linalg.svd(W, compute_uv=False)
    if len(singular_values) < 2:
        return 0.0
    return float(singular_values[1])


def metropolis_hastings_weights(adjacency: np.ndarray) -> np.ndarray:
    """Standard doubly stochastic construction for an undirected graph.

    W_ij = 1 / (1 + max(deg_i, deg_j)) for an edge (i,j), and
    W_ii = 1 - sum_j W_ij. This is the textbook Metropolis-Hastings weight
    choice for gossip/consensus algorithms and is symmetric, hence
    automatically doubly stochastic.
    """
    n = adjacency.shape[0]
    assert adjacency.shape == (n, n)
    assert np.array_equal(adjacency, adjacency.T), "adjacency must be symmetric (undirected graph)"
    deg = adjacency.sum(axis=1)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and adjacency[i, j] > 0:
                W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
        W[i, i] = 1.0 - W[i, :].sum()
    return W
