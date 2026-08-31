"""Topology generators for the consensus network.

Report reference: Phase 4; PROJECT_REPORT.md §R3.1 (required topologies),
§4.4 Table 4 (`shared_constraint`), Proposition 1 (`identity` as the
non-communicating control, sigma_2 = 1).

`identity` and `shared_constraint` are not decoration: Definition/Prop 1
uses W = I as the concentrated-attack control, and W = (1/N) 1 1^T
recovers the shared-constraint case of §3.1 of the paper.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from safelie.consensus.mixing import metropolis_hastings_weights


def complete_topology(n: int) -> np.ndarray:
    adjacency = np.ones((n, n)) - np.eye(n)
    return metropolis_hastings_weights(adjacency)


def ring_topology(n: int) -> np.ndarray:
    if n < 3:
        # A ring needs >= 3 nodes to be a simple graph without duplicate edges.
        return complete_topology(n)
    adjacency = np.zeros((n, n))
    for i in range(n):
        adjacency[i, (i + 1) % n] = 1
        adjacency[i, (i - 1) % n] = 1
    return metropolis_hastings_weights(adjacency)


def star_topology(n: int) -> np.ndarray:
    adjacency = np.zeros((n, n))
    for i in range(1, n):
        adjacency[0, i] = 1
        adjacency[i, 0] = 1
    return metropolis_hastings_weights(adjacency)


def erdos_renyi_topology(n: int, p: float, seed: int = 0, max_tries: int = 200) -> np.ndarray:
    """Erdos-Renyi random graph, resampled until connected.

    Proposition 1 requires a connected, aperiodic graph (W^m e_j -> (1/N)1).
    A disconnected sample would make the mixing-rate bound meaningless, so
    we resample rather than silently proceed on a disconnected graph.
    """
    rng = np.random.default_rng(seed)
    for _attempt in range(max_tries):
        g = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(0, 2**31 - 1)))
        if nx.is_connected(g):
            adjacency = nx.to_numpy_array(g)
            return metropolis_hastings_weights(adjacency)
    raise RuntimeError(
        f"Could not sample a connected Erdos-Renyi graph with n={n}, p={p} "
        f"in {max_tries} tries; increase p."
    )


def identity_topology(n: int) -> np.ndarray:
    """W = I: the non-communicating control, sigma_2(W) = 1 (Proposition 1)."""
    return np.eye(n)


def shared_constraint_topology(n: int) -> np.ndarray:
    """W = (1/N) 1 1^T: recovers the shared-constraint case of paper §3.1."""
    return np.full((n, n), 1.0 / n)


_BUILDERS = {
    "complete": lambda n, **kw: complete_topology(n),
    "ring": lambda n, **kw: ring_topology(n),
    "star": lambda n, **kw: star_topology(n),
    "erdos_renyi": lambda n, **kw: erdos_renyi_topology(n, p=kw["p"], seed=kw.get("graph_seed", 0)),
    "identity": lambda n, **kw: identity_topology(n),
    "shared_constraint": lambda n, **kw: shared_constraint_topology(n),
}


def list_topologies() -> list[str]:
    return list(_BUILDERS)


def build_topology(name: str, n_agents: int, **kwargs) -> np.ndarray:
    if name not in _BUILDERS:
        raise ValueError(f"Unknown topology '{name}'. Available: {list_topologies()}")
    return _BUILDERS[name](n_agents, **kwargs)
