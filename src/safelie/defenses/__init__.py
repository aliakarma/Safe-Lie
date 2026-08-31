"""The aggregator zoo — one interface, five aggregators (Table 4).

Report reference: PROJECT_REPORT.md Phase 6. `mean` and `rce` are the
compact-study minimum (decision D2); `coordinate_median`, `krum`, and
`trimmean` are Stage-3 ablation aggregators, implemented so the ablation
in Table 4 is a config change, not a code change.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from safelie.defenses.base import AggregateResult
from safelie.defenses.krum import krum_aggregator
from safelie.defenses.mean import mean_aggregator
from safelie.defenses.median import coordinate_median_aggregator
from safelie.defenses.rce import RceResult, rce_aggregate
from safelie.defenses.trimmean import trimmed_mean_aggregator

_REGISTRY: dict[str, Callable[..., AggregateResult]] = {
    "mean": mean_aggregator,
    "coordinate_median": coordinate_median_aggregator,
    "krum": krum_aggregator,
    "trimmean": trimmed_mean_aggregator,
}


def get_aggregator(name: str) -> Callable[..., AggregateResult]:
    if name == "rce":
        raise ValueError("Use safelie.defenses.rce.rce_aggregate directly; it needs beta.")
    if name not in _REGISTRY:
        raise ValueError(f"Unknown aggregator '{name}'. Available: {list(_REGISTRY) + ['rce']}")
    return _REGISTRY[name]


def aggregate(name: str, values: np.ndarray, f: int, beta: float = 1.5, **kwargs) -> AggregateResult:
    """Single dispatch point used by the training loop and by tests."""
    if name == "rce":
        return rce_aggregate(values, f, beta, **kwargs)
    return get_aggregator(name)(values, f)


__all__ = [
    "AggregateResult",
    "RceResult",
    "get_aggregator",
    "aggregate",
    "mean_aggregator",
    "coordinate_median_aggregator",
    "krum_aggregator",
    "trimmed_mean_aggregator",
    "rce_aggregate",
]
