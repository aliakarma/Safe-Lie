"""Oracle isolation: the learner must never be able to read true cost.

Report reference: PROJECT_REPORT.md §4.4 ("the oracle must obtain true
cost through a handle the learner process cannot write to. If the oracle
reads cost from the same object the learner can mutate, the detection gap
metric is meaningless and the entire experimental result is void") and
smoke test S10.

`[DECISION]` This repository strengthens the report's own suggested
contract. The report's Phase-1 sketch puts a `true_cost` field directly
on the per-step `DualCostStep` object and relies on a runtime guard to
stop the learner reading it. We instead never put true cost on the
object the learner receives at all (see `safelie.envs.dual_cost`) — the
leak vector the report warns about ("if the learner can reach true_cost
through any path... invalid") is removed structurally, not merely
guarded at the point of access. The guard below is the second,
belt-and-braces layer: even `env.oracle_handle()`, if a learner module
were to call it directly, returns a handle that raises rather than a
working one.
"""

from __future__ import annotations

from collections.abc import Callable


class LearnerAccessError(PermissionError):
    """Raised whenever code without the oracle's privilege tries to read
    true cost. A test catching anything other than this exact exception
    (or forgetting to catch it at all) has a broken isolation boundary."""


class OracleReadOnlyView:
    """The handle `env.oracle_handle()` returns to ordinary (learner-side)
    callers. Calling `.true_cost()` on it always raises.

    Only `safelie.eval.oracle.OracleEvaluator` obtains a working view, via
    the environment's `_oracle_handle_privileged()` — a name that is not
    part of the public `DualCostEnvWrapper` protocol and that the
    isolation test (`tests/isolation/test_oracle_isolation.py`) asserts no
    training or algorithm module ever calls.
    """

    def __init__(self, get_true_cost: Callable[[], dict[str, float]]):
        self._get_true_cost = get_true_cost

    def true_cost(self) -> dict[str, float]:
        raise LearnerAccessError(
            "true_cost() is not accessible from a bare oracle_handle(). "
            "Only safelie.eval.oracle.OracleEvaluator may read true cost "
            "(PROJECT_REPORT.md §4.4; smoke test S10)."
        )


class _PrivilegedOracleView(OracleReadOnlyView):
    """The working view. Constructed only inside
    `DualCostEnvWrapper._oracle_handle_privileged`."""

    def true_cost(self) -> dict[str, float]:
        return dict(self._get_true_cost())
