"""The dual-cost environment contract.

Report reference: PROJECT_REPORT.md Phase 1 — "the single most important
interface in the repository." Every environment used by this project must
conform to `DualCostEnvWrapper`, whether it is the synthetic CPU
environment (`safelie.envs.synthetic`) or a real Safe MAMuJoCo / Safety-
Gymnasium adapter (`safelie.envs.mamujoco`, not implemented — see that
module's docstring and docs/paper_implementation_mapping.md).

`DualCostStep` deliberately has **no** `true_cost` field (see
`safelie.envs.guards` for why this is a strengthening of the report's own
suggested contract). `reported_cost` is what the learner's cost critic
regresses against; it is numerically identical, per step, to what the
environment's internal true-cost stream produces — the environment is not
itself adversarial. The divergence the paper studies (`J_true_C` vs.
`J_reported_C`) arises entirely downstream: from critic estimation error
and from the attack module corrupting the return-scale residual after
critic evaluation and before consensus (main_iclr.tex App. B; Phase 5).
This is a `[DECISION]` resolving an ambiguity in the report's Phase-1
sketch; see docs/assumptions.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from safelie.envs.guards import OracleReadOnlyView

AgentID = str


@dataclass(frozen=True)
class DualCostStep:
    obs: dict[AgentID, np.ndarray]
    reward: float  # shared task reward R(s,a) -- Definition 1
    reported_cost: dict[AgentID, float]  # per-agent per-step cost, learner-visible
    terminated: dict[AgentID, bool]
    truncated: dict[AgentID, bool]
    info: dict


class DualCostEnvWrapper(Protocol):
    """Any environment used by this project implements this contract."""

    n_agents: int
    budget: float  # d^i, assumed equal across agents (paper's ManyAgent Ant setting)
    agent_ids: list[AgentID]

    def reset(self, seed: int | None = None) -> DualCostStep: ...

    def step(self, actions: dict[AgentID, np.ndarray]) -> DualCostStep: ...

    def oracle_handle(self) -> OracleReadOnlyView:
        """Returns a handle for reading true cost. Callable by anyone, but
        only ever *works* when called by `safelie.eval.oracle.OracleEvaluator`
        via the privileged constructor path (see `safelie.envs.guards`)."""
        ...
