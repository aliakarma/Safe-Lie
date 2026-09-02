"""The withheld oracle evaluator.

Report reference: main_iclr.tex §5.1 ("true cost, computed by an
evaluator with access to the uncorrupted simulator signal and withheld
from the learner"); PROJECT_REPORT.md §4.4, Phase 8; smoke test S10.

`OracleEvaluator` is the ONLY code in this repository allowed to call
`env._oracle_handle_privileged()`. It accumulates the discounted true
cost return over an episode, exactly analogous to `J_C^i(theta)`
(Eq. 1 of the paper) but computed from the environment's privileged
true-cost stream rather than from any critic. The isolation test
(`tests/isolation/test_oracle_isolation.py`) asserts no training or
algorithm module imports this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from safelie.envs.dual_cost import AgentID, DualCostEnvWrapper
from safelie.envs.guards import OracleReadOnlyView


class OracleCapableEnv(DualCostEnvWrapper, Protocol):
    """The environment contract as seen by the evaluator only: everything
    in `DualCostEnvWrapper`, plus the privileged handle constructor. This
    is deliberately a *separate*, wider protocol from the one the learner
    is typed against (`safelie.training` never imports this module — see
    `tests/isolation/test_oracle_isolation.py`), so the isolation boundary
    is visible in the type signatures, not only in the docstrings."""

    def _oracle_handle_privileged(self) -> OracleReadOnlyView: ...


@dataclass
class OracleEvaluator:
    """Accumulates the discounted true cost return per agent over an
    episode. Constructed once per evaluation episode; call `record` after
    every environment step, `episode_result` at episode end."""

    env: OracleCapableEnv
    gamma: float
    budget: float
    _t: int = field(default=0, init=False)
    _discounted_true_cost: dict[AgentID, float] = field(default_factory=dict, init=False)
    _peak_true_cost: dict[AgentID, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._handle = self.env._oracle_handle_privileged()  # the ONE privileged call site
        self._discounted_true_cost = {aid: 0.0 for aid in self.env.agent_ids}
        self._peak_true_cost = {aid: 0.0 for aid in self.env.agent_ids}
        self._t = 0

    def record(self) -> dict[AgentID, float]:
        """Call once per environment step, AFTER `env.step(...)`."""
        true_cost = self._handle.true_cost()
        for aid, c in true_cost.items():
            self._discounted_true_cost[aid] += (self.gamma**self._t) * c
            self._peak_true_cost[aid] = max(self._peak_true_cost[aid], c)
        self._t += 1
        return true_cost

    def episode_result(self) -> OracleEpisodeResult:
        violated = {aid: (v > self.budget) for aid, v in self._discounted_true_cost.items()}
        return OracleEpisodeResult(
            true_cost_return=dict(self._discounted_true_cost),
            peak_true_cost=dict(self._peak_true_cost),
            violated=violated,
            episode_length=self._t,
        )


@dataclass(frozen=True)
class OracleEpisodeResult:
    """P0 #6: this is the complete set of **evaluation** quantities, all
    computed as proper episodic discounted Monte-Carlo sums over one
    fresh, independent, privileged-free-of-training-noise oracle rollout
    -- never a GAE(lambda) training target and never a quantity carried
    over from the learner's own on-policy rollout. `episode_length` fixes
    what "episode" means here: exactly `rollout_length` consecutive steps
    from one `env.reset()` call, matching the training loop's own
    definition of a round -- even on backends (Safe MAMuJoCo) that
    auto-reset internally on termination mid-window, so "episode" always
    denotes this fixed window, never a termination-to-termination segment.

    `true_cost_return` is privileged (computed via
    `env._oracle_handle_privileged()`, `OracleEvaluator` only).
    `episodic_task_return` and `episodic_reported_cost_return` are not --
    they are populated by `safelie.eval.harness.evaluate_true_cost` from
    the same rollout's ordinary, learner-visible `DualCostStep.reward` /
    `.reported_cost`, using the identical discounting convention. They
    default to 0.0/{} here because `OracleEvaluator` (privileged-path-only
    by design, see this module's docstring) does not compute them; a
    result missing them is a bug in the caller, not a valid partial
    result -- `evaluate_true_cost` always fills them in before returning.
    """

    true_cost_return: dict[AgentID, float]  # J_true_C per agent, this episode
    peak_true_cost: dict[AgentID, float]  # max per-step true cost, this episode
    violated: dict[AgentID, bool]  # J_true_C > d, per agent
    episode_length: int
    episodic_task_return: float = 0.0  # sum_t gamma^t * reward_t (shared across agents, Definition 1)
    episodic_reported_cost_return: dict[AgentID, float] = field(default_factory=dict)
