"""The M-source registry and independence-class accounting.

Report reference: PROJECT_REPORT.md Phase 2; §R10.3 (D10); §13.4 (W4,
"the most dangerous [weakness] in practice").

The paper's Theorem 2 requires M >= 2f+1 *reporting sources*. The report's
central engineering finding is that this is not the same as M >= 2f+1
*reports*: three ensemble replicas living inside one agent's process are
corrupted simultaneously by a single host compromise, so counting them as
three independent sources overstates robustness. `effective_M` counts
distinct **independence classes** (failure domains), which is the
quantity Theorem 2 is actually entitled to use, and
`safelie.utils.config.ExperimentConfig` rejects any configuration whose
claimed M >= 2f+1 holds only via same-class replicas (S12).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from safelie.utils.config import SourcesConfig, SourceSpec

SourceType = Literal["own_critic", "peer_critic", "ensemble_replica", "monitor"]


@dataclass(frozen=True)
class SourceReport:
    """One M-source's report for one constraint, one round.

    `value` is a return-scale estimate of J_C^i — see the return-scale
    warning in `safelie.defenses.base`.
    """

    constraint_owner: str
    source_id: str
    source_type: SourceType
    independence_class: str
    value: float
    round_k: int


def effective_M(reports_or_specs: list[SourceReport] | list[SourceSpec]) -> int:
    """Number of distinct independence classes — the M Theorem 2 may use.

    Always <= len(reports_or_specs). Works on either raw config specs
    (before any round has run) or realized reports (after collection),
    since both carry `independence_class`.
    """
    return len({item.independence_class for item in reports_or_specs})


class SourceRegistry:
    """Assembles the M reports for one constraint, one round.

    This is deliberately a thin, stateless-per-call wrapper: it does not
    itself compute cost estimates (that is the caller's job — an own
    critic, a peer critic, an ensemble member, or a monitor), it only
    assembles them with typed provenance and lets the caller (and the
    config validator) check `effective_M` honestly.
    """

    def __init__(self, sources_config: SourcesConfig):
        self.specs: list[SourceSpec] = sources_config.sources

    @property
    def M(self) -> int:
        return len(self.specs)

    @property
    def effective_M(self) -> int:
        return effective_M(self.specs)

    def collect(
        self,
        constraint_owner: str,
        round_k: int,
        value_fn: Callable[[SourceSpec], float],
    ) -> list[SourceReport]:
        """Build one SourceReport per configured source.

        `value_fn` maps a SourceSpec to its return-scale cost estimate for
        this round; the caller supplies it because the estimate depends on
        which network (own critic / peer critic / ensemble member /
        monitor) the spec designates, which this registry does not own.
        """
        return [
            SourceReport(
                constraint_owner=constraint_owner,
                source_id=spec.source_id,
                source_type=spec.source_type,
                independence_class=spec.independence_class,
                value=float(value_fn(spec)),
                round_k=round_k,
            )
            for spec in self.specs
        ]


def default_m7_sources() -> SourcesConfig:
    """The paper's primary M=7 configuration (App. B): own critic + 4 peer
    critics + 2 replicated monitors, one independence class per source
    except that peer critics on the *same* physical agent process would
    share a class in a real deployment. Here each peer sits on a distinct
    agent, so each gets its own class (the paper's intended reading,
    §R10.3)."""
    specs = [SourceSpec(source_id="own_critic", source_type="own_critic", independence_class="ic_own")]
    for i in range(1, 5):
        specs.append(
            SourceSpec(
                source_id=f"peer_critic_{i}",
                source_type="peer_critic",
                independence_class=f"ic_peer_{i}",
            )
        )
    for i in range(1, 3):
        specs.append(
            SourceSpec(
                source_id=f"monitor_{i}",
                source_type="monitor",
                independence_class=f"ic_monitor_{i}",
            )
        )
    return SourcesConfig(sources=specs)


def default_m5_sources() -> SourcesConfig:
    """The paper's N=2 configuration (App. B): 2 peer critics + 3 ensemble
    replicas. The report flags (W4) that 3 replicas from one process are
    NOT independent; we encode that honestly with a *shared* independence
    class for the replicas, which correctly drives effective_M down to 3
    (2 peers + 1 replica class) rather than the nominal 5. This is
    intentional: it demonstrates the accounting decision D10 the report
    recommends, not a bug."""
    specs = [
        SourceSpec(source_id="peer_critic_1", source_type="peer_critic", independence_class="ic_peer_1"),
        SourceSpec(source_id="peer_critic_2", source_type="peer_critic", independence_class="ic_peer_2"),
    ]
    for i in range(1, 4):
        specs.append(
            SourceSpec(
                source_id=f"ensemble_replica_{i}",
                source_type="ensemble_replica",
                independence_class="ic_ensemble_shared",  # correlated: one host compromise
            )
        )
    return SourcesConfig(sources=specs)
