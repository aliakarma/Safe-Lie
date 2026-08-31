"""Structured, validated experiment configuration.

Report reference: Phase 0; smoke tests S2, S5, S12; decisions D3, D10, D12, D14.

The report recommends Hydra + structured configs. This repository uses
Pydantic v2 models over YAML instead: the project does not run Hydra-style
multi-run sweeps (compute is far too limited for the paper's 300+ run
grid, see PROJECT_REPORT.md §10.2), so Hydra's sweep machinery buys
nothing, while Pydantic gives the same fail-fast, schema-validated
construction with a much smaller dependency footprint. This is a
`[DECISION]` documented in docs/assumptions.md.

Every cross-field rule the report calls CRITICAL is enforced here, at
construction time, not at aggregation time:

- `M <= 2f` for a trimmed-mean-family defense raises (S5, D3).
- `effective_M` (independence classes, not raw report count) is what is
  checked against `2f + 1`, so a config that only "looks" robust by
  counting correlated replicas is rejected (S12, D10, W4).
- Topology agent count must match the environment agent count.
"""

from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class SourceSpec(BaseModel):
    source_id: str
    source_type: Literal["own_critic", "peer_critic", "ensemble_replica", "monitor"]
    independence_class: str


class SourcesConfig(BaseModel):
    sources: list[SourceSpec] = Field(min_length=1)

    @property
    def M(self) -> int:
        return len(self.sources)

    @property
    def effective_M(self) -> int:
        """Distinct independence classes — the M Theorem 2 is entitled to use (§R10.3)."""
        return len({s.independence_class for s in self.sources})


class TopologyConfig(BaseModel):
    name: Literal["complete", "ring", "star", "erdos_renyi", "identity", "shared_constraint"]
    n_agents: int = Field(gt=0)
    p: float | None = Field(default=None, ge=0.0, le=1.0)
    graph_seed: int = 0

    @model_validator(mode="after")
    def _erdos_renyi_needs_p(self) -> TopologyConfig:
        if self.name == "erdos_renyi" and self.p is None:
            raise ValueError("topology.p is required when topology.name == 'erdos_renyi'")
        return self


class EnvConfig(BaseModel):
    name: Literal["synthetic_constrained_marl", "manyagent_ant", "halfcheetah_2x3", "safety_gym_nav"]
    n_agents: int = Field(gt=0)
    budget: float = Field(gt=0)
    horizon: int = Field(default=200, gt=0)
    obs_dim: int = Field(default=8, gt=0)
    action_dim: int = Field(default=2, gt=0)


class AttackConfig(BaseModel):
    name: Literal["none", "primary", "stealth", "benign_control"] = "none"
    f: int = Field(default=0, ge=0)
    budget_ratio: float = Field(default=0.0, ge=0.0)  # B/d, [SPEC] {0, 0.25, 0.5, 1.0}
    kappa: float = Field(default=1.0, gt=0.0)  # [GAP] G7, no paper value
    nu_ratio: float = Field(default=0.05, ge=0.0)  # [GAP] G7, no paper value
    direction: Literal["negative", "positive", "mixed"] = "negative"
    support: Literal["persistent", "selective"] = "persistent"
    adaptivity: Literal["static", "adaptive"] = "static"
    consistency: Literal["consistent", "byzantine"] = "consistent"


class DefenseConfig(BaseModel):
    name: Literal["mean", "coordinate_median", "krum", "trimmean", "rce"] = "mean"
    f: int = Field(default=0, ge=0)
    beta: float = Field(default=1.5, ge=0.0)  # [SPEC] margin coefficient
    sigma_min: float = Field(default=1e-3, ge=0.0)  # floor on degenerate spread, §R10.2
    min_retained: int = Field(default=3, ge=1)  # |T| < this floors + warns, §R10.2
    use_reliability_weights: bool = False  # [GAP] G1, shipped OFF by decision D12


class PPOConfig(BaseModel):
    clip: float = 0.2  # [SPEC]
    gae_lambda: float = 0.95  # [SPEC]
    gamma: float = 0.99  # [SPEC]
    lr: float = 3e-4  # [SPEC]
    epochs: int = 4  # [GAP] G10, pinned per D14
    minibatches: int = 4  # [GAP] G10, pinned per D14
    entropy_coef: float = 0.0  # [GAP] G11, pinned per D14
    value_coef: float = 0.5  # [GAP] G11, pinned per D14
    grad_clip: float = 0.5  # [GAP] G11, pinned per D14
    hidden_dim: int = 64  # [GAP] G12, pinned per D14


class DualConfig(BaseModel):
    eta_lambda: float = 0.035  # [SPEC]
    lambda_max: float = 25.0  # [SPEC]


class ExperimentConfig(BaseModel):
    run_id: str
    seed: int = 0
    env: EnvConfig
    topology: TopologyConfig
    sources: SourcesConfig
    attack: AttackConfig = AttackConfig()
    defense: DefenseConfig = DefenseConfig()
    ppo: PPOConfig = PPOConfig()
    dual: DualConfig = DualConfig()
    total_steps: int = Field(gt=0)
    rollout_length: int = Field(default=200, gt=0)
    output_dir: str = "results/runs"

    @model_validator(mode="after")
    def _cross_field_checks(self) -> ExperimentConfig:
        if self.topology.n_agents != self.env.n_agents:
            raise ValueError(
                f"topology.n_agents ({self.topology.n_agents}) must equal "
                f"env.n_agents ({self.env.n_agents})"
            )

        f = self.defense.f
        m_eff = self.sources.effective_M
        if self.defense.name in ("trimmean", "rce"):
            if m_eff <= 2 * f:
                raise ValueError(
                    f"Defense '{self.defense.name}' requires effective_M > 2f "
                    f"(distinct independence classes, not raw report count), but "
                    f"effective_M={m_eff} and f={f} give M-2f={m_eff - 2 * f} <= 0. "
                    "Trimmed mean is mathematically undefined at this operating "
                    "point (PROJECT_REPORT.md §13.2, W2). Raising rather than "
                    "clamping or falling back, per decision D3."
                )
        if self.attack.f > self.sources.M:
            raise ValueError(
                f"attack.f ({self.attack.f}) cannot exceed the number of "
                f"sources M ({self.sources.M})"
            )
        return self


def load_experiment_config(path: str) -> ExperimentConfig:
    """Load and validate a YAML experiment config.

    Raises a Pydantic ``ValidationError`` (a ``ValueError`` subclass) on any
    malformed or scientifically-invalid config, before any environment,
    policy, or source is constructed. This is smoke test S2.
    """
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ExperimentConfig.model_validate(raw)
