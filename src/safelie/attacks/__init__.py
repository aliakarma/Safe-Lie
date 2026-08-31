"""Single dispatch point for the attack module.

Report reference: PROJECT_REPORT.md Phase 5 — "one clean hook, one
ledger. Do not scatter corruption logic across the critic and the
environment."
"""

from __future__ import annotations

import numpy as np

from safelie.attacks.adaptive import stealth_attack
from safelie.attacks.ledger import AttackLedger, LedgerEntry
from safelie.attacks.static import benign_control, static_attack
from safelie.attacks.taxonomy import OVER_REPORT_ATTACK, PRIMARY_ATTACK, STEALTH_ATTACK, TaxonomyPoint
from safelie.utils.config import AttackConfig


def apply_attack(
    cfg: AttackConfig,
    residuals: dict[str, float],
    corrupted: set[str],
    k: int,
    d: float,
    rng: np.random.Generator | None = None,
    ledger: AttackLedger | None = None,
) -> dict[str, float]:
    """Apply the configured attack at the report-injection hook and record
    the ground-truth ledger entry (S19)."""
    if cfg.name == "none" or not corrupted:
        output = dict(residuals)
    elif cfg.name == "primary":
        B = cfg.budget_ratio * d
        output = static_attack(
            residuals, corrupted, k, B, direction=cfg.direction, support=cfg.support
        )
    elif cfg.name == "stealth":
        B = cfg.budget_ratio * d
        nu = cfg.nu_ratio * d
        output = stealth_attack(residuals, corrupted, k, B, kappa=cfg.kappa, nu=nu, d=d)
    elif cfg.name == "benign_control":
        if rng is None:
            raise ValueError("benign_control requires an rng")
        sigma_equiv = cfg.budget_ratio * d
        output = benign_control(residuals, corrupted, k, sigma_equiv, rng)
    else:
        raise ValueError(f"Unknown attack '{cfg.name}'")

    if ledger is not None:
        ledger.record(k, residuals, output)
    return output


__all__ = [
    "apply_attack",
    "AttackLedger",
    "LedgerEntry",
    "TaxonomyPoint",
    "PRIMARY_ATTACK",
    "STEALTH_ATTACK",
    "OVER_REPORT_ATTACK",
    "static_attack",
    "stealth_attack",
    "benign_control",
]
