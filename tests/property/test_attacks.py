"""Attack mechanism smoke tests S16-S20.

Report reference: PROJECT_REPORT.md §R4.2.
"""

from __future__ import annotations

import numpy as np
import pytest
from safelie.attacks import AttackLedger, apply_attack
from safelie.utils.config import AttackConfig

D = 25.0


def _residuals(n_sources: int = 7, jhat: float = 24.0) -> dict[str, float]:
    return {f"s{i}": (jhat - D) for i in range(n_sources)}


# S16 — under-reporting lowers the safety signal, every round
def test_under_reporting_strictly_lowers_residual_every_round():
    residuals = _residuals()
    corrupted = {"s0"}
    cfg = AttackConfig(name="primary", f=1, budget_ratio=0.5, direction="negative")
    B = cfg.budget_ratio * D
    for k in range(20):
        out = apply_attack(cfg, residuals, corrupted, k, D)
        assert out["s0"] == pytest.approx(residuals["s0"] - B)
        for sid in residuals:
            if sid != "s0":
                assert out[sid] == residuals[sid]


# S17 — multiplier direction / dose-response
def test_attack_magnitude_is_monotone_in_delta():
    residuals = _residuals()
    corrupted = {"s0"}
    deltas = []
    for ratio in [0.0, 0.25, 0.5, 1.0]:
        cfg = AttackConfig(name="primary", f=1, budget_ratio=ratio, direction="negative")
        out = apply_attack(cfg, residuals, corrupted, 0, D)
        deltas.append(residuals["s0"] - out["s0"])  # magnitude of the drop
    assert deltas == sorted(deltas)
    assert deltas[0] == 0.0


# S18 — reported and true cost can diverge (toy mechanism check)
def test_lower_multiplier_mechanically_permits_higher_true_cost():
    """Toy loop: if the residual fed to the dual update is depressed,
    a smaller lambda results, which (in a toy environment where cost is
    a decreasing function of lambda) permits higher true cost while the
    corrupted report stays <= d. This is a plumbing check, not evidence
    about MAPPO-Lagrangian."""
    eta_lambda, lam_max, B = 0.035, 25.0, 12.5
    jhat = 27.0  # slightly above budget, so the *clean* residual is
    # positive and lambda actually rises under no attack -- the
    # discriminating case (a residual that starts negative never lifts
    # lambda off zero in either arm and the test is vacuous).
    lam_clean = lam_attacked = 0.0
    for k in range(200):
        residuals_clean = {"s0": jhat - D}
        residuals_attacked = apply_attack(
            AttackConfig(name="primary", f=1, budget_ratio=0.5), residuals_clean, {"s0"}, k, D
        )
        lam_clean = np.clip(lam_clean + eta_lambda * residuals_clean["s0"], 0.0, lam_max)
        lam_attacked = np.clip(lam_attacked + eta_lambda * residuals_attacked["s0"], 0.0, lam_max)

    assert lam_attacked < lam_clean

    # The reported cost an evaluator would see is jhat shifted by the
    # applied delta (-B, persistent): this is what the learner sees and
    # what standard monitoring observes.
    reported_attacked = jhat - B
    reported_clean = jhat

    # Toy true-cost model: true_cost = base + gain * (lam_max - lam) -- a
    # smaller multiplier mechanically permits a higher true cost. This
    # model exists only to demonstrate the pathway, not to claim anything
    # about MAPPO-Lagrangian's actual cost dynamics.
    true_cost_clean = 20.0 + 0.4 * (lam_max - lam_clean)
    true_cost_attacked = 20.0 + 0.4 * (lam_max - lam_attacked)

    assert reported_attacked <= D  # reported stays acceptable (the stealth)
    assert true_cost_attacked > D  # true violation (the harm)
    assert true_cost_attacked > true_cost_clean
    assert reported_clean > 0  # sanity: clean report is a real number, not a stub


# S19 — the ledger reconstructs sum_k 1^T delta_k exactly
def test_ledger_reconstructs_total_mass_exactly():
    residuals = _residuals()
    corrupted = {"s0", "s2"}
    cfg = AttackConfig(name="primary", f=2, budget_ratio=0.5, direction="negative")
    ledger = AttackLedger()
    B = cfg.budget_ratio * D
    K = 50
    for k in range(K):
        apply_attack(cfg, residuals, corrupted, k, D, ledger=ledger)

    expected_total = -B * K * len(corrupted)
    assert ledger.total_mass() == pytest.approx(expected_total, abs=1e-12)
    assert ledger.corrupted_source_ids() == corrupted
    per_source = ledger.total_mass_per_source()
    for sid in corrupted:
        assert per_source[sid] == pytest.approx(-B * K, abs=1e-12)


# S20 — benign control is unbiased
def test_benign_control_is_unbiased_and_magnitude_matched():
    rng = np.random.default_rng(0)
    residuals = _residuals()
    corrupted = {"s0"}
    cfg = AttackConfig(name="benign_control", f=1, budget_ratio=0.5)
    deltas = []
    for k in range(10_000):
        out = apply_attack(cfg, residuals, corrupted, k, D, rng=rng)
        deltas.append(out["s0"] - residuals["s0"])
    deltas = np.array(deltas)
    # sample mean indistinguishable from zero: within a few standard errors
    sigma_equiv = cfg.budget_ratio * D
    standard_error = sigma_equiv / np.sqrt(len(deltas))
    assert abs(deltas.mean()) < 4 * standard_error
    assert deltas.std() == pytest.approx(sigma_equiv, rel=0.1)
