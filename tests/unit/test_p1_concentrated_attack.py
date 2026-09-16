"""P1: the owner-localized (concentrated) corruption.

docs/p1_concentrated_attack_gates.md, gates P1-a .. P1-e.

The load-bearing test in this file is
`test_injected_delta_is_concentrated_on_one_owner`: it asserts that the
per-owner perturbation vector is approximately `[0, ..., -B/M, ..., 0]`
and NOT `[-B/M, ..., -B/M]`. Every other test here exists to stop that
one from passing for the wrong reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from safelie.consensus.mixing import assert_doubly_stochastic, second_largest_singular_value
from safelie.consensus.topologies import build_topology
from safelie.training.loop import ExperimentRun
from safelie.utils.config import ExperimentConfig

N_AGENTS = 4
BUDGET = 25.0
BUDGET_RATIO = 0.5
M = 3


def _cfg(tmp_path, *, owners, run_id, topology="ring", seed=0) -> ExperimentConfig:
    """A tiny synthetic config with M=3 neural sources and mean/f=0
    aggregation -- the A1 aggregator, so the predicted per-owner shift is
    exactly -B/M and can be asserted as an equality rather than a bound."""
    attack = {
        "name": "primary",
        "f": 1,
        "budget_ratio": BUDGET_RATIO,
        "direction": "negative",
        "support": "persistent",
        "adaptivity": "static",
        "consistency": "consistent",
        "corrupted_source_ids": ["peer_critic_1"],
    }
    if owners is not None:
        attack["corrupted_owner_ids"] = owners
    return ExperimentConfig(
        run_id=run_id,
        seed=seed,
        env={"name": "synthetic_constrained_marl", "n_agents": N_AGENTS,
             "budget": BUDGET, "horizon": 12, "obs_dim": 6, "action_dim": 2},
        topology={"name": topology, "n_agents": N_AGENTS},
        sources={"sources": [
            {"source_id": "own_critic", "source_type": "own_critic", "independence_class": "ic_own"},
            {"source_id": "peer_critic_1", "source_type": "peer_critic", "independence_class": "ic_peer1"},
            {"source_id": "monitor_1", "source_type": "monitor", "independence_class": "ic_mon1"},
        ]},
        attack=attack,
        defense={"name": "mean", "f": 0},
        ppo={"epochs": 1, "minibatches": 2, "hidden_dim": 16},
        total_steps=24, rollout_length=12,
        output_dir=str(tmp_path / "runs"),
    )


def _delta_vector(record: dict) -> np.ndarray:
    aids = sorted(record["constraints"])
    return np.array([record["constraints"][a]["injected_delta"] for a in aids])


# ---------------------------------------------------------------- P1-a ----
def test_injected_delta_is_concentrated_on_one_owner(tmp_path):
    """delta_k ~ c * e_j, NOT c * 1. This is the whole point of P1."""
    run = ExperimentRun(_cfg(tmp_path, owners=["agent_1"], run_id="conc"))
    try:
        for _ in range(2):
            rec = run.run_round()
            delta = _delta_vector(rec)
            expected = np.zeros(N_AGENTS)
            expected[1] = -BUDGET_RATIO * BUDGET / M
            np.testing.assert_allclose(delta, expected, atol=1e-9)
            # Stated the other way round too, so the test fails loudly if
            # someone reintroduces the uniform perturbation.
            assert not np.allclose(delta, np.full(N_AGENTS, delta[1]), atol=1e-9)
            nz = np.flatnonzero(np.abs(delta) > 1e-12)
            assert nz.tolist() == [1], f"perturbation touched owners {nz.tolist()}"
    finally:
        run.close()


def test_untargeted_owners_are_bitwise_unshifted(tmp_path):
    """Every owner but j must receive its uncorrupted aggregate exactly.

    Asserted as `post-attack aggregate == the SAME aggregator's estimate on
    the uncorrupted reports`, at exactly zero tolerance. Comparing against a
    `np.mean` recomputed in the test instead would fail in the last ULP on a
    correct implementation -- floating-point summation is not associative, so
    the reduction order has to be the aggregator's own, not a transcription
    of it.
    """
    run = ExperimentRun(_cfg(tmp_path, owners=["agent_1"], run_id="unshift"))
    try:
        for _ in range(2):
            rec = run.run_round()
            for aid, blk in rec["constraints"].items():
                clean = blk["clean_point_estimate"]
                got = blk["aggregate"]["point_estimate"]
                # The counterfactual must itself be right, independently of
                # the equality below: a bug that made BOTH estimates equal
                # to the same wrong number would otherwise pass.
                assert clean == pytest.approx(
                    float(np.mean([r["value"] for r in blk["reports"]])), rel=1e-12
                )
                if aid == "agent_1":
                    assert got < clean - 1e-9
                    assert blk["corrupted_source_ids"] == ["peer_critic_1"]
                    assert blk["owner_targeted"] is True
                else:
                    assert got == clean, f"{aid} was shifted by {got - clean!r}"
                    assert blk["injected_delta"] == 0.0
                    assert blk["corrupted_source_ids"] == []
                    assert blk["owner_targeted"] is False
    finally:
        run.close()


def test_two_named_owners_give_two_nonzero_coordinates(tmp_path):
    """The field scopes to the set it names -- not hardcoded to one owner."""
    run = ExperimentRun(_cfg(tmp_path, owners=["agent_0", "agent_2"], run_id="two"))
    try:
        delta = _delta_vector(run.run_round())
        assert np.flatnonzero(np.abs(delta) > 1e-12).tolist() == [0, 2]
    finally:
        run.close()


# ---------------------------------------------------------------- P1-b ----
def test_omitting_owner_ids_reproduces_the_uniform_A1_perturbation(tmp_path):
    """The default path is the historical one: delta_k = -(B/M) * 1.

    This is the behavioural half of the A1 no-change argument -- a config
    that does not mention `corrupted_owner_ids` must still corrupt every
    owner, which is what A1/A2/A3 ran.
    """
    run = ExperimentRun(_cfg(tmp_path, owners=None, run_id="uniform"))
    try:
        delta = _delta_vector(run.run_round())
        np.testing.assert_allclose(
            delta, np.full(N_AGENTS, -BUDGET_RATIO * BUDGET / M), atol=1e-9
        )
    finally:
        run.close()


def test_default_path_is_unchanged_by_the_new_field(tmp_path):
    """Two runs, identical but for the *absence* of the new field, must be
    bitwise identical in every pre-existing logged quantity."""
    keys = ("point_estimate", "spread", "retained_n")
    out = []
    for tag in ("a", "b"):
        run = ExperimentRun(_cfg(tmp_path, owners=None, run_id=f"reg_{tag}"))
        try:
            recs = [run.run_round() for _ in range(2)]
        finally:
            run.close()
        out.append([
            [tuple(blk["aggregate"][k] for k in keys), blk["lambda_after"], blk["constraint_residual"]]
            for r in recs for blk in (r["constraints"][a] for a in sorted(r["constraints"]))
        ])
    for ra, rb in zip(*out, strict=True):
        assert ra[0] == rb[0]
        assert ra[1] == rb[1] and ra[2] == rb[2]


# ---------------------------------------------------------------- P1-c ----
def test_identity_topology_is_exactly_I_and_does_not_mix(tmp_path):
    W = build_topology("identity", N_AGENTS)
    np.testing.assert_array_equal(W, np.eye(N_AGENTS))
    assert_doubly_stochastic(W)
    assert second_largest_singular_value(W) == pytest.approx(1.0)
    # The property that matters: a concentrated vector stays concentrated
    # under any number of applications of W.
    v = np.zeros(N_AGENTS)
    v[2] = -4.0
    for _ in range(50):
        v = W @ v
    assert np.flatnonzero(np.abs(v) > 1e-12).tolist() == [2]


def test_ring_topology_is_connected_doubly_stochastic_and_mixes():
    W = build_topology("ring", 6)
    assert_doubly_stochastic(W)
    s2 = second_largest_singular_value(W)
    assert s2 < 1.0, "ring must have a spectral gap or Proposition cor:spread is vacuous"
    v = np.zeros(6)
    v[0] = -4.0
    for _ in range(500):
        v = W @ v
    np.testing.assert_allclose(v, np.full(6, -4.0 / 6), atol=1e-9)


def test_run_uses_the_configured_topology(tmp_path):
    for name, expect in (("identity", np.eye(N_AGENTS)), ("ring", build_topology("ring", N_AGENTS))):
        run = ExperimentRun(_cfg(tmp_path, owners=["agent_1"], run_id=f"top_{name}", topology=name))
        try:
            np.testing.assert_array_equal(run.W, expect)
        finally:
            run.close()


# ---------------------------------------------------------------- P1-d ----
def test_config_rejects_unknown_owner(tmp_path):
    with pytest.raises(ValueError, match="do not exist"):
        _cfg(tmp_path, owners=["agent_99"], run_id="bad")


def test_config_rejects_empty_owner_list(tmp_path):
    with pytest.raises(ValueError, match="empty list"):
        _cfg(tmp_path, owners=[], run_id="bad")


def test_config_rejects_duplicate_owners(tmp_path):
    with pytest.raises(ValueError, match="duplicates"):
        _cfg(tmp_path, owners=["agent_1", "agent_1"], run_id="bad")


# ---------------------------------------------------------------- P1-e ----
def test_no_rng_leak_from_owner_scoping(tmp_path):
    """Scoping the attack must not consume a different amount of RNG.

    The primary attack draws no randomness at all, so the attack stream's
    state must be untouched after a round under either scope; if it ever
    is, the concentrated and uniform conditions would silently desynchronise
    from their common-random-number pairing.
    """
    states = {}
    for tag, owners in (("conc", ["agent_1"]), ("unif", None)):
        run = ExperimentRun(_cfg(tmp_path, owners=owners, run_id=f"rng_{tag}"))
        try:
            before = run.attack_rng.bit_generator.state["state"]["state"]
            run.run_round()
            states[tag] = (before, run.attack_rng.bit_generator.state["state"]["state"])
        finally:
            run.close()
    assert states["conc"][0] == states["conc"][1]
    assert states["unif"][0] == states["unif"][1]
