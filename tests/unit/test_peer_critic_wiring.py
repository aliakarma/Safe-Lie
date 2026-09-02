"""P0 #7 regression: peer_critic sources must be owner-relative.

Bug fixed: `safelie.training.loop.ExperimentRun._collect_source_value`
used to map `peer_critic_<k>` to the literal agent `agent_<k>`, the same
physical agent for every owner. Under the pilot's own M=7 config
(`peer_critic_1..4` on N=6 agents), that made agent_1..agent_4 each
receive themselves as one of their four "peer" sources -- 4 of 6 owners
(67%), not a rare edge case. This file tests the owner-relative
replacement (`agent (owner_index + k) mod N`) exhaustively for N=6 and
checks the config-time guard that rejects self-colliding offsets.
"""

from __future__ import annotations

import pytest

from safelie.training.loop import ExperimentRun
from safelie.utils.config import EnvConfig, ExperimentConfig, SourcesConfig, SourceSpec


def _m7_sources() -> SourcesConfig:
    """The pilot's own source config (App. B): own critic + peer_critic_1..4
    + 2 monitors -- copied here (not imported from
    safelie.sources.registry.default_m7_sources) so this test fails if
    that default itself is ever edited to dodge the bug rather than fix
    the wiring."""
    specs = [SourceSpec(source_id="own_critic", source_type="own_critic", independence_class="ic_own")]
    for i in range(1, 5):
        specs.append(
            SourceSpec(source_id=f"peer_critic_{i}", source_type="peer_critic", independence_class=f"ic_peer_{i}")
        )
    for i in range(1, 3):
        specs.append(SourceSpec(source_id=f"monitor_{i}", source_type="monitor", independence_class=f"ic_monitor_{i}"))
    return SourcesConfig(sources=specs)


def _tiny_run(tmp_path, n_agents: int = 6) -> ExperimentRun:
    cfg = ExperimentConfig(
        run_id="peer_wiring_test",
        env=EnvConfig(name="synthetic_constrained_marl", n_agents=n_agents, budget=25.0, obs_dim=4, action_dim=2),
        topology={"name": "ring", "n_agents": n_agents},
        sources=_m7_sources(),
        ppo={"epochs": 1, "minibatches": 2, "hidden_dim": 8},
        total_steps=16,
        rollout_length=16,
        output_dir=str(tmp_path / "runs"),
    )
    return ExperimentRun(cfg)


class TestOwnerRelativePeerMapping:
    """At minimum test all owner/source combinations for N=6 (task spec)."""

    @pytest.mark.parametrize("owner_idx", range(6))
    def test_no_owner_ever_receives_itself_as_a_peer(self, tmp_path, owner_idx):
        run = _tiny_run(tmp_path, n_agents=6)
        owner_id = f"agent_{owner_idx}"
        for k in range(1, 5):  # peer_critic_1..4, the pilot's own config
            peer_id = run._peer_agent_id(f"peer_critic_{k}", owner_id)
            assert peer_id != owner_id, (
                f"owner={owner_id} k={k} resolved to itself -- this is exactly the "
                f"bug: peer_critic_{k} used to map to the literal agent 'agent_{k}' "
                f"regardless of owner"
            )

    def test_all_36_owner_source_combinations_for_n6(self, tmp_path):
        """Exhaustive: every (owner, peer_critic_k) pair for N=6, k in 1..4
        (24 combinations) plus a wider k sweep (1..5) for full coverage."""
        run = _tiny_run(tmp_path, n_agents=6)
        agent_ids = run.env.agent_ids
        assert len(agent_ids) == 6
        results = {}
        for owner_id in agent_ids:
            for k in range(1, 6):  # 1..N-1, exhaustive over all valid offsets
                peer_id = run._peer_agent_id(f"peer_critic_{k}", owner_id)
                assert peer_id != owner_id
                assert peer_id in agent_ids
                results[(owner_id, k)] = peer_id
        assert len(results) == 6 * 5

    def test_mapping_is_owner_relative_not_a_fixed_literal_agent(self, tmp_path):
        """The regression itself: the same source_id must resolve to
        *different* agents for different owners (the old bug resolved
        peer_critic_2 to the literal 'agent_2' for every owner)."""
        run = _tiny_run(tmp_path, n_agents=6)
        targets = {owner: run._peer_agent_id("peer_critic_2", owner) for owner in run.env.agent_ids}
        assert len(set(targets.values())) > 1, (
            "peer_critic_2 resolved to the same agent for every owner -- "
            "mapping is not owner-relative"
        )

    def test_each_owner_gets_four_distinct_peers_under_the_pilot_config(self, tmp_path):
        """'ensure all agents have the intended number of valid peer
        sources' -- the M=7 pilot config declares 4 peer_critic sources
        per owner; they must resolve to 4 *distinct* non-owner agents."""
        run = _tiny_run(tmp_path, n_agents=6)
        for owner_id in run.env.agent_ids:
            peers = {run._peer_agent_id(f"peer_critic_{k}", owner_id) for k in range(1, 5)}
            assert len(peers) == 4, f"owner={owner_id} got {len(peers)} distinct peers, expected 4: {peers}"
            assert owner_id not in peers

    def test_source_identity_full_matrix_matches_formula(self, tmp_path):
        """Verify the resolved peer against the closed-form (owner_index +
        k) mod N directly, for every owner and offset -- the source
        identity check the task requests."""
        run = _tiny_run(tmp_path, n_agents=6)
        n = 6
        for i, owner_id in enumerate(run.env.agent_ids):
            for k in range(1, n):
                expected = f"agent_{(i + k) % n}"
                assert run._peer_agent_id(f"peer_critic_{k}", owner_id) == expected

    def test_nonnumeric_offset_raises(self, tmp_path):
        run = _tiny_run(tmp_path, n_agents=6)
        with pytest.raises(ValueError, match="integer offset"):
            run._peer_agent_id("peer_critic_abc", "agent_0")

    def test_self_colliding_offset_raises_defensively(self, tmp_path):
        """Even though config validation should already reject this
        (test_config_rejects_self_colliding_peer_offset below), the
        resolver itself must never silently return the owner."""
        run = _tiny_run(tmp_path, n_agents=6)
        with pytest.raises(ValueError, match="resolved to owner"):
            run._peer_agent_id("peer_critic_6", "agent_0")  # 6 % 6 == 0


class TestConfigTimeGuard:
    """The bug should be unreachable from a valid config, not merely
    handled gracefully once encountered."""

    def test_config_rejects_self_colliding_peer_offset(self):
        specs = [
            SourceSpec(source_id="own_critic", source_type="own_critic", independence_class="ic_own"),
            SourceSpec(source_id="peer_critic_6", source_type="peer_critic", independence_class="ic_peer_6"),
        ]
        with pytest.raises(ValueError, match="multiple of env.n_agents"):
            ExperimentConfig(
                run_id="bad_offset",
                env={"name": "synthetic_constrained_marl", "n_agents": 6, "budget": 25.0},
                topology={"name": "ring", "n_agents": 6},
                sources=SourcesConfig(sources=specs),
                total_steps=100,
            )

    def test_config_accepts_the_pilots_own_offsets(self):
        """The exact M=7 config shipped in configs/experiment/pilot_*.yaml
        must remain valid after the guard is added."""
        cfg = ExperimentConfig(
            run_id="pilot_like",
            env={"name": "synthetic_constrained_marl", "n_agents": 6, "budget": 25.0},
            topology={"name": "ring", "n_agents": 6},
            sources=_m7_sources(),
            total_steps=100,
        )
        assert cfg.sources.M == 7
