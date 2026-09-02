"""G2-peer (docs/g2_gates.md, Gate G2a): source-target correctness for the
new constraint-report-head wiring.

G1 fixed `own_critic` and left `peer_critic` untouched: a peer's PPO cost
critic (`AgentBundle.cost_value`, trained against `ret_c`, a GAE(lambda)
bootstrap target, for GAE/advantage estimation) evaluated at the owner's
observation. `docs/g2_gates.md` records the measured consequence --
corr(peer_critic prediction, true cost) ~= 0 across all three G1 seeds --
and the fix: `safelie.training.loop.ExperimentRun.constraint_report_heads`,
one dedicated `DiversifiedReplica` per physical agent, refit once per
round on that agent's OWN masked MC cost-to-go targets, then queried
(without refitting) at whichever owner's observation asks for it as a
`peer_critic` source.

This file tests the three concrete failure modes the G2 brief calls out
by name, exhaustively for N=6:

  1. a source silently estimating a DIFFERENT agent's constraint than the
     one its `source_id`/offset resolves to (owner-relative agent
     identity -- re-asserted here against the NEW code path; the identity
     formula itself is already covered by test_peer_critic_wiring.py and
     is not re-derived here);
  2. a peer source queried using the WRONG owner's observation;
  3. a head trained against a DIFFERENT agent's cost target.

It does not re-test PPO/GAE: nothing in this repair touches
`safelie.training.gae`, `safelie.training.ppo`, or
`AgentBundle.cost_value_net`'s optimizer, and the existing full suite
(`tests/unit/test_dual_estimator_wiring.py`,
`tests/unit/test_agent_bundle_optimizers.py`, `tests/unit/test_gae.py`)
already exercises a full round after this change.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from safelie.sources.estimators import DiversifiedReplica
from safelie.sources.registry import default_m7_sources
from safelie.training.buffer import AgentRollout
from safelie.training.loop import ExperimentRun
from safelie.utils.config import EnvConfig, ExperimentConfig


def _tiny_run(tmp_path, seed: int = 0, n_agents: int = 6) -> ExperimentRun:
    cfg = ExperimentConfig(
        run_id="g2_peer_wiring_test",
        seed=seed,
        env=EnvConfig(
            name="synthetic_constrained_marl", n_agents=n_agents, budget=25.0, obs_dim=4, action_dim=2
        ),
        topology={"name": "ring", "n_agents": n_agents},
        sources=default_m7_sources(),
        ppo={"epochs": 1, "minibatches": 2, "hidden_dim": 8},
        total_steps=16,
        rollout_length=16,
        output_dir=str(tmp_path / "runs"),
    )
    return ExperimentRun(cfg)


class TestDiversifiedReplicaRefitPredictSplit:
    """`refit_and_predict` is now `refit()` then `predict()`; pin that the
    split is behaviourally identical to the pre-split fused method, since
    `ensemble_replica`/`monitor` still call it that way every round."""

    def test_refit_and_predict_equals_refit_then_predict(self):
        rng = np.random.default_rng(0)
        obs = rng.normal(size=(20, 4)).astype(np.float32)
        targets = rng.normal(size=20).astype(np.float32)
        query = rng.normal(size=4).astype(np.float32)

        fused = DiversifiedReplica(obs_dim=4, seed=42)
        split = DiversifiedReplica(obs_dim=4, seed=42)

        fused_result = fused.refit_and_predict(obs, targets, query)
        split.refit(obs, targets)
        split_result = split.predict(query)

        assert math.isclose(fused_result, split_result, rel_tol=1e-9)

    def test_predict_has_no_side_effect_and_is_repeatable(self):
        rng = np.random.default_rng(1)
        obs = rng.normal(size=(20, 4)).astype(np.float32)
        targets = rng.normal(size=20).astype(np.float32)
        query = rng.normal(size=4).astype(np.float32)

        head = DiversifiedReplica(obs_dim=4, seed=7)
        head.refit(obs, targets)
        first = head.predict(query)
        second = head.predict(query)
        assert first == second, "predict() must not mutate the head's weights"

    def test_predict_is_sensitive_to_the_query_point(self):
        """Sanity: a head fit on non-trivial data is not a constant
        function -- otherwise 'queried at the wrong observation' would be
        untestable because it wouldn't matter."""
        rng = np.random.default_rng(2)
        obs = rng.normal(size=(30, 4)).astype(np.float32)
        targets = (obs[:, 0] * 5.0).astype(np.float32)  # clear obs-dependence
        head = DiversifiedReplica(obs_dim=4, seed=3)
        head.refit(obs, targets)
        p_a = head.predict(np.array([5.0, 0.0, 0.0, 0.0], dtype=np.float32))
        p_b = head.predict(np.array([-5.0, 0.0, 0.0, 0.0], dtype=np.float32))
        assert not math.isclose(p_a, p_b, abs_tol=1e-6)


class TestPeerCriticQueriesTheCorrectHeadAtTheCorrectObservation:
    """Failure modes 1 and 2: wrong agent's head, or the right agent's
    head queried with the wrong owner's state."""

    def test_every_owner_offset_combination_for_n6(self, tmp_path):
        run = _tiny_run(tmp_path, n_agents=6)
        agent_ids = run.env.agent_ids
        assert len(agent_ids) == 6

        orig = run._collect_source_value
        calls: list[tuple[str, str, np.ndarray, float]] = []

        def wrapper(spec, owner_id, owner_finalized):
            val = orig(spec, owner_id=owner_id, owner_finalized=owner_finalized)
            if spec.source_type == "peer_critic":
                calls.append((spec.source_id, owner_id, np.array(owner_finalized["obs"][0], copy=True), val))
            return val

        run._collect_source_value = wrapper
        run.run_round()

        assert len(calls) == 6 * 4, "expected 4 peer_critic reports per owner, N=6 owners"

        for source_id, owner_id, obs0, value in calls:
            peer_id = run._peer_agent_id(source_id, owner_id)
            assert peer_id != owner_id, f"owner {owner_id} received itself as {source_id}"
            # The value must be EXACTLY the correctly-identified peer's
            # head, evaluated at THIS owner's obs0 -- re-querying that
            # exact head/obs pair post-hoc must reproduce it bit-for-bit,
            # since `predict` is a pure, non-mutating forward pass and no
            # round boundary (which would refit the heads) has elapsed.
            expected = run.constraint_report_heads[peer_id].predict(obs0)
            assert value == expected, (
                f"{source_id} for owner={owner_id} (peer={peer_id}) did not come from "
                f"constraint_report_heads[{peer_id!r}] queried at the owner's own obs0 -- "
                f"either the wrong agent's head was used, or it was queried with the "
                f"wrong owner's observation"
            )

            # Cross-check against every OTHER agent's head at the same
            # obs0: a report that also matches some other agent's head
            # would be at best inconclusive, but a report that DOESN'T
            # match the correct peer's head while matching a WRONG one is
            # exactly the wiring bug this test exists to catch.
            for other_id, other_head in run.constraint_report_heads.items():
                if other_id == peer_id:
                    continue
                other_val = other_head.predict(obs0)
                assert value != other_val, (
                    f"{source_id} for owner={owner_id} matches a different agent's "
                    f"({other_id}) head at the same observation instead of the correct "
                    f"peer's ({peer_id})"
                )

    @pytest.mark.parametrize("owner_idx", range(6))
    def test_no_owner_ever_receives_itself_as_a_peer_via_the_new_path(self, tmp_path, owner_idx):
        run = _tiny_run(tmp_path, n_agents=6)
        owner_id = f"agent_{owner_idx}"
        for k in range(1, 5):
            peer_id = run._peer_agent_id(f"peer_critic_{k}", owner_id)
            assert peer_id != owner_id
            assert peer_id in run.constraint_report_heads

    def test_a_peer_report_queried_with_a_different_owners_observation_changes(self, tmp_path):
        """Directly exercises failure mode 2: swapping in a different
        owner's obs0 for the same (peer) head must generally change the
        reported value, proving the report is sensitive to (and therefore
        actually uses) the calling owner's own state."""
        run = _tiny_run(tmp_path, n_agents=6)
        run.run_round()

        owner_a = "agent_0"
        peer_id = run._peer_agent_id("peer_critic_1", owner_a)
        head = run.constraint_report_heads[peer_id]

        obs_a = np.array([1.0, 0.3, -0.2, 0.05], dtype=np.float32)
        obs_b = np.array([-2.5, 0.1, 0.4, -0.1], dtype=np.float32)
        assert head.predict(obs_a) != head.predict(obs_b), (
            "a peer_critic head's report did not depend on which observation it was "
            "queried with -- a wrong-owner substitution would be silently undetectable"
        )


class TestEachHeadIsRefitOnOnlyItsOwnAgentsData:
    """Failure mode 3: a head trained against a different agent's cost
    target. Captures the EXACT (obs, cost-to-go) arrays each agent's head
    was refit on this round and checks each against that same agent's own
    `_cost_to_go_targets(finalized)` -- never another agent's."""

    def test_head_refit_data_matches_the_same_agents_own_masked_mc_targets(self, tmp_path, monkeypatch):
        run = _tiny_run(tmp_path, n_agents=6)
        agent_ids = run.env.agent_ids

        captured_finalize: list[dict] = []
        orig_finalize = AgentRollout.finalize

        def finalize_wrapper(self, *args, **kwargs):
            result = orig_finalize(self, *args, **kwargs)
            captured_finalize.append(result)
            return result

        monkeypatch.setattr(AgentRollout, "finalize", finalize_wrapper)

        head_refit_calls: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for aid, head in run.constraint_report_heads.items():
            orig_refit = head.refit

            def make_wrapper(_aid=aid, _orig=orig_refit):
                def wrapper(obs, targets):
                    head_refit_calls[_aid] = (np.array(obs, copy=True), np.array(targets, copy=True))
                    return _orig(obs, targets)

                return wrapper

            head.refit = make_wrapper()

        run.run_round()

        assert len(captured_finalize) == len(agent_ids), "expected one finalize() call per agent, in order"
        assert set(head_refit_calls) == set(agent_ids)

        for aid, finalized in zip(agent_ids, captured_finalize, strict=True):
            expected_obs, expected_targets = run._cost_to_go_targets(finalized)
            got_obs, got_targets = head_refit_calls[aid]
            np.testing.assert_allclose(got_obs, expected_obs)
            np.testing.assert_allclose(got_targets, expected_targets)

        # And, as a cross-check, verify no agent's head was fit on any
        # OTHER agent's finalized targets: the per-agent cost streams in
        # the synthetic env are driven by that agent's own (independently
        # initialized) policy, so distinct agents' mc_cost_to_go arrays
        # are not expected to coincide.
        target_arrays = {aid: head_refit_calls[aid][1] for aid in agent_ids}
        for i, aid_a in enumerate(agent_ids):
            for aid_b in agent_ids[i + 1 :]:
                assert not np.array_equal(target_arrays[aid_a], target_arrays[aid_b]), (
                    f"{aid_a} and {aid_b} were refit on identical cost-to-go targets -- "
                    f"suspicious cross-agent data reuse"
                )


class TestOwnCriticAndReplicaSourcesAreUnaffectedByThisRepair:
    """own_critic already reports the raw MC sum (unchanged since G1);
    ensemble_replica/monitor already refit-per-owner-per-call on MC
    targets (unchanged since G1). Both must still hold after wiring in
    `constraint_report_heads` for peer_critic."""

    def test_own_critic_still_reports_the_raw_mc_window_return(self, tmp_path):
        run = _tiny_run(tmp_path, n_agents=6)
        rec = run.run_round()
        for aid in run.env.agent_ids:
            reports = {r["source_id"]: r["value"] for r in rec["constraints"][aid]["reports"]}
            block = rec["constraints"][aid]["constraint_estimators"]
            assert math.isclose(reports["own_critic"], block["mc_window"], rel_tol=1e-12)

    def test_monitor_sources_are_still_fit_per_owner_per_call(self, tmp_path):
        """Unlike the new peer-critic heads, `self.replicas` (backing
        monitor/ensemble_replica) must still be refit fresh for EVERY
        owner within the same round -- i.e. queried via
        `refit_and_predict`, not the new `refit`-once/`predict`-many
        pattern."""
        run = _tiny_run(tmp_path, n_agents=6)
        monitor_ids = [s.source_id for s in run.cfg.sources.sources if s.source_type == "monitor"]
        assert monitor_ids, "test config must include at least one monitor source"

        call_counts = {sid: 0 for sid in monitor_ids}
        for sid in monitor_ids:
            replica = run.replicas[sid]
            orig = replica.refit_and_predict

            def make_wrapper(_sid=sid, _orig=orig):
                def wrapper(obs, targets, query_obs):
                    call_counts[_sid] += 1
                    return _orig(obs, targets, query_obs)

                return wrapper

            replica.refit_and_predict = make_wrapper()

        run.run_round()
        for sid in monitor_ids:
            assert call_counts[sid] == len(run.env.agent_ids), (
                f"{sid} was refit {call_counts[sid]} times, expected once per owner "
                f"({len(run.env.agent_ids)})"
            )


class TestConstraintReportHeadStateSurvivesCheckpointRoundTrip:
    def test_checkpoint_restore_reproduces_head_predictions(self, tmp_path):
        run = _tiny_run(tmp_path, n_agents=6)
        run.run_round()
        ckpt = tmp_path / "ckpt.pt"
        run.checkpoint(ckpt)

        probe = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        before = {aid: h.predict(probe) for aid, h in run.constraint_report_heads.items()}

        fresh = _tiny_run(tmp_path, n_agents=6)
        fresh.restore(ckpt)
        after = {aid: h.predict(probe) for aid, h in fresh.constraint_report_heads.items()}

        for aid in run.env.agent_ids:
            assert math.isclose(before[aid], after[aid], rel_tol=1e-6), (
                f"constraint_report_heads[{aid!r}] did not round-trip through checkpoint/restore"
            )
