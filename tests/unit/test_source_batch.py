"""G9 (docs/g9_gates.md): the parallel trajectory-batch source layer.

These tests exist to make four of G9's gates *checked* rather than
*claimed*:

  * **G9g** -- the M replica RNG streams are disjoint, and the number of
    worker processes cannot change a single reported number. The second
    half matters more than it looks: if worker count changed results, then
    `workers` would be a scientific parameter disguised as a performance
    setting, and no G9 number would be reproducible on another machine.
  * **G9h** -- the policy really is pinned across replicas, and collection
    mutates nothing.
  * **G9a-iv/v** -- source collection contributes nothing to PPO and runs
    no neural estimator.
  * the config-level guard that the neural and trajectory-batch source
    paths cannot be mixed.

Everything here runs on `synthetic_constrained_marl` with a short rollout,
so the suite stays fast; the properties under test are properties of the
collection machinery, not of any particular environment.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from safelie.training.loop import ExperimentRun
from safelie.training.source_batch import policy_checksum, policy_payload
from safelie.utils.config import ExperimentConfig

SRC = Path(__file__).resolve().parents[2] / "src" / "safelie"


def _cfg(tmp_path, workers: int = 1, R_m: int = 4, M: int = 3, rounds=(0,)) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "run_id": "g9_unit",
            "seed": 0,
            "env": {
                "name": "synthetic_constrained_marl",
                "n_agents": 3,
                "budget": 5.0,
                "obs_dim": 6,
                "action_dim": 2,
            },
            "topology": {"name": "ring", "n_agents": 3},
            "sources": {
                "sources": [
                    {
                        "source_id": f"batch_{i}",
                        "source_type": "trajectory_batch",
                        "independence_class": f"ic_batch{i}",
                    }
                    for i in range(1, M + 1)
                ]
            },
            "attack": {"name": "none"},
            "defense": {"name": "mean", "f": 0},
            "source_collection": {
                "mode": "parallel_trajectory_batch",
                "M": M,
                "R_m": R_m,
                "workers": workers,
                "validation_rounds": list(rounds),
                "R_ref": 6,
            },
            "total_steps": 64,
            "rollout_length": 16,
            "output_dir": str(tmp_path),
        }
    )


class TestConfigGuards:
    def test_trajectory_batch_source_requires_the_batch_collection_mode(self, tmp_path):
        """A `trajectory_batch` spec under the neural path has no estimator
        behind it and would have nothing to report -- so the config must
        refuse rather than fail at round 0."""
        raw = _cfg(tmp_path).model_dump()
        raw["source_collection"]["mode"] = "neural"
        with pytest.raises(ValueError, match="parallel_trajectory_batch"):
            ExperimentConfig.model_validate(raw)

    def test_mixing_neural_and_batch_sources_is_rejected(self, tmp_path):
        """Averaging a state-conditional head prediction into a policy-level
        Monte-Carlo mean produces a number that estimates neither."""
        raw = _cfg(tmp_path).model_dump()
        raw["sources"]["sources"][0]["source_type"] = "own_critic"
        with pytest.raises(ValueError, match="trajectory_batch"):
            ExperimentConfig.model_validate(raw)

    def test_M_must_match_the_number_of_configured_sources(self, tmp_path):
        raw = _cfg(tmp_path).model_dump()
        raw["source_collection"]["M"] = 5
        with pytest.raises(ValueError, match="same"):
            ExperimentConfig.model_validate(raw)

    def test_validation_rounds_outside_the_horizon_are_rejected(self, tmp_path):
        raw = _cfg(tmp_path).model_dump()
        raw["source_collection"]["validation_rounds"] = [999]
        with pytest.raises(ValueError, match="validation_rounds"):
            ExperimentConfig.model_validate(raw)


class TestPolicyPinning:
    """G9h."""

    def test_checksum_covers_normalization_statistics_not_only_weights(self, tmp_path):
        """`obs_rms` is applied before every policy forward pass, so a run
        with the same weights and different normalization statistics is a
        different sampling policy. A checksum blind to that would report
        'pinned' across a real policy change."""
        run = ExperimentRun(_cfg(tmp_path), _skip_calibration=True)
        aids = list(run.env.agent_ids)
        before = policy_checksum(run.agents, aids)

        run.agents[aids[0]].obs_rms.mean = run.agents[aids[0]].obs_rms.mean + 1.0
        assert policy_checksum(run.agents, aids) != before

    def test_checksum_ignores_the_critics(self, tmp_path):
        """The critics are not consulted during collection and do not enter
        the action distribution; a pinning check that fired on them would
        fire for reasons unrelated to what it is checking."""
        run = ExperimentRun(_cfg(tmp_path), _skip_calibration=True)
        aids = list(run.env.agent_ids)
        before = policy_checksum(run.agents, aids)
        with torch.no_grad():
            for p in run.agents[aids[0]].cost_value_net.parameters():
                p.add_(1.0)
        assert policy_checksum(run.agents, aids) == before

    def test_collection_does_not_mutate_the_policy(self, tmp_path):
        run = ExperimentRun(_cfg(tmp_path), _skip_calibration=True)
        aids = list(run.env.agent_ids)
        before = policy_checksum(run.agents, aids)
        result = run.batch_sources.collect(run.agents, 0)
        assert policy_checksum(run.agents, aids) == before
        assert result.policy_checksum == before
        assert all(c == before for c in result.worker_checksums.values())

    def test_payload_carries_no_optimizer_state(self, tmp_path):
        """A worker that received optimizer state could in principle step
        it. Not sending it is a structural guarantee, not a size saving."""
        run = ExperimentRun(_cfg(tmp_path), _skip_calibration=True)
        payload = policy_payload(run.agents, list(run.env.agent_ids))
        for blob in payload.values():
            assert set(blob) == {"policy", "obs_rms_mean", "obs_rms_var", "obs_rms_count"}


class TestRngIndependence:
    """G9g."""

    def test_replica_seed_sets_are_pairwise_disjoint(self, tmp_path):
        run = ExperimentRun(_cfg(tmp_path, R_m=8), _skip_calibration=True)
        result = run.batch_sources.collect(run.agents, 0)
        seed_sets = [{e for e, _ in pairs} for pairs in result.seeds.values()]
        for i in range(len(seed_sets)):
            for j in range(i + 1, len(seed_sets)):
                assert not (seed_sets[i] & seed_sets[j])

    def test_no_seed_is_ever_reissued_across_rounds(self, tmp_path):
        run = ExperimentRun(_cfg(tmp_path, R_m=8), _skip_calibration=True)
        for k in range(4):
            run.batch_sources.collect(run.agents, k)
        audit = run.batch_sources.seed_audit()
        assert audit["duplicate_seed_events"] == 0
        assert audit["n_env_seeds_issued"] == 4 * 3 * 8

    def test_reference_stream_is_disjoint_from_every_source_stream(self, tmp_path):
        run = ExperimentRun(_cfg(tmp_path, R_m=8), _skip_calibration=True)
        result = run.batch_sources.collect(run.agents, 0, collect_reference=True)
        src = {e for pairs in result.seeds.values() for e, _ in pairs}
        ref = {e for e, _ in result.reference_seeds}
        assert not (src & ref)
        assert len(ref) == run.cfg.source_collection.R_ref

    def test_collector_rng_state_survives_a_checkpoint_round_trip(self, tmp_path):
        """A resumed run must not restart its source streams from the spawn
        point and re-issue seeds it has already used."""
        cfg = _cfg(tmp_path, R_m=4)
        run = ExperimentRun(cfg, _skip_calibration=True)
        run.batch_sources.collect(run.agents, 0)
        state = run.batch_sources.state_dict()
        expected = run.batch_sources.collect(run.agents, 1).seeds

        run2 = ExperimentRun(cfg, _skip_calibration=True)
        run2.batch_sources.load_state_dict(state)
        assert run2.batch_sources.collect(run2.agents, 1).seeds == expected


class TestWorkerCountIsNotAScientificParameter:
    """G9g / the determinism requirement in docs/g9_gates.md.

    The single most important property of this module: `workers` is a
    compute knob. If this test ever fails, every G9 number becomes
    machine-dependent and the campaign is void.
    """

    def test_one_worker_and_three_workers_agree_bitwise(self, tmp_path):
        run1 = ExperimentRun(_cfg(tmp_path / "a", workers=1, R_m=4), _skip_calibration=True)
        run3 = ExperimentRun(_cfg(tmp_path / "b", workers=3, R_m=4), _skip_calibration=True)
        try:
            r1 = run1.batch_sources.collect(run1.agents, 0)
            r3 = run3.batch_sources.collect(run3.agents, 0)
            assert r1.seeds == r3.seeds
            for rid in r1.source_means:
                for aid in r1.source_means[rid]:
                    assert r1.source_means[rid][aid] == r3.source_means[rid][aid]
                    assert r1.per_trajectory[rid][aid] == r3.per_trajectory[rid][aid]
        finally:
            run1.close()
            run3.close()


class TestSourceValuesAreWhatTheEstimatorSays:
    """G9a-v: the reported source is a plain sample mean of
    `discounted_window_return`, not anything else."""

    def test_source_mean_is_the_mean_of_its_own_trajectories(self, tmp_path):
        run = ExperimentRun(_cfg(tmp_path, R_m=5), _skip_calibration=True)
        result = run.batch_sources.collect(run.agents, 0)
        for rid, per_owner in result.per_trajectory.items():
            for aid, vals in per_owner.items():
                assert len(vals) == 5
                assert not any(np.isnan(v) for v in vals)
                assert result.source_means[rid][aid] == pytest.approx(float(np.mean(vals)), abs=1e-12)

    def test_the_three_sources_are_actually_different_draws(self, tmp_path):
        """Identical source values across replicas would mean shared
        randomness -- the failure docs/g9_gates.md G9f's lower bound is
        there to catch."""
        run = ExperimentRun(_cfg(tmp_path, R_m=5), _skip_calibration=True)
        result = run.batch_sources.collect(run.agents, 0)
        rids = sorted(result.source_means)
        aid = list(run.env.agent_ids)[0]
        vals = [result.source_means[r][aid] for r in rids]
        assert len(set(vals)) == len(vals)


class TestNoNeuralPathInTheBatchSourcePipeline:
    """G9a-v, at AST level: whatever else `source_batch.py` does, it must
    not reach for a critic, GAE, a fitted head, or an optimizer."""

    FORBIDDEN_ATTRS = {
        "cost_value",
        "cost_value_net",
        "value_net",
        "refit",
        "refit_and_predict",
        "update_normalization_stats",
        "backward",
        "step",  # optimizer.step; env.step is called via a local name below
    }

    def test_module_never_references_a_neural_estimator(self):
        tree = ast.parse((SRC / "training" / "source_batch.py").read_text(encoding="utf-8"))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in self.FORBIDDEN_ATTRS:
                # `env.step(...)` is the environment transition, which is
                # the one legitimate `.step` in a rollout loop.
                if node.attr == "step" and isinstance(node.value, ast.Name) and node.value.id == "env":
                    continue
                offenders.append(node.attr)
        assert not offenders, f"source_batch.py references neural/optimizer paths: {sorted(set(offenders))}"

    def test_module_imports_no_gae_and_no_estimators(self):
        tree = ast.parse((SRC / "training" / "source_batch.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in ("safelie.training.gae", "safelie.sources.estimators")


class TestPpoIsUnaffected:
    """G9a-iv: the 3 x R_m source trajectories are not PPO's data."""

    def test_collection_does_not_touch_normalization_or_optimizer_state(self, tmp_path):
        run = ExperimentRun(_cfg(tmp_path, R_m=4), _skip_calibration=True)
        aids = list(run.env.agent_ids)
        before_counts = {aid: run.agents[aid].obs_rms.count for aid in aids}
        before_opt = {
            aid: len(run.agents[aid].policy_optimizer.state_dict()["state"]) for aid in aids
        }
        run.batch_sources.collect(run.agents, 0)
        for aid in aids:
            assert run.agents[aid].obs_rms.count == before_counts[aid]
            assert len(run.agents[aid].policy_optimizer.state_dict()["state"]) == before_opt[aid]

    def test_a_full_round_still_reports_three_sources_and_moves_the_policy(self, tmp_path):
        run = ExperimentRun(_cfg(tmp_path, R_m=3), _skip_calibration=True)
        aids = list(run.env.agent_ids)
        before = policy_checksum(run.agents, aids)
        rec = run.run_round()
        for aid in aids:
            reports = rec["constraints"][aid]["reports"]
            assert len(reports) == 3
            assert sorted(r["source_id"] for r in reports) == ["batch_1", "batch_2", "batch_3"]
            # The aggregate the dual update consumed is the mean of the
            # three batch means, nothing else.
            assert rec["constraints"][aid]["mechanism_reported_cost_return"] == pytest.approx(
                float(np.mean([r["value"] for r in reports])), abs=1e-12
            )
        assert "source_batch" in rec
        assert rec["source_batch"]["theta_k_checksum_stable_through_dual"] is True
        # PPO ran: theta_k -> theta_{k+1}.
        assert policy_checksum(run.agents, aids) != before

    def test_no_constraint_report_head_is_refit_in_batch_mode(self, tmp_path):
        """G9a-v at the loop level: under the neural path every head is
        refit once per round; under the batch path none is, because none is
        queried."""
        run = ExperimentRun(_cfg(tmp_path, R_m=3), _skip_calibration=True)
        aids = list(run.env.agent_ids)
        before = [
            [p.detach().clone() for p in run.constraint_report_heads[aid].net.parameters()] for aid in aids
        ]
        run.run_round()
        after = [
            [p.detach().clone() for p in run.constraint_report_heads[aid].net.parameters()] for aid in aids
        ]
        for b_agent, a_agent in zip(before, after, strict=True):
            for b, a in zip(b_agent, a_agent, strict=True):
                assert torch.equal(b, a)


class TestValidationReferenceIsWithheld:
    """docs/g9_gates.md step 3b: the R_ref reference exists to grade the
    sources, so it must never be one of them."""

    def test_reference_is_absent_from_the_round_record(self, tmp_path):
        run = ExperimentRun(_cfg(tmp_path, R_m=3, rounds=(0,)), _skip_calibration=True)
        rec = run.run_round()
        assert rec["source_batch"]["reference_collected"] is True
        blob = str(rec)
        assert "reference_mean" not in blob
        for aid in run.env.agent_ids:
            assert len(rec["constraints"][aid]["reports"]) == 3

    def test_reference_is_written_to_its_own_file(self, tmp_path):
        cfg = _cfg(tmp_path, R_m=3, rounds=(0,))
        run = ExperimentRun(cfg, _skip_calibration=True)
        run.run_round()
        run.close()
        ref_path = run.output_dir / "validation_reference.jsonl"
        assert ref_path.exists()
        import json

        rec = json.loads(ref_path.read_text(encoding="utf-8").splitlines()[0])
        assert rec["R_ref"] == cfg.source_collection.R_ref
        assert set(rec["reference_mean"]) == set(run.env.agent_ids)
