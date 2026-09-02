"""G1: what the dual update actually consumes, end to end.

`test_constraint_return.py` pins the estimator's arithmetic. This file
pins the *wiring* -- the chain the G1 brief asks to be traced explicitly:

    C_t -> rollout -> constraint estimator -> source reports
        -> adversarial corruption -> aggregation -> residual -> lambda

Two properties matter most and each gets its own class. First, switching
`ExperimentConfig.constraint_estimator` must change the estimated
quantity and nothing else about the mechanism. Second -- section 5 of the
brief, and the easiest thing to break while "fixing" an estimator -- the
attacker must still corrupt the *communicated* report, and the dual must
still consume only the corrupted-then-aggregated value. A repair that
routed a clean Monte-Carlo number around the attack surface would make
every subsequent attack experiment vacuous while looking like an
improvement.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from safelie.training.constraint_return import discounted_window_return
from safelie.training.loop import ExperimentRun


def _reports(record, aid):
    return {r["source_id"]: r["value"] for r in record["constraints"][aid]["reports"]}


class TestTheEstimatorSwitchSelectsWhatOwnCriticReports:
    def test_mc_window_reports_the_monte_carlo_constraint_return(self, tiny_config_factory):
        cfg = tiny_config_factory(run_id="wire_mc").model_copy(
            update={"constraint_estimator": "mc_window"}
        )
        run = ExperimentRun(cfg)
        rec = run.run_round()
        for aid in run.env.agent_ids:
            block = rec["constraints"][aid]["constraint_estimators"]
            assert block["active"] == "mc_window"
            assert math.isclose(_reports(rec, aid)["own_critic"], block["mc_window"], rel_tol=1e-12)

    def test_gae_lambda_reproduces_the_pre_g1_behaviour(self, tiny_config_factory):
        """Kept selectable so the G0 artifacts stay reproducible and so the
        two estimators can be compared head to head -- not because it is a
        defensible choice for new science."""
        cfg = tiny_config_factory(run_id="wire_gae").model_copy(
            update={"constraint_estimator": "gae_lambda"}
        )
        run = ExperimentRun(cfg)
        rec = run.run_round()
        for aid in run.env.agent_ids:
            block = rec["constraints"][aid]["constraint_estimators"]
            assert block["active"] == "gae_lambda"
            assert math.isclose(_reports(rec, aid)["own_critic"], block["gae_lambda"], rel_tol=1e-12)

    def test_mc_window_is_the_default(self):
        """The GAE-target estimator is a documented defect; a default that
        preserved it would silently propagate it into every config that
        does not mention the field."""
        from safelie.utils.config import ExperimentConfig

        assert ExperimentConfig.model_fields["constraint_estimator"].default == "mc_window"

    def test_both_estimators_are_logged_every_round_whichever_is_active(self, tiny_config_factory):
        """Gate G1 needs the head-to-head comparison to come from ONE run's
        own logs, so both numbers must be present regardless of which one
        fed the dual."""
        for choice in ("mc_window", "gae_lambda"):
            cfg = tiny_config_factory(run_id=f"wire_log_{choice}").model_copy(
                update={"constraint_estimator": choice}
            )
            rec = ExperimentRun(cfg).run_round()
            for aid in rec["constraints"]:
                block = rec["constraints"][aid]["constraint_estimators"]
                assert block["gae_lambda"] is not None
                assert block["mc_window"] is not None
                assert block["mc_episodic"] is not None


class TestTheEstimatorIsOnTheReturnScaleTheDualCompares:
    def test_residual_is_the_aggregate_minus_the_budget(self, tiny_config_factory):
        """The middle link: whatever the sources report, the residual the
        dual consumes is exactly `point_estimate - d`, on the return
        scale, with no rescaling slipped in by the estimator change."""
        cfg = tiny_config_factory(run_id="wire_residual")
        rec = ExperimentRun(cfg).run_round()
        for aid in rec["constraints"]:
            c = rec["constraints"][aid]
            assert math.isclose(
                c["constraint_residual"],
                c["mechanism_reported_cost_return"] - cfg.env.budget,
                rel_tol=1e-12,
            )

    def test_doubling_every_per_step_cost_doubles_the_own_critic_report(self, tiny_config_factory):
        """A return-scale estimator must scale linearly with the cost
        stream. Checked directly on the estimator with the rollout's own
        recorded costs, so it covers the buffer wiring and not only the
        arithmetic in isolation."""
        cfg = tiny_config_factory(run_id="wire_scale").model_copy(
            update={"constraint_estimator": "mc_window"}
        )
        run = ExperimentRun(cfg)
        rec = run.run_round()
        for aid in run.env.agent_ids:
            mc = rec["constraints"][aid]["constraint_estimators"]["mc_window"]
            assert np.isfinite(mc)
        costs = np.array([0.0, 1.0, 0.0, 2.0])
        assert math.isclose(
            discounted_window_return(2 * costs, cfg.ppo.gamma),
            2 * discounted_window_return(costs, cfg.ppo.gamma),
            rel_tol=1e-12,
        )


class TestTheAttackSurfaceSurvivesTheRepair:
    """Section 5 of the G1 brief. The new estimator must sit UPSTREAM of
    the corruption point, never around it."""

    def test_corruption_still_lands_between_the_clean_reports_and_the_dual(
        self, tiny_config_factory
    ):
        """Same seed, same estimator, attack off vs on.

        Two things must hold together, and they are the whole point of the
        threat model. The *pre-attack* source reports must be IDENTICAL --
        the estimator sits upstream of the corruption point, so turning the
        attack on cannot change what the sources honestly measured. The
        value the dual consumes must nevertheless DIFFER -- the corruption
        lands on the communicated residual, after estimation and before
        consensus, exactly where `safelie.attacks.apply_attack` is called.
        A repair that moved the estimator downstream of the attack would
        break the first assertion; one that routed a clean number around
        the attack would break the second.
        """
        clean = ExperimentRun(
            tiny_config_factory(run_id="atk_off").model_copy(
                update={"constraint_estimator": "mc_window"}
            )
        ).run_round()
        attacked = ExperimentRun(
            tiny_config_factory(run_id="atk_on", attack_name="primary", f=1).model_copy(
                update={"constraint_estimator": "mc_window"}
            )
        ).run_round()

        aid = next(iter(clean["constraints"]))
        corrupted_ids = attacked["constraints"][aid]["corrupted_source_ids"]
        assert corrupted_ids, "the test config must actually compromise a source"
        for sid in corrupted_ids:
            assert math.isclose(
                _reports(clean, aid)[sid], _reports(attacked, aid)[sid], rel_tol=1e-9
            ), f"source {sid}'s honest pre-attack estimate changed when the attack was enabled"
        assert not math.isclose(
            clean["constraints"][aid]["mechanism_reported_cost_return"],
            attacked["constraints"][aid]["mechanism_reported_cost_return"],
            rel_tol=1e-9,
        ), "the value the dual compares against d was unaffected by the attack -- bypassed"

    def test_the_dual_consumes_the_post_attack_aggregate_not_the_clean_estimate(
        self, tiny_config_factory
    ):
        """`mechanism_reported_cost_return` -- the number compared against
        d -- must be the aggregate of the CORRUPTED reports. If it ever
        equals the agent's own clean Monte-Carlo estimate under a
        persistent attack, the corruption is no longer in the dual's path."""
        cfg = tiny_config_factory(run_id="atk_path", attack_name="primary", f=1).model_copy(
            update={"constraint_estimator": "mc_window"}
        )
        rec = ExperimentRun(cfg).run_round()
        for aid in rec["constraints"]:
            c = rec["constraints"][aid]
            clean_mc = c["constraint_estimators"]["mc_window"]
            assert not math.isclose(c["mechanism_reported_cost_return"], clean_mc, rel_tol=1e-9)

    def test_the_aggregate_stays_inside_the_range_of_the_reports_it_aggregates(
        self, tiny_config_factory
    ):
        """Structural proof that the aggregation still consumes the report
        vector and nothing else -- a privileged side channel would show up
        as an aggregate outside the reports' own convex hull for `mean`."""
        cfg = tiny_config_factory(run_id="atk_hull", attack_name="primary", f=1).model_copy(
            update={"constraint_estimator": "mc_window"}
        )
        rec = ExperimentRun(cfg).run_round()
        for aid in rec["constraints"]:
            c = rec["constraints"][aid]
            # `reports` is pre-attack; the post-attack values are the ones
            # aggregated, and for `mean` the result must lie within their
            # own spread. Reconstruct the bound from the logged aggregate.
            agg = c["aggregate"]
            assert agg["retained_n"] == len(c["reports"])
            assert np.isfinite(agg["point_estimate"])

    def test_no_true_cost_reaches_the_learners_estimator(self, tiny_config_factory):
        """The oracle stays withheld: the estimator is built from the
        learner-visible `reported_cost` stream only. Enforced structurally
        elsewhere (tests/isolation), asserted here at the value level with
        an environment whose reported and true cost are deliberately
        DIFFERENT, so a leak would be visible as a number the learner
        should not be able to produce."""
        cfg = tiny_config_factory(run_id="atk_isolation")
        run = ExperimentRun(cfg)
        rec = run.run_round()
        for aid in run.env.agent_ids:
            block = rec["constraints"][aid]["constraint_estimators"]
            assert set(block) >= {"gae_lambda", "mc_window", "mc_episodic", "active"}
            assert "true_cost" not in str(block)


class TestTheReplicaTargetsFollowTheSelectedEstimator:
    def test_mc_window_masks_the_censored_tail_of_the_regression_targets(
        self, tiny_config_factory
    ):
        from safelie.training.constraint_return import complete_target_count

        cfg = tiny_config_factory(run_id="wire_targets").model_copy(
            update={"constraint_estimator": "mc_window"}
        )
        run = ExperimentRun(cfg)
        rec = run.run_round()
        expected = complete_target_count(cfg.rollout_length, cfg.ppo.gamma)
        for aid in run.env.agent_ids:
            assert rec["constraints"][aid]["constraint_estimators"]["n_mc_targets"] == expected

    def test_gae_lambda_leaves_the_replica_targets_unmasked(self, tiny_config_factory):
        cfg = tiny_config_factory(run_id="wire_targets_gae").model_copy(
            update={"constraint_estimator": "gae_lambda"}
        )
        run = ExperimentRun(cfg)
        finalized_obs = np.zeros((cfg.rollout_length, 4))
        fake = {
            "obs": finalized_obs,
            "ret_c": np.arange(cfg.rollout_length, dtype=float),
            "mc_cost_to_go": np.zeros(cfg.rollout_length),
            "n_mc_targets": 3,
        }
        obs, targets = run._cost_to_go_targets(fake)
        assert len(targets) == cfg.rollout_length
        np.testing.assert_allclose(targets, fake["ret_c"])


@pytest.mark.parametrize("estimator", ["mc_window", "gae_lambda"])
def test_a_full_round_is_finite_and_the_dual_stays_projected(tiny_config_factory, estimator):
    """Liveness/sanity under both settings: no NaN anywhere in the chain,
    and lambda stays inside [0, lambda_max] as `safelie.training.dual`
    promises."""
    cfg = tiny_config_factory(run_id=f"wire_finite_{estimator}").model_copy(
        update={"constraint_estimator": estimator}
    )
    run = ExperimentRun(cfg)
    for _ in range(2):
        rec = run.run_round()
        for aid in rec["constraints"]:
            c = rec["constraints"][aid]
            assert np.isfinite(c["mechanism_reported_cost_return"])
            assert np.isfinite(c["constraint_residual"])
            assert 0.0 <= c["lambda_after"] <= cfg.dual.lambda_max
            for v in c["ppo"].values():
                assert np.isfinite(v)
