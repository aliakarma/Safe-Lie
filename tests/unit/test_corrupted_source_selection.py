"""A1 (docs/a1_attack_gates.md §3): the adversary's source is named, not
positional.

The historical rule -- first `f` non-`own_critic` sources in config order
-- always selects `batch_1` under the G10 source list, which would make
"which source was attacked" a constant across all three training seeds
and therefore a confound with source identity. A1 balances it. These
tests pin the two properties that balancing must not cost:

  1. naming a source actually corrupts that source and no other;
  2. naming it does NOT reorder `cfg.sources.sources`, so every replica
     keeps the spawned RNG stream its position assigns it -- which is
     what makes a seed's clean and attacked runs share source seeds.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from safelie.attacks import apply_attack
from safelie.training.loop import select_corrupted_sources
from safelie.utils.config import ExperimentConfig

BATCH_IDS = ["batch_1", "batch_2", "batch_3"]


def _batch_kwargs(**overrides):
    kwargs = dict(
        run_id="a1_test",
        env={"name": "manyagent_ant", "n_agents": 6, "budget": 25.0,
             "cost_mode": "per_agent_velocity", "velocity_threshold": 0.75},
        topology={"name": "ring", "n_agents": 6},
        sources={"sources": [
            {"source_id": sid, "source_type": "trajectory_batch",
             "independence_class": f"ic_{sid}"} for sid in BATCH_IDS
        ]},
        source_collection={"mode": "parallel_trajectory_batch", "M": 3, "R_m": 30},
        total_steps=2000,
        rollout_length=2000,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize("target", BATCH_IDS)
def test_named_source_is_the_one_selected(target):
    cfg = ExperimentConfig(**_batch_kwargs(
        attack={"name": "primary", "f": 1, "budget_ratio": 0.5,
                "corrupted_source_ids": [target]}
    ))
    assert select_corrupted_sources(
        cfg.sources.sources, cfg.attack.f, cfg.attack.corrupted_source_ids
    ) == {target}


def test_default_rule_is_unchanged_when_no_ids_are_named():
    """Every pre-A1 config must keep its exact meaning."""
    cfg = ExperimentConfig(**_batch_kwargs(
        attack={"name": "primary", "f": 1, "budget_ratio": 0.5}
    ))
    assert cfg.attack.corrupted_source_ids is None
    assert select_corrupted_sources(
        cfg.sources.sources, cfg.attack.f, cfg.attack.corrupted_source_ids
    ) == {"batch_1"}


@pytest.mark.parametrize("target", BATCH_IDS)
def test_naming_a_source_does_not_reorder_the_source_list(target):
    """The RNG-stream invariant. `ParallelBatchSourceCollector` zips
    `cfg.sources.sources` against `SeedSequence(seed_entropy).spawn()`
    children by position; if attacking `batch_2` required moving it to
    the front, it would inherit `batch_1`'s stream and the clean/attacked
    pair would no longer share source seeds."""
    cfg = ExperimentConfig(**_batch_kwargs(
        attack={"name": "primary", "f": 1, "budget_ratio": 0.5,
                "corrupted_source_ids": [target]}
    ))
    assert [s.source_id for s in cfg.sources.sources] == BATCH_IDS


@pytest.mark.parametrize("target", BATCH_IDS)
def test_only_the_named_source_is_shifted_and_by_exactly_minus_B(target):
    """§16's mechanism identity at the hook itself: the attacked source's
    residual moves by -B and the other two do not move at all."""
    d, B = 25.0, 12.5
    cfg = ExperimentConfig(**_batch_kwargs(
        attack={"name": "primary", "f": 1, "budget_ratio": 0.5,
                "corrupted_source_ids": [target]}
    ))
    clean = {"batch_1": 3.0, "batch_2": -1.5, "batch_3": 0.25}
    out = apply_attack(cfg.attack, clean, {target}, k=0, d=d,
                       rng=np.random.default_rng(0))
    for sid in BATCH_IDS:
        expected = clean[sid] - (B if sid == target else 0.0)
        assert out[sid] == pytest.approx(expected, abs=1e-12)
    # and the mean aggregate moves by exactly -B/M
    assert (np.mean(list(out.values())) - np.mean(list(clean.values()))) == pytest.approx(
        -B / 3, abs=1e-12
    )


def test_unknown_source_id_is_rejected_at_config_time():
    with pytest.raises(ValidationError, match="do not exist"):
        ExperimentConfig(**_batch_kwargs(
            attack={"name": "primary", "f": 1, "budget_ratio": 0.5,
                    "corrupted_source_ids": ["batch_7"]}
        ))


def test_count_must_equal_f():
    with pytest.raises(ValidationError, match="attack.f"):
        ExperimentConfig(**_batch_kwargs(
            attack={"name": "primary", "f": 1, "budget_ratio": 0.5,
                    "corrupted_source_ids": ["batch_1", "batch_2"]}
        ))


def test_duplicates_are_rejected():
    with pytest.raises(ValidationError, match="duplicates"):
        ExperimentConfig(**_batch_kwargs(
            attack={"name": "primary", "f": 2, "budget_ratio": 0.5,
                    "corrupted_source_ids": ["batch_1", "batch_1"]}
        ))
