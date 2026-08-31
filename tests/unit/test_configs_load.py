"""Every shipped experiment config must parse and validate.

This does not mean every config is *runnable* locally: `env.name:
manyagent_ant` configs (the Stage-2 pilot specs) validate successfully as
configuration but raise NotImplementedError only when actually executed,
because safelie.envs.mamujoco is intentionally not implemented (see that
module's docstring). This test catches config-schema regressions, not
runtime capability.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from safelie.utils.config import load_experiment_config

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "experiment"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_experiment_config_validates(path):
    cfg = load_experiment_config(str(path))
    assert cfg.run_id
    assert cfg.sources.effective_M >= 1
