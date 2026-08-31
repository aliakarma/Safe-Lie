"""Deterministic seeding with independently reproducible child streams.

Report reference: Phase 0 (PROJECT_REPORT.md); smoke test S3.

A single global seed is not enough for this project: the attack module,
the environment, policy initialization, and evaluation sampling must be
independently reproducible so that (for example) the attack schedule can
be held fixed while policy initialization is varied across seeds. We
derive all child seeds from one `numpy.random.SeedSequence` via its
`spawn` mechanism, which is the numpy-documented way to get independent,
non-overlapping streams from one root seed.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedBundle:
    """Per-component seeds derived from one root seed.

    Each field is an independent `numpy.random.Generator` seed (an int)
    suitable for `np.random.default_rng(seed)` or for seeding torch.
    """

    root: int
    env: int
    policy_init: int
    attack: int
    eval: int
    torch: int

    def rng(self, component: str) -> np.random.Generator:
        return np.random.default_rng(getattr(self, component))


def seed_everything(seed: int, deterministic: bool = True) -> SeedBundle:
    """Seed python, numpy, and torch (CPU + CUDA if present).

    Returns a :class:`SeedBundle` of independently reproducible child
    seeds for env / policy-init / attack / eval randomness. This is the
    single entry point every script and test must call before touching
    any other component.
    """
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(5)
    bundle = SeedBundle(
        root=seed,
        env=int(children[0].generate_state(1)[0]),
        policy_init=int(children[1].generate_state(1)[0]),
        attack=int(children[2].generate_state(1)[0]),
        eval=int(children[3].generate_state(1)[0]),
        torch=int(children[4].generate_state(1)[0]),
    )

    random.seed(bundle.root)
    np.random.seed(bundle.root % (2**32 - 1))
    os.environ["PYTHONHASHSEED"] = str(bundle.root)

    try:
        import torch

        torch.manual_seed(bundle.torch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(bundle.torch)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return bundle
