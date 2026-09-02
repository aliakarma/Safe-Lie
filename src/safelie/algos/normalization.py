"""Running mean/std normalization for observations and value targets.

P0 #2 (no observation normalization) and P0 #3 (no reward/value
normalization): neither existed anywhere in the pipeline before this
module. Unnormalized observations and raw-return-scale MSE regression
targets, combined with a single small `grad_clip` shared across the
policy and both critics (see `safelie.algos.networks.AgentBundle`'s
previous single-optimizer design, fixed alongside this), is a standard
and well-documented failure mode for on-policy continuous-control RL: a
critic loss whose gradient scale is set by the raw return magnitude (up
to the budget `d`, i.e. potentially O(10)) dwarfs a small grad-norm clip
(0.5), so nearly all of the critic's true gradient direction is discarded
every update and the critic converges far slower than the rollout horizon
gives it rounds to do so.

`RunningMeanStd` is the standard (Chan et al. 1979 parallel-variance,
as used in OpenAI Baselines' `VecNormalize` and Stable-Baselines3)
online, checkpointable, per-dimension mean/std tracker. It is used two
ways here, both load-bearing for keeping the return-scale semantics the
rest of the pipeline (GAE bootstrap, the source/aggregation/dual
pipeline, `d`-comparisons) depends on intact:

  - **Observations**: normalized on every network input (policy and both
    critics), never denormalized -- the network only ever needs to see a
    consistent input distribution.
  - **Returns**: the critic *network* is trained to predict a normalized
    target; every PUBLIC query of a critic's value
    (`AgentBundle.value`/`.cost_value`, used by the rollout, the source
    pipeline, and `scripts/calibrate_cost.py`) denormalizes back to
    return scale before returning, so nothing outside the training loss
    itself (`safelie.training.ppo`) ever sees a normalized value. GAE
    (`safelie.training.gae`) therefore keeps receiving return-scale
    `values`/`cost_values`/`last_value`/`last_cost_value` exactly as
    before -- this module changes how the critic is *trained*, not the
    mathematical meaning of any value already flowing through the rest
    of the codebase.
"""

from __future__ import annotations

import numpy as np
import torch


class RunningMeanStd:
    """Online per-dimension mean/variance, updated in batches.

    `shape=()` gives a scalar tracker (used for the two return streams);
    `shape=(obs_dim,)` gives a per-dimension tracker (used for
    observations). `count` starts at `epsilon`, not 0, both to avoid a
    division by zero on the very first update and so a single early
    outlier batch cannot swing the estimate to its full value -- the same
    convention OpenAI Baselines' `RunningMeanStd` uses.
    """

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray | list[float]) -> None:
        """`x`: an array of samples, shape `(n, *shape)` or `(*shape,)`
        for a single sample (e.g. one round's `(T, obs_dim)` observations,
        or one round's `(T,)` returns)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.mean.shape):
            x = x[np.newaxis, ...]
        if x.shape[0] == 0:
            return
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def _mean_std_tensors(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.as_tensor(self.mean, dtype=dtype)
        std = torch.as_tensor(np.sqrt(self.var + 1e-8), dtype=dtype)
        return mean, std

    def normalize(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        mean, std = self._mean_std_tensors(x.dtype)
        return torch.clamp((x - mean) / std, -clip, clip)

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        mean, std = self._mean_std_tensors(x_norm.dtype)
        return x_norm * std + mean

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])
