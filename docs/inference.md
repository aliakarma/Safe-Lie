# Inference

"Inference" here means loading a trained checkpoint and rolling out the
policy for inspection — there is no separate serving stack, since this is
an on-policy RL research repository, not a deployed service.

## Loading a checkpoint and rolling out

```bash
python scripts/infer.py \
  --config configs/experiment/smoke.yaml \
  --checkpoint results/runs/smoke_clean/checkpoint.pt \
  --episodes 5
```

This restores an `ExperimentRun`'s agent weights via
`ExperimentRun.restore()`, then rolls out `--episodes` fresh evaluation
episodes using `safelie.eval.harness.evaluate_true_cost` — the same
withheld-oracle evaluation path the training loop's orchestrator uses,
so the reported numbers (true cost return, peak true cost, whether the
budget was violated) are directly comparable to what a completed run's
`oracle.jsonl` would show for the current policy.

Note: no checkpoint is written automatically by `scripts/train.py` /
`run_experiment_with_oracle` as shipped — call `ExperimentRun.checkpoint()`
explicitly (see [training.md](training.md)) if you want one to load here.

## Programmatic use

```python
from pathlib import Path
from safelie.eval.harness import evaluate_true_cost
from safelie.training.loop import ExperimentRun
from safelie.utils.config import load_experiment_config
from safelie.envs.synthetic import SyntheticConstrainedMarlEnv

cfg = load_experiment_config("configs/experiment/smoke.yaml")
run = ExperimentRun(cfg)
run.restore(Path("results/runs/smoke_clean/checkpoint.pt"))

def env_factory():
    return SyntheticConstrainedMarlEnv(
        n_agents=cfg.env.n_agents, budget=cfg.env.budget,
        obs_dim=cfg.env.obs_dim, action_dim=cfg.env.action_dim,
        horizon=cfg.rollout_length,
    )

result = evaluate_true_cost(
    env_factory=env_factory, agents=run.agents, gamma=cfg.ppo.gamma,
    budget=cfg.env.budget, rollout_length=cfg.rollout_length, seed=42,
)
print(result.true_cost_return, result.violated)
```

## On a real environment (Safe MAMuJoCo)

Once `safelie/envs/mamujoco.py` is completed (see that module's docstring
and [reproducibility.md](reproducibility.md)), the same `evaluate_true_cost`
function works unchanged — it is written against the `DualCostEnvWrapper`
protocol, not against the synthetic environment specifically. Only the
`env_factory` needs to change.
