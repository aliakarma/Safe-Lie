# Training

## Running an experiment

```bash
python scripts/train.py --config configs/experiment/local_demo_attack.yaml
```

This loads and validates the config (raising immediately on any
malformed or scientifically-invalid setting — see below), then calls
`safelie.experiment.run_experiment_with_oracle`, which runs the learner
(`safelie.training.loop.ExperimentRun`) and, every `--eval-every` rounds
(default: every round), an independent oracle evaluation episode.
Artifacts land in `<output_dir>/<run_id>/`:

- `rounds.jsonl` — one record per round, written by the learner: raw
  per-source reports, which sources were corrupted, the aggregate
  (point estimate, spread, retained count, whether the retained set was
  degenerate), the multiplier before/after, task return, and the
  learner's own reported cost-return estimate.
- `oracle.jsonl` — one record per evaluated round, written by the
  orchestrator using the withheld evaluator: true cost return, peak true
  cost, whether the true budget was violated, the running violation rate,
  and the detection gap (`true - reported`).

## Config schema

Every experiment config (`configs/experiment/*.yaml`) validates against
`safelie.utils.config.ExperimentConfig` (Pydantic v2). Key sections:

| Section | Meaning |
|---|---|
| `env` | Which environment, agent count, budget `d`, horizon, obs/action dims |
| `topology` | Consensus mixing matrix: `complete`, `ring`, `star`, `erdos_renyi`, `identity`, `shared_constraint` |
| `sources` | The list of M reporting sources, each with a `source_type` and an `independence_class` |
| `attack` | `none` / `primary` / `stealth` / `benign_control`, plus the four taxonomy axes |
| `defense` | `mean` / `coordinate_median` / `krum` / `trimmean` / `rce`, plus `f`, `beta`, degenerate-case handling |
| `ppo` | PPO clip, GAE lambda, gamma, learning rate, and the pinned-but-unspecified-by-the-paper hyperparameters (epochs, minibatches, hidden dim) |
| `dual` | `eta_lambda`, `lambda_max` |
| `total_steps`, `rollout_length` | Training duration |

Validation failures raise immediately, before any environment, policy, or
source is constructed — for example:

```bash
python -c "
from safelie.utils.config import ExperimentConfig
ExperimentConfig(run_id='x', env={'name':'synthetic_constrained_marl','n_agents':6,'budget':25.0},
                  topology={'name':'ring','n_agents':6},
                  sources={'sources':[{'source_id':'a','source_type':'own_critic','independence_class':'ic1'}]},
                  defense={'name':'rce','f':1}, total_steps=1000)
"
# ValueError: Defense 'rce' requires effective_M > 2f ... effective_M=1 and f=1
```

## The shipped configs

| Config | Environment | Purpose |
|---|---|---|
| `smoke.yaml` | synthetic | CPU smoke test, seconds |
| `local_demo_clean.yaml` | synthetic | No-attack reference baseline |
| `local_demo_attack.yaml` | synthetic | f=1 persistent under-reporting, undefended (mean) |
| `local_demo_rce.yaml` | synthetic | Same attack, RCE defense |
| `local_demo_benign.yaml` | synthetic | The falsification control: unbiased noise |
| `pilot_A_clean.yaml` .. `pilot_E_clean_rce.yaml` | `manyagent_ant` | The literal Stage-2 Colab pilot matrix (`PROJECT_REPORT.md` §R8) — **validates but does not run** without the MuJoCo adapter |

The `local_demo_*` configs use `budget=5.0`, not the paper's `d=25`: see
[assumptions.md](assumptions.md) for why (the synthetic environment's
cost scale differs from Safe MAMuJoCo's, and at `d=25` the constraint
never binds within a laptop-feasible run — the exact "constraint not
binding" confound `PROJECT_REPORT.md` §R6.1 warns about).

## Checkpointing and resuming

```python
from pathlib import Path
from safelie.training.loop import ExperimentRun
from safelie.utils.config import load_experiment_config

cfg = load_experiment_config("configs/experiment/local_demo_attack.yaml")
run = ExperimentRun(cfg)
for _ in range(10):
    run.run_round()
run.checkpoint(Path("results/runs/local_demo_attack/checkpoint.pt"))
```

A checkpoint captures every source of randomness touched between rounds,
not just model weights: the environment/attack seed generators, each
diversified-replica source's own network and optimizer state, and —
easy to miss — the *global* torch and numpy RNGs that
`safelie.training.ppo` (minibatch shuffling) and action sampling read from
implicitly. `tests/smoke/test_determinism.py::test_checkpoint_restore_continues_bitwise_identically`
verifies a restored run continues bitwise-identically to an uninterrupted
one — this was the one non-trivial bug found and fixed while building
this repository (see [SMOKE_TEST_REPORT.md](../SMOKE_TEST_REPORT.md)).

## What the training loop deliberately does not do

`safelie.training.loop.ExperimentRun` never imports `safelie.eval.oracle`
and never reads true cost — see [architecture.md](architecture.md)'s
isolation-boundary section. If you are extending the training loop,
do not add an oracle reference to it; add the metric to
`safelie.eval.harness` / `safelie.experiment` instead, or the isolation
test (`tests/isolation/test_oracle_isolation.py`) will fail your PR, by
design.
