# Unconstrained MAPPO — the constraint-binding control

**Status: PRE-DECLARED. Written 2026-09-12, before any run of this
condition exists.** Nothing below was chosen after seeing an unconstrained
number, because no unconstrained number exists yet. The clean constrained
values quoted in §6 are already committed (G9/G10, analysed in
`results/runs_a1/a1_report.json`) and are reproduced here only so that the
comparison is fixed in advance rather than assembled afterwards.

Condition label: **U** (unconstrained). It joins the existing labels
A (clean, constrained), B (attacked), C (attacked + RCE), D (zero-mean
perturbation), E (clean + RCE).

---

## 1. Question

**Does the safety constraint actually influence the constrained learner?**

Every empirical claim in this paper is a claim about a *dual* variable
being moved by corrupted cost feedback. That claim is vacuous if the dual
was never doing any work in the first place — if the policy that G10
learned would have been learned anyway, the "attack" is moving a number
that does not reach the policy, and A1's +3.99 cost increase would have to
come from somewhere else.

A1 and A2 cannot answer this. Both are differences *between* constrained
runs; both hold the dual mechanism fixed and vary what is fed into it.
Neither contains a run in which the dual is absent, so neither can
establish that the dual is load-bearing. That is what this control is for,
and it is the only thing it is for.

## 2. Treatment

Unconstrained MAPPO: the identical learner with the multiplier pinned to
zero.

```
dual.lambda_max:  25.0  ->  0.0
```

This is the repository's existing mechanism for removing the constraint,
not a new one. `safelie.training.dual.dual_update` ends in
`np.clip(mixed + eta_lambda * residual, 0.0, lam_max)`; the multiplier is
initialised to `np.zeros(n_agents)` at `loop.py:129`. With `lam_max = 0.0`
the projection interval is the single point {0}, so `lambda` is exactly
0.0 at every round, for every agent, by construction rather than by
convergence. `safelie.training.ppo` forms
`combined_adv = adv_r - lam * adv_c`, which at `lam ≡ 0` is `adv_r`
identically — plain MAPPO on the task reward.

`DualConfig.lambda_max` carries no positivity constraint
(`src/safelie/utils/config.py:224`), so 0.0 loads and validates.

**Everything else is bit-for-bit the G10 clean config.** No observation
normalisation, no reward normalisation, no optimizer change, no entropy
change, no learning-rate change, no rollout-length change, no horizon
change. The dual update itself still executes unconditionally every round
(Proposition 3 liveness is structural and is not being disabled); it is
the projection that renders it inert.

**What is deliberately NOT switched off.** The source collector, the cost
critic, the aggregator and the residual computation all still run, and
the per-round `mechanism_reported_cost_return` and `constraint_residual`
are still logged. Three reasons, all of them pre-committed:

1. §7 of this document requires reporting the mechanism-reported cost and
   the residual for condition U. Those fields do not exist if collection
   is switched off.
2. Disabling collection would change what the run consumes from its RNG
   streams and break the common-random-number pairing against condition A
   (§4), which is the whole basis of the paired analysis.
3. "Smallest configuration change that genuinely creates the
   unconstrained condition" (the instruction this control was written
   under) is one field. Switching off collection would be four or five.

The cost of keeping collection on is ~82 % of the wall clock (§8) for data
that cannot influence the policy. That cost is accepted.

## 3. Seeds

`0, 1, 2`. Three seeds, matching A1 and A2, under the existing statistical
policy (decision D6, `MIN_SEEDS_FOR_INFERENCE = 5`). No seed will be added
after seeing a result. See §9.

## 4. Configuration, exactly

Three files, `configs/experiment/unconstrained/u_seed{0,1,2}.yaml`, each
being `configs/experiment/g10_batch_clean_seed{k}.yaml` with **four**
fields changed:

| field | clean constrained | control |
|---|---|---|
| `run_id` | `seed{k}` / `g9_batch_clean` | `U_seed{k}` |
| `output_dir` | `results/runs_constraint_batch_g10` | `results/runs_unconstrained` |
| `dual.lambda_max` | `25.0` | **`0.0`** |
| `source_collection.seed_entropy` | per-seed | **unchanged, copied verbatim** |

Only `lambda_max` is a scientific field. The other three are identity and
filing.

**Common random numbers.** `seed_entropy` is copied verbatim from the
paired clean run of the same seed — the same convention A1's `b_seed*.yaml`
and A2's `c_seed*.yaml` use:

| seed | `seed_entropy` | paired clean run |
|---|---|---|
| 0 | `286314957402113664887331205920951063913` | `results/runs_constraint_batch_g9/g9_batch_clean` |
| 1 | `52021175099534868945312500562751741008` | `results/runs_constraint_batch_g10/seed1` |
| 2 | `335268198726974212397240672597355197200` | `results/runs_constraint_batch_g10/seed2` |

The three entropies are distinct, so the source streams are not shared
across seeds; they are identical to the paired constrained run, so the
comparison at each seed is against matched draws rather than a second
sample. `scripts/g10_derive_source_entropy.py` already verified offline
that the three seeds' 23,100 env seeds share zero values pairwise.

Everything else — `manyagent_ant`, N=6, d=25.0,
`velocity_threshold=0.75`, `cost_mode=per_agent_velocity`, ring topology,
M=3 trajectory-batch sources at R_m=30, `attack: none`, `defense: mean`,
`eta_lambda=0.035`, clip 0.2, GAE 0.95, gamma 0.99, lr 3e-4, 4 epochs,
4 minibatches, hidden 64, `rollout_length=2000`, `total_steps=500000`
(250 rounds), `validation_rounds=[25,75,125,175,225]`, `R_ref=120`,
`workers=12`, `chunks_per_worker=4` — is inherited unchanged.

**Verification is on the loaded `ExperimentConfig`, not on the YAML text.**
`scripts/verify_unconstrained_config.py` loads each config through
`load_experiment_config`, asserts `cfg.dual.lambda_max == 0.0`, and
asserts field-by-field equality with the paired clean config on every
other field of every sub-model. A run may not start until that passes.

## 5. Integrity preconditions (checked before launch, recorded per run)

- Clean git working tree; the run's `run_metadata.json` records the sha
  and `dirty: false`.
- A config hash recorded per run, and the loaded-config diff against the
  paired clean config being exactly `{dual.lambda_max, run_id,
  output_dir, seed, source_collection.seed_entropy}`.
- Distinct output directory per seed under `results/runs_unconstrained/`.
- Distinct `seed_entropy` per seed (§4 table).
- `attack.name == "none"`, `attack.f == 0`, `defense.name == "mean"` —
  i.e. no attack and no RCE on any arm of this control.

## 6. Primary comparison

Paired by seed against the validated clean constrained runs (condition A),
the same three runs A1 and A2 already pair against.

**Primary quantity:**

    J_C,unconstrained − J_C,constrained  =  U − A

on whole-run network-average true cost (`true_cost_net_whole`), higher is
worse — the same metric, window and pairing as A1's primary contrast.

**The constrained side, already committed and frozen** (from
`results/runs_a1/a1_report.json`; reproduced here so the target is fixed
in advance):

| | seed 0 | seed 1 | seed 2 | mean (sd) |
|---|---|---|---|---|
| `true_cost_net_whole` | 25.2143 | 24.9550 | 24.9782 | **25.0492** (0.1435) |
| `true_cost_net_last50` | 25.8218 | 26.2197 | 24.5748 | 25.5388 (0.8582) |
| `task_return_last50` | −44.1922 | −42.5286 | −45.2154 | −43.9787 (1.3561) |
| `task_return_first20` | −181.9909 | −180.7243 | −174.6102 | −179.1084 (3.9468) |
| `violation_rate_whole` | 0.4733 | 0.4607 | 0.4647 | 0.4662 (0.0065) |
| `lambda_mean` | 1.5097 | 1.6000 | 2.1777 | 1.7625 (0.3624) |
| `mech_reported_whole` | 24.7660 | 24.9694 | 24.7851 | 24.8402 (0.1124) |

Note where the constrained system sits: the network-average true cost is
**25.05 against a budget of d = 25.0**. The per-agent spread at the same
time is wide — seed 0's whole-run per-agent costs are
(21.62, 21.57, 21.69, 23.92, 28.50, 33.99). The average is pinned; the
individuals are not. That is Proposition 1 (a connected doubly-stochastic
W pins only the network average) visible in the data, and it is the
reason the control's primary metric is the network average.

**Reported for the control** (the list this document is required to fix
in advance):

- network-average true cost — whole-run and final-50;
- per-agent true cost — whole-run and final-50, all six agents, per seed;
- task return — first-20, last-50, and learning gain;
- learning trajectory — the per-round return and cost series;
- lambda — mean, max, fraction positive, fraction saturated, per-agent
  final-50;
- constraint residual — per-agent final-50 and its sum;
- reported mechanism cost — whole-run and final-50;
- violation rate — whole-run and final-50;
- and, for continuity with Part 1's reanalysis, the mechanism-level
  detection gap `true_cost − mechanism_reported`.

## 7. What this control can and cannot establish

**Its purpose is not to prove safety.** An unconstrained run is not a
safety baseline and no safety claim rests on it.

Its purpose is to show that **removing the dual constraint changes the
learned safety-cost behaviour** — that the constrained learner's
J_C ≈ d is a consequence of the constraint rather than an accident of the
environment, the reward, or the velocity threshold.

If U − A is clearly positive, the constraint is behaviourally active, and
A1's claim that corrupting the dual's input raises true cost is a claim
about a live mechanism.

If U − A is near zero, then the environment's unconstrained optimum
already satisfies the budget, the dual is not load-bearing at this
operating point, and **A1 and A2 must be re-read in that light**. That
outcome will be reported plainly, in the paper, with the same prominence
as the other direction. Nothing in this document is contingent on the
sign.

**What it does not establish, in any outcome:** that the constraint is
*sufficient* for safety (the constrained runs violate on ~47 % of rounds
and three of six agents sit above d); that the effect size transfers to
another environment, topology, or budget; anything about f > 1, M ≠ 3, or
any aggregator other than the mean.

## 8. Gates

**There is no numerical effect-size gate, and none is being invented
here.** No pre-existing pre-declaration fixes a bar for U − A, and writing
one now — with the constrained side already in hand and the control not
yet run — would be choosing a bar against half-known data. The primary
quantity is reported as an estimate with per-seed values and a paired 95 %
CI, and interpreted in words per §7.

What *is* gated, as structural validity rather than outcome:

| gate | requirement |
|---|---|
| **U-S1** | Run completes 250 rounds; `rounds.jsonl` and `oracle.jsonl` both have 250 rows. |
| **U-S2** | All logged quantities finite — no NaN, no Inf. |
| **U-S3** | `lambda == 0.0` exactly, at every round, for every agent. `lambda_mean == lambda_max == 0.0`. |
| **U-S4** | `attack.name == "none"`, `corrupted_source_ids == []`, `defense.name == "mean"` in the run's own recorded config snapshot — no accidental attack, no accidental RCE. |
| **U-S5** | Loaded-config diff against the paired clean config is exactly the five fields of §4/§5. |
| **U-S6** | `run_metadata.json` records `git.dirty == false` and `status == "complete"`. |
| **U-S7** | 23,100 source env seeds recorded; zero shared with the other two control seeds. |

A structural failure halts the queue. These are validity checks on the
artifact, not on the result, so enforcing them cannot bias the estimate.

**Learning health** is reported, not gated (the learner is unchanged and
already validated by G10): PPO KL median and p95, entropy first-10 vs
last-10, and the task learning gain. A control whose learner diverged
would be caught by U-S2 regardless.

## 9. Statistical policy

Unchanged from the existing policy. n = 3.

- Paired by seed; per-seed differences reported individually.
- Paired effect size (Cohen's dz) and paired 95 % CI reported.
- The paired t is emitted labelled **SENSITIVITY ONLY**, per decision D6
  (`MIN_SEEDS_FOR_INFERENCE = 5`); `welch_t_test` refuses below five seeds
  and that is not being worked around.
- No power calculation will be retrofitted. No seed will be added after
  seeing a result — the same rule that makes A1's B−D permanently
  descriptive applies here.
- Language: the U−A contrast is described in effect-size and per-seed-sign
  terms, not in significance terms.

## 10. Execution order and cost

Sequential, seed 0 first. The source collector already occupies all 12
logical CPUs (`workers: 12, chunks_per_worker: 4`), so concurrent seeds
would not finish sooner and would make a per-run stop rule unusable — the
same reasoning `scripts/a1_run_queue.py` records.

Seed 0 runs and is verified against U-S1..U-S7 **before** seeds 1 and 2
start. A structural defect is then caught after one run rather than after
three.

Measured cost, from the paired clean runs on this machine: 27,211 –
36,390 s wall clock per run, of which ~115 s/round is source collection.
Budget ~8–10 h per seed, ~24–30 h for all three, sequential. 45,000,000
source environment steps + 500,000 PPO steps + 500,000 oracle steps per
seed.

## 11. Analysis

`scripts/analyze_unconstrained.py`, written against this document, pairing
U against A by seed and reporting exactly the §6 list. It will be written
before seed 0 finishes and will not be modified after any control number
is visible.
