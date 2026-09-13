# A3 cross-platform probe — Mac mini report

Session scope: setup, verification, one 15-minute probe, this report.
**No A3 production experiment was run.** Nothing under `results/runs_a1/`,
`results/runs_a2/`, `results/runs_constraint_batch_g9/`,
`results/runs_constraint_batch_g10/` or `results/a2_mechanism_check/` was
modified; no `docs/*_gates.md`, config, threshold, seed or `seed_entropy`
was edited; `workers` was left at 12 as declared.

Probe run: 2026-09-09, git `6e38718` (clean except the new probe outputs).

---

## A. Machine

| item | value |
|---|---|
| Model | Mac mini (`Mac16,10`), model number MU9E3AB/A |
| OS | macOS 26.5, build 25F71 (`macOS-26.5-arm64-arm-64bit`) |
| Chip | Apple M4 |
| Total cores | 10 |
| **Performance cores** | **4** |
| **Efficiency cores** | **6** |
| Logical CPUs | 10 (no SMT; 4P/6E physical = logical) |
| RAM | 16 GB (17,179,869,184 bytes) |
| Architecture | arm64 |
| Python | 3.11.15 (Homebrew, `main, Mar 3 2026`, Clang 21.0.0) |

The probe config declares `workers: 12` against 10 physical cores. Per HARD
RULE 5 this was **not** changed. The run is therefore oversubscribed 12:10,
which affects wall clock only and cannot affect a number — the collector's
per-worker checksum equality check passed on every round
(`worker_checksums_all_match: true`, 48 chunks/round).

## B. Environment

Windows reference column is the set given in the task brief, plus
`run_metadata.json` from the committed reference run.

| package | Windows | this Mac mini | differs? |
|---|---|---|---|
| Python | 3.11.9 (MSC v.1938 64-bit AMD64) | **3.11.15** (Clang 21.0.0) | **YES — patch version** |
| torch | 2.12.1+cpu | **2.12.1** | build tag only (see note) |
| numpy | 2.2.6 | 2.2.6 | no |
| scipy | *not stated* | 1.13.0 | **unverifiable** (see note) |
| mujoco | 3.12.0 | 3.12.0 | no |
| gymnasium | 1.3.0 | 1.3.0 | no |
| gymnasium-robotics | 1.4.2 | 1.4.2 | no |
| pydantic | *not stated* | 2.13.4 | **unverifiable** (see note) |

Also installed, matching `requirements.txt`: pyyaml 6.0.3, networkx 3.6.1,
pandas 2.3.3, statsmodels 0.14.5, pytest 9.0.3.

**Notes, because a version mismatch is a plausible cause of a probe
difference and must not be silently absorbed:**

1. **Python 3.11.9 vs 3.11.15 is a real difference** and is the one version
   delta I can confirm. It is a CPython patch release; it is not a plausible
   cause of a float-path difference on its own, but it is recorded here
   rather than absorbed. 3.11.9 is not installable from Homebrew on this
   machine (only 3.11.15 is offered), so matching it exactly would have
   required building CPython from source — not done, and flagged rather than
   worked around.
2. **`torch 2.12.1+cpu` vs `torch 2.12.1`** is the same upstream version. The
   `+cpu` local tag is the Windows CPU-only wheel; on macOS arm64 the default
   PyPI wheel *is* CPU-only and carries no local tag. Same version, different
   build for a different architecture — which is precisely the thing the
   probe exists to measure, not a setup error.
3. **scipy and pydantic were not stated for the Windows box.** I installed
   the `requirements.txt` pins (scipy 1.13.0, pydantic 2.13.4), which that
   file documents as the versions the repo was built and tested against. I
   cannot verify these against the Windows machine and am reporting the gap
   rather than resolving it. Neither is on the source-collection float path.
4. `pip install -e ".[dev]"` alone resolves to **torch 2.14.0 / numpy 2.4.6 /
   pandas 3.0.5 / scipy 1.17.1**, which would have introduced four more
   version deltas against Windows. I then installed `requirements.txt` to pin
   torch 2.12.1 and numpy 2.2.6 and match the Windows box. This is the
   "match versions where you can" instruction; no config, threshold or seed
   was touched.

## C. Setup problems

**None.** Every package installed cleanly and MuJoCo built and ran headless
on Apple Silicon with no `glfw` workaround required:

```
mujoco import OK 3.12.0
mj_step OK, qpos0: [ 0.000e+00  0.000e+00 -3.924e-05]
```

One benign, non-blocking notice is printed by `gymnasium-robotics` at import,
once per worker process (so ~13x per run):

```
AdroitHandRelocateDense-v1, AdroitHandHammerDense-v1, AdroitHandDoorDense-v1
environment's reward functions were updated in v1.2.1 without an environment
version update. Therefore, use gymnasium-robotics==1.2.0 for v1
reproducibility or use v2 in gymnasium-robotics>=1.4.3.
```

It concerns Adroit hand environments, which this project does not use
(`env.name = manyagent_ant`). It is emitted identically on the Windows box's
pinned `gymnasium-robotics==1.4.2` and is not a setup problem.

## D. Test suite

```
258 passed, 6 warnings in 28.36s
```

**258/258 passed, 0 failed.** No failing test to name. The 6 warnings are the
documented M=3 degenerate-retained-set `RuntimeWarning` from
`src/safelie/defenses/__init__.py:41` — the expected behaviour per
`docs/a2_rce_gates.md` §4, not a defect.

## E. Probe result

Full console output of `scripts/a3_platform_probe.py`, verbatim:

```
======================================================================
A3 CROSS-PLATFORM PROBE
======================================================================
  reference : results/a2_mechanism_check/mech_rce_clean
  candidate : results/a3_platform_probe/probe   (arm64, macOS-26.5-arm64-arm-64bit)
  rounds compared: 4

  [OK ] level 0  same experiment
  [OK ] level 1  source seeds identical
  [-- ] level 2  policy init identical at round 0
  [-- ] level 3  source values identical at round 0 (max |diff| = 8.856e-01)

  VERDICT: NOT bit-identical -> BRANCH-B
           A3 splits by whole SEED, 4 runs per machine, 2 seeds, ~2.2 days.
           Every contrast stays on one machine. This is the expected
           outcome across x86-64 and ARM64 and is not a failure.

-> results/a3_platform_probe/a3_platform_probe.json
```

### The four level verdicts

| level | tests | verdict |
|---|---|---|
| **0 — same experiment** | `seed`, `total_steps`, `rollout_length`, `seed_entropy` agree | **PASS.** The two runs are the same experiment. Only `run_id`/`output_dir` differ, as permitted. |
| **1 — source seeds** | `SeedSequence -> PCG64 -> integers` | **MATCH, all 4 rounds.** Integer path is platform-independent as designed. **This is the critical one: it rules out a config problem**, so the branch may be chosen. |
| **2 — policy checksum** | `torch.manual_seed` -> network init | **DIFFERS**, at round 0 and at all rounds. PyTorch does not initialise bit-identically across x86-64 and ARM64. |
| **3 — source values** | full MuJoCo + PyTorch float path | **DIFFERS.** `round0_max_abs_diff = 0.8855680270579285`. |

Level 2 checksums at round 0:

```
reference (Windows) : f371a4f1fea262d9ccb981638ade88eb2d23b916e2de25ba062ea68638868600
candidate (Mac mini): a0b04d72983fc0e092deca8ec027115185315a46028922e700758e30949d0f54
```

### Exact `round0_max_abs_diff`

```
round0_max_abs_diff  = 0.8855680270579285
round0_mean_abs_diff = 0.44845150740594675
tolerance            = 1e-09
```

Per-round max |diff| across all 4 rounds:

| round | max abs diff |
|---|---|
| 0 | 0.8855680270579285 |
| 1 | 3.102668844832202 |
| 2 | 1.251873820082885 |
| 3 | 1.3990073943191206 |

`all_rounds_max_abs_diff = 3.102668844832202`.

## F. Branch

Quoted verbatim from the tool:

> **`BRANCH-B`**

> ```
> VERDICT: NOT bit-identical -> BRANCH-B
>          A3 splits by whole SEED, 4 runs per machine, 2 seeds, ~2.2 days.
>          Every contrast stays on one machine. This is the expected
>          outcome across x86-64 and ARM64 and is not a failure.
> ```

The probe did **not** report INVALID. Level 1 matched, which is the condition
`docs/a3_gates.md` §9 sets for a branch decision to be meaningful at all: the
source seeds are identical, so the configs agree and the observed differences
are a property of the hardware, not of the setup.

Per §4, BRANCH-B means: each machine runs **all four conditions** of the seeds
it owns; 8 runs over 2 seeds (seed 0 and seed 1; **seed 2 is dropped**);
~2.2 days critical path. A3-G3 mechanism claims stay primary and fully
powered; A3-G5's interaction is reported **DIRECTIONAL ONLY** with an explicit
`n = 2, directional` marker. This is a pre-declared, expected outcome and is
not a failure.

## G. Timing

### What is and is not instrumented

Per-round **total** wall clock is not logged anywhere. `rounds.jsonl` records
`source_batch.wall_clock_s` per round; `run_metadata.json` records run-level
aggregates only (`s_per_round = wall_clock_s / rounds_run`,
`src/safelie/experiment.py:269`). The per-round numbers below are therefore
source-collection time, which is the dominant term, plus the run-level
aggregate. I am not synthesising a per-round total that was never measured.

### The 4 probe rounds — `source_batch.wall_clock_s` (this Mac mini)

| round | source_batch.wall_clock_s | reference_wall_clock_s |
|---|---|---|
| 0 | 33.563 | 0.0 |
| 1 | 29.634 | 0.0 |
| 2 | 30.766 | **41.037** (validation round) |
| 3 | 28.822 | 0.0 |

```
mean   = 30.696 s
median = 30.200 s
```

### Calibration round (1 round, `calib_probe`)

```
source_batch.wall_clock_s = 34.038 s
```

`calib_probe/run_metadata.json` is not written, so the calibration phase has
no recorded total wall clock. Measured externally, the whole invocation took
**219.7 s** (train.py's own figure; wall-clock 04:22:00 -> 04:25:40 = 220 s),
of which the main run's `wall_clock_s` is 180.441 s, leaving **~39.3 s** for
the calibration phase plus process setup.

### Run-level aggregates, this machine vs the Windows reference

| quantity | Windows `mech_rce_clean` | Mac mini `probe` | ratio |
|---|---|---|---|
| `wall_clock_s` (4 rounds) | 530.353 | **180.441** | 2.94x |
| **`s_per_round`** | **132.588** | **45.110** | **2.94x faster** |
| `source_s_per_round` | 115.616 | 40.956 | 2.82x |
| source_batch mean/round | 86.586 | **30.696** | 2.82x |
| source_batch median/round | 85.407 | **30.200** | 2.83x |
| `oracle_s` | 26.142 | 6.563 | 3.98x |
| `learner_total_s` | 504.097 | 173.840 | 2.90x |
| calibration round source | 90.336 | 34.038 | 2.65x |

Both runs did identical work: `env_steps = {ppo: 8000, source: 720000,
oracle: 8000}`, 90 trajectories/round, 48 chunks/round.

**On the brief's "~105 s/round" figure.** No committed artifact is at exactly
105 s/round. The committed probe *reference* (`mech_rce_clean`) is
**132.588 s/round**, and that is the run this probe was compared against. The
Windows 250-round production runs span **103.96–158.70 s/round** (median
109.94): `E_seed1` 103.96, `C_seed1` 108.83, g10 `seed2` 108.84, `E_seed0`
109.94, g10 `seed1` 140.23, g9 `g9_batch_clean` 145.56, `C_seed0` 158.70. So
"~105 s/round" describes the fast end of the A2 production runs, not the
probe reference. Reported rather than resolved — the ratios above are stated
against the reference actually used.

### Implied estimate for a 250-round M=5 run on this machine

M=5, `R_m=30` collects 150 trajectories/round against the probe's 90, so
source cost scales by **150/90 = 1.6667**. Decomposing the probe's measured
180.441 s:

| component | measured (4 rounds) | per round |
|---|---|---|
| source_batch collection | 122.785 | 30.696 |
| validation reference collection | 41.037 | (1 round in 4) |
| oracle eval | 6.563 | 1.641 |
| learner / dual / IO (residual) | 10.055 | 2.514 |
| **total** | **180.441** | **45.110** |

Scaling only the source term (the oracle, learner and dual do not depend
on M):

```
source/round at M=5   = 30.696 x 1.6667  = 51.16 s
non-source/round      = 1.641 + 2.514    =  4.15 s
per-round total       ~ 55.32 s
250 rounds            ~ 13,829 s         =  3.84 h
```

Plus, per run:

* **Validation reference collections.** `R_ref = 120` is a fixed config value
  and is **M-independent** (`src/safelie/utils/config.py:77`), so this term
  does not scale. Production uses `validation_rounds = [25,75,125,175,225]`,
  5 rounds x 41.04 s = **205 s (0.06 h)**.
* **The 20-round calibration phase, RCE runs only** (`C'` and `E'`):
  20 x (34.038 x 1.6667 + 4.15) = **1,218 s (0.34 h)**.

| run type | estimate on this machine |
|---|---|
| `A'` / `B'` (no calibration) | **~3.90 h** |
| `C'` / `E'` (with calibration) | **~4.24 h** |
| **BRANCH-B share: 4 runs (`A' B' C' E'`, one seed)** | **~16.3 h ≈ 0.68 days** |

For comparison, `docs/a3_gates.md` §10 predicts ~13 h/run and ~2.2 days
critical path from the Windows machine's measured rate. This machine's
measured 2.94x advantage puts it at **~4.2 h/run**, i.e. roughly **3.1x
faster than the §10 per-run figure**. Two caveats, both stated rather than
absorbed:

1. This extrapolates a 4-round run to 250 rounds. The probe's rounds are
   early-training and its per-round source time was still falling
   (33.6 -> 28.8 s); a 250-round run's mix may differ.
2. §10's ~2.2-day BRANCH-B critical path is set by the **slower** machine.
   Nothing here changes that. If the Windows box holds ~13 h/run, its 4 runs
   remain ~2.2 days and that is still the critical path; this machine would
   finish its 4 runs in ~0.7 days and idle. **A3's §4 seed assignment is
   pre-declared and this is not a proposal to change it** — it is reported
   because §4 says the split is chosen by measurement, and the measurement is
   that the two machines are not equally fast.

## H. Anything surprising

Four things, all measured on this machine, none of them predicted by the gate
docs.

**1. Level 2 fails, which undercuts the stated reason level 3's round 0 is
decisive.** `docs/a3_gates.md` §9 describes level 3 round 0 as decisive
because "round 0 is the only round where both runs are still under the same
policy, so a difference there is pure numerics rather than accumulated
divergence" — and `a3_platform_probe.json` repeats that sentence verbatim in
its own `level3_source_values.note`. But level 2 measures exactly whether the
round-0 policies are the same, and here they are **not**
(`round0_identical: false`). The two runs are under *different* policies at
round 0. So the round-0 difference of 0.886 is **not** attributable to pure
numerics; it is the combined effect of a different initial policy and the
float path, and the two cannot be separated from this probe's outputs. The
branch outcome is unaffected — BRANCH-B is selected on either reading, and
level 1 passing rules out a config problem — but the *interpretive* claim
attached to level 3 in §9 holds only when level 2 passes, and that condition
is not stated there. I am flagging this rather than editing §9, per HARD
RULE 3.

**2. The round-0 magnitude is macroscopic, not last-bit.** 0.886 is not a
rounding artifact. At round 0 the reference's source values run 36.85–62.62
with a between-source sd of 0.76–1.88 per agent. **The cross-machine
difference is the same order as the between-source dispersion the experiment
is built to measure** — roughly half a source-sd, ~2.3% relative, and ~15
orders of magnitude above float64 epsilon. This is consistent with point 1
(genuinely different starting weights) rather than with float-path drift
alone. It also means BRANCH-B is not a marginal call: the machines are not
"nearly identical", they are visibly different from the first round.

**3. The divergence does not grow monotonically over the 4 rounds.** Max
|diff| goes 0.886 -> 3.103 -> 1.252 -> 1.399. Under a pure accumulated-
divergence picture one would expect growth; instead it jumps at round 1 and
then settles. Four rounds is far too short to characterise this and I am not
extrapolating — recording it because §9's framing ("accumulated divergence"
in later rounds) implies a monotone picture the data does not show.

**4. The RCE mechanism quantities are bit-identical across the two machines,
even though the float path is not.** Over all 24 aggregation cells (4 rounds
x 6 agents), both machines produce exactly the same tuple:

```
(spread, retained_n, applied_margin, degenerate) = (0.001, 1, 0.0015, True)
```

and identical `guarantee_calibration.json` (`epsilon_offline = 0.001`,
disagreement mean/std/min/max = 0.001/0.0/0.001/0.001). This is a positive
cross-platform confirmation of `docs/a2_rce_gates.md` §4: at M=3, f=1 those
quantities are structural constants, provably independent of the input
values, so they survive a change of architecture that moves every underlying
float. It is consistent with §4 and does not contradict anything — but §4
proves it algebraically and had not, before now, been checked on a second
architecture. It also sharpens §4's point: at M=3 the margin is so inert that
it is invariant even to a change of CPU architecture.

**Not surprising, confirmed as expected:** the degenerate-retained-set
`RuntimeWarning` fired on every round, as `docs/a2_rce_gates.md` §4 predicts.

**One confound ruled out:** the reference run was produced at git
`60dd2ef`, this probe at `6e38718`. `git diff 60dd2ef 6e38718 -- src/` is
**empty** — the source tree is byte-identical between the two commits, so no
code change stands between the reference and this candidate. The only config
change is the addition of `configs/experiment/a3/_platform_probe.yaml`
itself.

---

### Artifacts

Committed locally as `824e508 chore(a3): cross-platform probe result from the
Mac mini` (10 files, not pushed):

```
results/a3_platform_probe/a3_platform_probe.json
results/a3_platform_probe/probe/{rounds,source_seeds,oracle,validation_reference}.jsonl
results/a3_platform_probe/probe/{run_metadata,guarantee_calibration}.json
results/a3_platform_probe/calib_probe/{rounds,source_seeds,validation_reference}.jsonl
```

This report (`MAC_REPORT.md`) is written but left **untracked**, matching the
instruction order (commit the probe artifacts, then write the report). Add it
to the commit if you want it carried with them.
