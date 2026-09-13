# A3 production runbook — GCP `t2d-standard-60`

Operational instructions for the A3 campaign. The science is fixed by
[`a3_gates.md`](a3_gates.md); §15 there records the platform amendment this
runbook implements. **Nothing here may change an experimental parameter.**

---

## 1. Instance

| setting | value | why |
|---|---|---|
| machine type | **`t2d-standard-60`** | 1 vCPU = 1 **physical** Milan core (no SMT), so the advertised count is real parallelism. Milan-only, so the CPU platform is pinned by the machine type and cannot vary between runs. |
| zone | any with T2D capacity (e.g. `us-central1-a`) | — |
| provisioning | **standard, NOT spot/preemptible** | The resume path has never been exercised under preemption and A3 is pre-declared. See §15.6 of the gates doc. |
| boot disk | 50 GB balanced PD | Artifacts are ~10 MB/run, ~120 MB for the campaign. The OS, Python and MuJoCo dominate. |
| image | Debian 12 / Ubuntu 22.04 | — |

```bash
gcloud compute instances create safelie-a3 \
    --machine-type=t2d-standard-60 \
    --zone=us-central1-a \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-balanced \
    --image-family=debian-12 --image-project=debian-cloud
```

Do **not** substitute another CPU family. N2/C3/C3D sell hyperthreads, so
"60 vCPU" there is 30 physical cores and roughly half the throughput per
rented vCPU; they also require `--min-cpu-platform` to pin the silicon.

---

## 2. Environment

MuJoCo needs a headless GL stack; everything else is the repo's own
dependencies.

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv git libgl1 libglew-dev libosmesa6-dev

git clone <repo> safelie && cd safelie
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e .
```

### Thread settings — required

Each source worker is single-threaded by design (`_init_worker` calls
`torch.set_num_threads(1)`), but that does not cover BLAS in the **parent**
process or in libraries that read the environment at import time. With three
seeds at 20 workers each -- 60 worker processes on 60 cores, every core
committed -- an unpinned BLAS spawning its own pool per process is the single
easiest way to lose most of the machine to context switching. At this worker
count there is no core headroom, so these settings are load-bearing rather
than merely advisable.

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

Set these in the shell that launches the queue, before any Python starts.
They are recorded into every run's `run_metadata.json` under
`provenance.thread_env`, so a run launched without them is identifiable
afterwards rather than silently slower.

Optionally pin the machine type for the provenance record (the metadata
server is probed automatically, so this is only needed if that is blocked):

```bash
export SAFELIE_MACHINE_TYPE=t2d-standard-60
```

---

## 3. Pre-flight — run all of this before launching

```bash
# 1. the twelve configs are the declared A3 design (A3-G1-i)
python scripts/a3_verify_frozen.py

# 2. scheduling, locking and isolation behave
python -m pytest tests/unit/test_a3_queue_isolation.py \
                 tests/unit/test_a3_verify_frozen_workers.py -q

# 3. the mechanism is intact on this machine
python -m pytest tests/property/test_aggregators.py tests/unit/test_source_batch.py -q

# 4. the queue walks correctly without starting anything
python scripts/a3_run_queue.py --seed 0 --dry-run
python scripts/a3_run_queue.py --seed 1 --dry-run
python scripts/a3_run_queue.py --seed 2 --dry-run

# 5. the tree is clean -- run this LAST, after steps 1-4
git status --porcelain
```

All four checks must pass. `a3_verify_frozen.py` must print `ALL CONFIGS
PASS`.

**Step 5 must print nothing.** `safelie.experiment._git_sha` records `dirty`
and `dirty_paths` in every run's `run_metadata.json`, so whatever the tree
looks like at launch is stamped into all twelve production artifacts. It is
step 5 rather than step 1 because steps 2 and 4 are themselves writers: the
queue's per-seed status files, its lock files and its atomic-write
temporaries are rewritten by the dry run and cycled by the test fixture.
Those paths are gitignored precisely so this step can come after them and
still print nothing; if it prints something else, resolve it before
launching rather than after.

---

## 4. Launch

All three seeds concurrently, 20 workers each — 3 x 20 = 60, that is all 60
vCPUs, and no phase leaves the instance half idle.

```bash
cd ~/safelie && . .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p results/runs_a3/logs

nohup python scripts/a3_run_queue.py --seed 0 > results/runs_a3/logs/queue_seed0.out 2>&1 &
nohup python scripts/a3_run_queue.py --seed 1 > results/runs_a3/logs/queue_seed1.out 2>&1 &
nohup python scripts/a3_run_queue.py --seed 2 > results/runs_a3/logs/queue_seed2.out 2>&1 &
```

Each process runs its seed's four conditions in order `A' -> B' -> C' -> E'`
and gates each run before starting the next. A stop condition halts only that
seed; the other two continue, and each remains a complete analysable design on
its own.

Start all three together. Launching a fourth queue process, or raising
`workers` above 20 while three seeds are running, oversubscribes the instance
and slows every seed.

### Monitoring

```bash
python - <<'PY'
import glob, json
for p in sorted(glob.glob("results/runs_a3/a3_queue_status_seed*.json")):
    st = json.load(open(p))
    print(f"--- {p} --- halted: {st.get('halted')}")
    for name, r in st.get("runs", {}).items():
        print(f"   {name:10s} {str(r.get('status')):12s} gates={r.get('gates_pass')}")
PY
```

Per-round progress: `wc -l results/runs_a3/*/rounds.jsonl` (250 = complete).

### If a seed halts

The queue stops that seed and writes the reason to its status file. Do not
restart it blindly — a halt means a section-7 stop condition fired, and the
implementation must be fixed before continuing. Re-running the same command
resumes from the last completed round once the cause is addressed.

If a queue process is killed, its lock file remains. Confirm nothing is still
running for that seed, then remove it:

```bash
rm results/runs_a3/.a3_queue_seed0.lock
```

---

## 5. After the campaign

```bash
# homogeneity check -- all twelve runs on one platform
python - <<'PY'
import json, glob
seen = {}
for p in sorted(glob.glob("results/runs_a3/[ABCE]_seed*/run_metadata.json")):
    pr = json.load(open(p)).get("provenance", {})
    key = (pr.get("machine_type"), pr.get("architecture"), pr.get("cpu_model"),
           json.dumps(pr.get("libraries"), sort_keys=True))
    seen.setdefault(key, []).append(p.split("/")[-2])
print(f"{len(seen)} distinct platform signature(s) across {sum(len(v) for v in seen.values())} runs")
for k, v in seen.items():
    print("  ", k[0], k[1], k[2], "->", v)
PY

python scripts/analyze_a3.py
```

The homogeneity check must report **exactly one** distinct platform signature.
The analyser requires all twelve production runs and reads only
`results/runs_a3/`.

---

## 6. Expected cost

Projected from the measured Windows anchor (0.159 traj/s per physical Zen 3
core, 157.2 s/round at M=5), clock-scaled to Milan. Treat as ±25%.

| | value |
|---|---|
| trajectories, whole campaign | 475,200 |
| core-hours of source collection | ~667 |
| source env-steps per run | 75,000,000 |
| PPO env-steps per run | 500,000 |
| source env-steps, whole campaign | 900,000,000 |

| phase | cores used | trajectories | wall clock |
|---|---|---|---|
| single phase — seeds 0+1+2 concurrent, 20w each | 60 | 475,200 | **~24-29 h** |
| **total** | | 475,200 | **~24-29 h** |

This is the first of the two recovery options the earlier revision of this
section listed and declined. It is taken now, and the reason it can be taken
is that the declared value was changed in the pre-declaration itself
(`a3_gates.md` §16) rather than by this runbook — a runbook still may not
change a declared value.

What it buys and what it costs:

* **buys** the idle half of the instance. The previous plan ran seeds 0+1 at
  30 workers and then seed 2 alone, leaving 30 of 60 cores idle for the whole
  second phase — seed 2 did a third of the work but took as long as the first
  two together (~33-42 h in total).
* **costs** about 6% to granularity: 150 trajectories over 20 workers is
  `ceil(150/20) = 8` waves against 7.5 ideal, where 30 workers divided the
  round exactly into 5. It also leaves **no core headroom**, which is why the
  thread settings in §2 are mandatory rather than advisory.

The trade is favourable because the granularity loss is ~6% of one phase while
the idle half of P2 was ~30% of the campaign.

The second option — raising seed 2's configs to `workers: 50` once it is the
only queue running — is now moot, since no seed runs alone.

Per-round timing appears in each run's `run_metadata.json` under
`timing.source_s_per_round`. At 20 workers expect **~67 s/round**
(`ceil(150/20) = 8` waves against ~42 s at 30 workers' 5 waves); materially
above that means the machine is underperforming the projection or the thread
settings did not take. Check this on the **first** completed round of the
first seed, not at the end — the Windows smoke anchor measured 0.10-0.12
traj/s/core against the 0.159 this projection assumes, so the first real
GCP round is the first honest data point on throughput.
