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
process or in libraries that read the environment at import time. With 30
worker processes on 60 cores, an unpinned BLAS spawning its own pool per
process is the single easiest way to lose most of the machine to context
switching.

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
```

All four must pass. `a3_verify_frozen.py` must print `ALL CONFIGS PASS`.

---

## 4. Launch

Two seeds concurrently, 30 workers each — that is all 60 vCPUs.

```bash
cd ~/safelie && . .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

nohup python scripts/a3_run_queue.py --seed 0 > results/runs_a3/logs/queue_seed0.out 2>&1 &
nohup python scripts/a3_run_queue.py --seed 1 > results/runs_a3/logs/queue_seed1.out 2>&1 &
```

Each process runs its seed's four conditions in order `A' -> B' -> C' -> E'`
and gates each run before starting the next. A stop condition halts only that
seed.

**Seed 2 starts when one of the first two finishes**, freeing 30 cores:

```bash
nohup python scripts/a3_run_queue.py --seed 2 > results/runs_a3/logs/queue_seed2.out 2>&1 &
```

Do not start seed 2 early. Three seeds at 30 workers would oversubscribe the
instance 1.5x and slow all three.

### Monitoring

```bash
for s in 0 1 2; do
  echo "--- seed $s ---"
  python - <<'PY'
import json, glob
for p in sorted(glob.glob("results/runs_a3/a3_queue_status_seed*.json")):
    st = json.load(open(p))
    print(p.split("_")[-1], "halted:", st.get("halted"))
    for name, r in st.get("runs", {}).items():
        print(f"   {name:10s} {r.get('status'):12s} gates={r.get('gates_pass')}")
PY
done
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
| P1 — seeds 0+1 concurrent, 30w each | 60 | 316,800 | **16.6-21.1 h** |
| P2 — seed 2 alone, 30w | 30 (**30 idle**) | 158,400 | **16.6-21.1 h** |
| **total** | | 475,200 | **~33-42 h** |

**P2 leaves half the instance idle**, and that is a real cost: seed 2 does a
third of the campaign's work but takes as long as the first two seeds
together. Two ways to recover it, neither configured and neither taken here
because the worker count is a declared value and this runbook does not change
declared values:

* run all three seeds concurrently at 20 workers each — ~24-29 h, but it
  costs 6% to granularity (`ceil(150/20) = 8` waves against 7.5 ideal) and
  leaves no core headroom;
* raise seed 2's four configs to `workers: 50` once it is the only queue
  running — ~28-37 h total. Legitimate under §15.3 (all four conditions of
  the seed still agree), but it is a config edit mid-campaign and would need
  recording.

Per-round timing appears in each run's `run_metadata.json` under
`timing.source_s_per_round`. At 30 workers expect **~42 s/round**
(`ceil(150/30) = 5` waves); materially above that means the machine is
underperforming the projection or the thread settings did not take.
