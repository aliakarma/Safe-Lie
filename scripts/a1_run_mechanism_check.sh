#!/usr/bin/env bash
# A1 §16 mechanism validation. Sequential: the source collector already
# uses all 12 logical CPUs, so overlapping the three arms would only make
# each slower and would perturb nothing scientific.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
for arm in clean attack noise; do
  echo "=== $arm $(date -u +%H:%M:%S) ==="
  "$PY" scripts/train.py \
    --config "configs/experiment/a1/_mechanism_check_${arm}.yaml" \
    --eval-every 1 \
    > "results/a1_mechanism_check/${arm}.log" 2>&1
  echo "   rc=$? $(date -u +%H:%M:%S)"
done
echo "ALL DONE"
