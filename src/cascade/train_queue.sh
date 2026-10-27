#!/bin/bash
# Train level-1 models sequentially (4 cores): waits for the harness baseline, then one model at a time.
# usage: bash src/cascade/train_queue.sh cond pairs xgb ovr mlp cat
cd "$(dirname "$0")/../.."
while [ ! -f results/probs_baseline.npz ]; do sleep 20; done
for m in "$@"; do
  ( time python src/cascade/train_base.py $m ) > results/logs/train_$m.log 2>&1
done
