#!/bin/bash
# Generator end-to-end: render + featurize generated GBT and PKS cadences, validate against the real bundle, and test
# generated-PKS rows as extra training data in the bundle_eval harness (baseline in the same run).
# usage: src/gen_pipeline.sh [N] [TAG]      (N cadences per telescope, default 9000)
set -e
cd "$(dirname "$0")/.."
N=${1:-9000}; TAG=${2:-v1}
mkdir -p data/gen results logs
NT=2 python src/gen_make.py $N GBT 101 data/gen/gbt_$TAG.npz > logs/gen_gbt_$TAG.log 2>&1 &
NT=2 python src/gen_make.py $N PKS 202 data/gen/pks_$TAG.npz > logs/gen_pks_$TAG.log 2>&1 &
wait
python src/gen_validate.py data/gen/gbt_$TAG.npz data/gen/pks_$TAG.npz
python src/gen_aug_eval.py data/gen/pks_$TAG.npz
