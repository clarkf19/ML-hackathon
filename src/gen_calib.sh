#!/bin/bash
# One calibration round: balanced GBT sample (N/9 per class), fast feature sets, per-class comparison with REAL train.
# usage: src/gen_calib.sh TAG [N] [SETS]   (GEN_OVR env passes parameter overrides) -> results/calib_TAG.txt
cd "$(dirname "$0")/.."
TAG=$1; N=${2:-900}; SETS=${3:-phys,tex,rep,rep2,lvl}
python src/gen_make.py $N GBT 5 data/gen/calib_$TAG.npz --balanced --sets=$SETS > logs/calib_$TAG.log 2>&1
python src/gen_compare.py data/gen/calib_$TAG.npz > results/calib_$TAG.txt 2>/dev/null
python src/gen_patterns.py data/gen/calib_$TAG.npz 2>/dev/null | sed -n '/calib/,$p' >> results/calib_$TAG.txt
tail -12 results/calib_$TAG.txt
