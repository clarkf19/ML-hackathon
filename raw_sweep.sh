#!/bin/bash
# Residual-information sweep on the RAW cadences (cloud_prompts/1_residual_sweep.md). Needs data/signal-hunters-round2/*.npy
# (see CLOUD.md). ~208 raw-domain statistics on train / test / two fake-PKS raw renderings -> shift check -> residual scan
# against the current model's OOF probabilities -> harness score of the top candidates and of the full set vs baseline.
set -e
cd "$(dirname "$0")"
PY="${PY:-python}"
RAW="${H4H_RAW:-data/signal-hunters-round2}"; D="${H4H_DATA:-data}"; export H4H_RAW="$RAW" H4H_DATA="$D"
mkdir -p results logs
# fake-PKS raw renderings: pipeline.sh deg2/deg4 seeds and knobs, test-matched panel gain jitter (GAINSD=1.9) as for lvl
[ -f "$D/rs_d2_cadences.npy" ] || OUT=rs_d2 SEED=123 GAINSD=1.9 $PY src/degrade_lin.py
[ -f "$D/rs_d4_cadences.npy" ] || OUT=rs_d4 SEED=321 GRATE=0.0085 GLO=22 GHI=60 GAINSD=1.9 $PY src/degrade_lin.py
D2=rs_d2; D4=rs_d4
if [ -n "$QMAP" ]; then $PY src/qmap_raw.py rs_d2 rs_d2q; $PY src/qmap_raw.py rs_d4 rs_d4q; D2=rs_d2q; D4=rs_d4q; fi
$PY src/raw_stats.py "$D/rstats.npz" clean="$RAW/train_cadences.npy" test="$RAW/test_cadences.npy" \
    deg2="$D/${D2}_cadences.npy" deg4="$D/${D4}_cadences.npy"
$PY src/shift_check.py "$D/rstats.npz" rstats | tee logs/shift_rstats.txt
$PY src/resid_scan.py "$D/rstats.npz" rstats | tee logs/resid_rstats.txt
$PY src/subset_npz.py "$D/rstats.npz" "$D/rstats_top.npz" --pairauc results/resid_rstats_pairauc.csv --k 40 \
    --shift results/shift_rstats.csv --maxshift 0.75
[ -n "$NOBASE" ] || $PY src/bundle_eval.py | tee logs/eval_baseline.txt
$PY src/bundle_eval.py "$D/rstats_top.npz" | tee logs/eval_rstats_top.txt
$PY src/bundle_eval.py "$D/rstats.npz" | tee logs/eval_rstats_all.txt
