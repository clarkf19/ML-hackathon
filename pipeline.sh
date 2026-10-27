#!/bin/bash
# Reproduces submission4.csv (and the intermediate files) from the raw competition data.
# Data: unzip signal-hunters-round2.zip into data/  -> data/signal-hunters-round2/*.npy|csv
# All randomness is seeded (folds seed 42, degrade seed 123, LightGBM default seeds).
set -e
PY="${PY:-python} -X utf8"
cd "$(dirname "$0")"
$PY src/prep.py train test                      # telescope-agnostic normalisation -> data/z_*.npy
$PY src/degrade.py                              # fake-PKS copy of train (noise, glitches, dropped rows)
$PY src/prep.py train_deg
$PY src/qmap.py train_deg train_deg2            # match test pixel marginal
for s in train test train_deg2; do
  $PY src/phys.py $s; $PY src/tex.py $s; $PY src/track.py $s; $PY src/rep.py $s; $PY src/rep2.py $s
done
$PY src/gbdt_aug.py A_deg2 deg2                 # 5-fold LGBM on clean+fake-PKS -> data/pte_A_deg2.npy
$PY -c "import sys; sys.path.insert(0,'src'); from common import *; write_sub(np.load(D+'/pte_A_deg2.npy').argmax(1),'submission4.csv')"
