#!/bin/bash
# Reproduces submissions/cand_A_d2468_trk2.csv from the raw competition data (CV 0.9195 clean / 0.8873 fake-PKS).
# Data: unzip signal-hunters-round2.zip into data/  -> data/signal-hunters-round2/*.npy|csv
# All randomness is seeded (folds seed 42, simulator seeds below, LightGBM default seeds).
set -e
PY="${PY:-python} -X utf8"
cd "$(dirname "$0")"
$PY src/prep.py train test                                   # telescope-agnostic normalisation -> data/z_*.npy
# four simulated-PKS copies of train (noise level/shape, glitch rows, dropped rows) + quantile map to the test marginal
OUT=train_deg  SEED=123                                    $PY src/degrade.py && $PY src/prep.py train_deg  && $PY src/qmap.py train_deg  train_deg2
OUT=train_deg3 SEED=321 GRATE=0.0085 GLO=22 GHI=60          $PY src/degrade.py && $PY src/prep.py train_deg3 && $PY src/qmap.py train_deg3 train_deg4
OUT=train_deg5 SEED=555 GRATE=0.007  GLO=18 GHI=60 LNSTD=0.43 $PY src/degrade.py && $PY src/prep.py train_deg5 && $PY src/qmap.py train_deg5 train_deg6
OUT=train_deg7 SEED=777 GRATE=0.0075 GLO=20 GHI=55 LNSTD=0.47 $PY src/degrade.py && $PY src/prep.py train_deg7 && $PY src/qmap.py train_deg7 train_deg8
for s in train test train_deg2 train_deg4 train_deg6 train_deg8; do
  $PY src/phys.py $s; $PY src/tex.py $s; $PY src/track.py $s; $PY src/rep.py $s; $PY src/rep2.py $s; $PY src/trk2.py $s
done
SETS=phys,tex,trk,rep,rep2,trk2 $PY src/gbdt_aug.py A_d2468_trk2 deg2,deg4,deg6,deg8   # -> data/pte_A_d2468_trk2.npy
$PY -c "import sys; sys.path.insert(0,'src'); from common import *; print(write_sub(np.load(D+'/pte_A_d2468_trk2.npy').argmax(1),'cand_A_d2468_trk2.csv'))"
