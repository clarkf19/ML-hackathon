#!/bin/bash
# Raw-byte scintillation features for train / test / two fake-PKS renderings (same seeds + params as the lvl copies in
# pipeline.sh, GAINSD=1.9), then score with the harness.  Needs data/signal-hunters-round2/{train,test}_cadences.npy.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python}"
$PY src/scint_raw.py degrade lv_d2 123                         # = pipeline.sh  OUT=lv_d2 SEED=123 GAINSD=1.9
$PY src/scint_raw.py degrade lv_d4 321 0.0085 22 60            # = pipeline.sh  OUT=lv_d4 SEED=321 GRATE=0.0085 GLO=22 GHI=60 GAINSD=1.9
$PY src/scint_raw.py feats train train
$PY src/scint_raw.py feats test test
$PY src/scint_raw.py feats lv_d2 train_deg2
$PY src/scint_raw.py feats lv_d4 train_deg4
$PY src/scint_raw.py bundle
$PY src/scint_auc.py results/feats_scint_raw.npz
$PY src/bundle_eval.py                                         # baseline, same run
$PY src/bundle_eval.py results/feats_scint_raw.npz
