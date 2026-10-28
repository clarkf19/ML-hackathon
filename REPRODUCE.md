# Reproducing the final submission (Hack4Hack 2026 R2 "Signal Hunters")

Everything is trained only on the provided data. Real-test pseudo-labels are produced by our own earlier models
(self-training); the exact pseudo-label / teacher files used are shipped in `repro/` so every model can be retrained
deterministically without replaying the whole chain. Python 3.11, LightGBM, PyTorch (+ torchvision on the GPU).

## 1. Data and features (CPU, ~3 h)
Unzip the competition data into `data/signal-hunters-round2/`, then:
```
bash pipeline.sh                                   # normalisation, 4 fake-PKS copies, phys/tex/trk/rep/rep2/trk2/lvl features
python src/trk3.py  train test train_deg2 train_deg4 train_deg6 train_deg8    # ETI sub-channel track features
python src/sweep.py train test train_deg2 train_deg4 train_deg6 train_deg8    # bidirectional sweep features (swp)
QMAP=1 COPIES=d2,d4,d6,d8 python src/rawsim.py && python src/rawstats.py train test d2q d4q d6q d8q && python src/build_rs.py   # rs
bash run_inj.sh && bash run_zr.sh                  # injected ETI/trap cadences + level-preserving arrays for the CNNs
cp repro/*.npy data/                               # pseudo-labels (pl_test*.npy) and teacher probabilities (pte_submission*.npy)
```

## 2. Feature models (LightGBM, `src/gbdt_aug.py`, 5 fixed folds, clean + 4 fake-PKS copies)
```
B=phys,tex,trk,rep,rep2,trk2,lvl
PL=submission6  PLTH=0.8 SETS=$B,rs                        python src/gbdt_aug.py A_rs_PL3      deg2,deg4,deg6,deg8
PL=submission7  PLTH=0.8 SETS=$B,rs,swp                    python src/gbdt_aug.py A_rsswp_PL4   deg2,deg4,deg6,deg8
PL=submission10 PLTH=0.8 SETS=$B,rs,swp                    python src/gbdt_aug.py A_rsswp_PL5   deg2,deg4,deg6,deg8
PL=submission10 PLTH=0.6 SETS=$B,rs,swp GSEED=7            python src/gbdt_aug.py A_rsswp_PL5t6 deg2,deg4,deg6,deg8
PL=submission10 PLTH=0.6 SETS=$B,rs     GSEED=44           python src/gbdt_aug.py A_rs_PL5t6    deg2,deg4,deg6,deg8
PL=submission10 PLTH=0.6 SETS=$B,trk3   GSEED=33           python src/gbdt_aug.py A_trk3_PL5t6  deg2,deg4,deg6,deg8
```

## 3. CNNs (Colab T4, `colab/signal_hunters_gpu.ipynb`, built by `colab/build_colab.py`)
Upload `colab/h4h_bundle.zip` + `repro/pl_test*.npy` to `MyDrive/h4h_r2/`, set `CFGS` / `FOLDS` in the last cell, run all.
Configs used: `B3pl2`, `B3w`, `B3plw` (from scratch) and ImageNet-pretrained `P1` (ResNet-34), `E1`, `E3` (EfficientNet-B0),
`E2` (B2), `E4` (B1), `P3` (ResNet-34). Copy `out/oofpart_*`, `out/oofdpart_*`, `out/ptepart_*` into `data/`.

## 4. Final blend
```
GSEP=1 GBDT=A_rsswp_PL5,A_rsswp_PL5t6,A_rs_PL5t6,A_trk3_PL5t6,A_rsswp_PL4,A_rs_PL3 python -X utf8 src/morning.py <name>
```
`src/morning.py`: every CNN with all 5 folds becomes its own input of a stacked LightGBM meta-model (`src/final_blend.py`,
trained on clean + fake-PKS out-of-fold probabilities); CNNs with only some folds are mixed on top with one weight chosen on
the folds where they exist (`src/mix_partial.py`). Output: `submissions/<name>.csv`.
