# Reproducing the final submission (Hack4Hack 2026 R2 "Signal Hunters")

Everything is trained only on the provided data. Real-test pseudo-labels are produced by our own earlier models
(self-training); the exact pseudo-label / teacher files used are shipped in `repro/` so every model can be retrained
deterministically without replaying the whole chain. Python 3.11, LightGBM, PyTorch (+ torchvision on the GPU).

**Shortcut:** the final step alone (section 4) can be re-run from the committed model predictions in `data/`, with no
retraining; see README.md. **Paths:** scripts use the absolute root `C:\Users\clark\hack4hack-r2`; clone there or replace
that string in `src/*.py`, `run_*.sh` and `pipeline.sh`. The final upload is `submissions/submission17.csv`.

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
`E2` (B2), `E7` (B3), `E8` (B2, 3x upsampling), `E1p6` (B0, round-6 pseudo-labels `pl_test6.npy`), `R18p6` (ResNet-18, same),
`E4` (B1), `P3` (ResNet-34). Copy `out/oofpart_*`, `out/oofdpart_*`, `out/ptepart_*` into `data/`.
Folds can also be trained on a local CUDA GPU: `python src/make_cnn_gpu.py` writes `src/cnn_gpu.py` (same patch as the
notebook), then e.g. `BS=32 <same env as the notebook config> python src/cnn_gpu.py 4 20 E7g` (E7 fold 4 and E1p6 folds 3-4
were trained this way on an 8 GB RTX 5060 with batch 32).

## 4. Final blend
```
EXCL=E4g,P3g GSEP=1 GBDT=A_rsswp_PL5,A_rsswp_PL5t6,A_rs_PL5t6,A_trk3_PL5t6,A_rsswp_PL4,A_rs_PL3 META_NEST=600 META_LR=0.02 \
  python -X utf8 src/final_build.py <name>
```
`src/final_build.py` runs `src/morning.py` (below) and then averages in a logistic-regression stacker on the same inputs
(`src/meta_lr.py`, weight 0.2).
`src/morning.py`: every CNN with all 5 folds becomes its own input of a stacked LightGBM meta-model (`src/final_blend.py`,
trained on clean + fake-PKS out-of-fold probabilities); CNNs with only some folds are mixed on top with one weight chosen on
the folds where they exist (`src/mix_partial.py`). Output: `submissions/<name>.csv`.
