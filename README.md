# Hack4Hack 2026 Round 2 — "Signal Hunters" (team The Loss Minimizers)

9-class classification of SETI ON/OFF cadences (6 panels × 16 time steps × 128 channels), scored by macro-F1.
Train = "GBT" telescope, test = "PKS" telescope (noisier, more glitches, larger panel gain changes, different label mix).

**Final upload: `submissions/submission17.csv`** (public 0.9369; `submission18.csv` is the same blend with the
logistic-regression stacker at 0.33). Full results log with every experiment and number: `SUMMARY.md`.
Step-by-step rebuild: `REPRODUCE.md`.

## Solution in one page
1. **Telescope-agnostic preprocessing** (`src/prep.py`): bandpass flattening, robust noise scaling; a level-preserving variant
   (`KEEPGAIN=1`, arrays `zr_*`) keeps per-panel brightness (scintillation / pulsar ON brightening).
2. **Fake-PKS simulator** (`src/degrade.py`, `src/qmap.py`): re-renders training cadences with PKS-like noise, glitch and
   dropped rows, panel gain jitter and the test pixel marginal. Used as validation ("fake-PKS" CV, which tracks the public
   leaderboard within about 0.005) and as training augmentation (4 copies).
3. **Physics feature models** (LightGBM, `src/gbdt_aug.py`): drift / ON-track searches in global time, refined ETI track
   tests, dedispersion, repeat-k persistence, second-source masking, panel levels from raw bytes, robust raw statistics,
   bidirectional sweeps (`src/phys.py`, `tex.py`, `track.py`, `rep.py`, `rep2.py`, `trk2.py`, `trk3.py`, `lvl.py`, `sweep.py`,
   `rawsim.py` + `rawstats.py` + `build_rs.py`). Trained on clean + 4 fake-PKS copies + confident real-test pseudo-labels
   from our own earlier submissions (self-training rounds; the files used are in `repro/`).
4. **CNNs on a global-time canvas** (`src/cnn.py`; Colab notebook `colab/signal_hunters_gpu.ipynb` built by
   `colab/build_colab.py`): ImageNet-pretrained EfficientNet-B0/B2/B3, ResNet-34/18 with ETI/trap injection, 50 % fake-PKS
   mixing and real-test pseudo-labels; 5 folds each.
5. **Stacking** (`src/final_build.py` → `src/morning.py` → `src/final_blend.py`, plus `src/meta_lr.py`): every feature model
   and every complete CNN is a separate input of a LightGBM meta-model trained on clean + fake-PKS out-of-fold
   probabilities, averaged with a 50/50 blend and with a logistic-regression stacker (weight 0.2).
   Final 5-fold fake-PKS macro-F1 0.9384.

## Re-running only the final step (no GPU needed)
The out-of-fold and test probabilities of all 6 feature models and 11 CNNs used by the final blend are committed in
`data/` (oofd_/oofc_/pte_*, oofpart_/oofdpart_/ptepart_*), together with the fixed folds (`data/folds.npy`).
Put the competition files in `data/signal-hunters-round2/`, then:
```
EXCL=E4g,P3g,E3p6g GSEP=1 GBDT=A_rsswp_PL5,A_rsswp_PL5t6,A_rs_PL5t6,A_trk3_PL5t6,A_rsswp_PL4,A_rs_PL3 META_NEST=600 META_LR=0.02 \
  python -X utf8 src/final_build.py submission17
```
This reproduces `submissions/submission17.csv` exactly (checked: 0 of 10,000 labels differ, meta probabilities bit-identical).

**Paths:** the scripts use the absolute project root `C:\Users\clark\hack4hack-r2` (Windows, Python 3.11, LightGBM, PyTorch).
Clone the repo there, or replace that string in `src/*.py`, `run_*.sh` and `pipeline.sh`.
