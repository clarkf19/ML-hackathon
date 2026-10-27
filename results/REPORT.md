# Residual-information sweep (cloud brief 1) — report

**Question.** The last big win (lvl, +0.010 / +0.010) came from information the preprocessing erased (per-panel level).
Is there more information in the raw uint8 cadences that the current 427 bundle features / A_lvl model do not have?

**Answer: yes, a moderate amount.** 226 cheap raw-domain statistics, filtered for domain shift (`feats/rs_pruned.txt`), add
**+0.0055 clean / +0.0043 fake-PKS macro-F1** in the standard harness (0.9331 / 0.8974 vs 0.9275 / 0.8931; paired bootstrap
90% CI on fake-PKS +0.0022…+0.0065).
The gain is concentrated in **radar_chirp (+0.016 fPKS)**, noise, frb and pulsar. The source is mostly
*adjacent-channel pixel correlation* and *row-wise spectral kurtosis* (ON−OFF contrasts), i.e. broadband / sweep structure
inside one time row that the z-normalised features smooth away. A second, smaller source is *time structure of the broadband level*, which
affects scintillation. Scintillation itself gains on clean (+0.014…+0.020) but barely on fake-PKS (+0.002…+0.003), so the
per-panel level win is not repeated for that class.

## Numbers (all `python src/bundle_eval.py`, same 5 folds, LightGBM on clean + deg2 + deg4)
| Run | extra features | clean-val | fakePKS-val | Δ vs baseline (clean / fPKS) | file |
|---|---|---|---|---|---|
| baseline (this session) | – | 0.9275 | 0.8931 | – | results/eval_baseline.txt (brief: 0.9279 / 0.8931) |
| rs_top60 (top-60 by clean stacking gain) | 60 | 0.9332 | 0.8959 | +0.0057 / +0.0028 | results/eval_rs_top60.txt |
| rs_robust (all stats with fake-vs-test shift < 0.1) | 245 | 0.9326 | 0.8974 | +0.0051 / +0.0043 | results/eval_rs_robust.txt |
| rs_robust, degrade-only fake copies (no raw qmap) | 245 | 0.9325 | 0.8972 | +0.0050 / +0.0041 | results/eval_rs_robust_noq.txt |
| **rs_pruned (rs_robust minus dead groups G7 glitch / G9 edge / G10 quant)** | **226** | **0.9331** | **0.8974** | **+0.0056 / +0.0043** | results/eval_rs_pruned.txt |
| rs_ftop120 (top-120 by fake-PKS-aware stacking gain) | 120 | 0.9332 | 0.8950 | +0.0057 / +0.0019 | results/eval_rs_ftop120.txt |

Paired bootstrap (resampling cadences, `src/compare_probs.py`):
- **rs_pruned vs baseline: clean +0.0055 [90% CI +0.0037, +0.0075], fakePKS +0.0043 [+0.0022, +0.0065], P(gain>0) = 1.00.**
  Per-class fPKS: radar +0.016, noise +0.006, frb +0.004, eti +0.003, pulsar +0.003, scint +0.003, rfi_pers +0.003, rest ±0.002.
- rs_robust vs baseline: clean +0.0050 [+0.0031, +0.0070], fakePKS +0.0042 [+0.0020, +0.0063]; rs_pruned vs rs_robust
  +0.0005 / +0.0001 (CIs straddle 0). The 19 glitch / edge / quantisation statistics carry nothing, so they are dropped.
- rs_top60 vs baseline: clean +0.0056 [+0.0038, +0.0076], fakePKS +0.0028 [+0.0007, +0.0049]. Its scint goes
  +0.020 clean but −0.007 fPKS: the clean-selected subset overfits to GBT-only scint cues. The broad sets do not.
- rs_ftop120 vs baseline: clean +0.0056 [+0.0037, +0.0077], fakePKS +0.0018 [−0.0003, +0.0039]. Even selection by the
  fake-PKS-aware stacker loses about half of the fake-PKS gain. **Selecting a subset by importance overfits; the broad,
  shift-filtered set is better.**
- rs_robust vs rs_robust_noq (raw qmap vs degrade-only fake copies): +0.0000 / +0.0002, CIs straddle 0. The harness cannot
  tell the two simulator variants apart, but the raw qmap makes the statistics match the real test distribution much
  better (below), so the qmap version is preferred.

**Blend with the current 4-copy model** (A_lvl probabilities in bundle/meta.npz; argmax of the averaged probabilities;
weight 0.5 fixed in advance):
| | clean-val | fakePKS-val |
|---|---|---|
| A_lvl (submission2 model, 4 fake copies) | 0.9292 | 0.8972 |
| 0.5 A_lvl + 0.5 rs_robust harness model | 0.9329 | 0.8991 |
| 0.5 A_lvl + 0.5 rs_robust_noq harness model | 0.9330 | 0.8994 |
| **0.5 A_lvl + 0.5 rs_pruned harness model** (`submissions/cand_rs_blend.csv`) | **0.9334** | **0.8992** |

## What was computed (src/rawstats.py, 281 statistics per cadence)
Log domain D = bytes − OFF-panel per-channel median (bandpass flattened, **panel level kept**). Linear domain R = power /
OFF per-channel median power. Per-panel values are summarised as `_on`, `_off`, `_d` (on − off) and `_mm` (min ON − max OFF).
| group | n | content |
|---|---|---|
| G1 px | 53 | per-panel mean/std/skew/kurt, quantiles q01…max, robust sd of D |
| G2 lin | 20 | log mean, std, skew, kurt, q99 of linear power ratio |
| G3 row | 40 | per-row level profiles (median of D, log mean of R): std, lag-1 autocorrelation, range, excursion, slope |
| G4 ch | 40 | per-channel level profiles: std, lag-1/lag-4 autocorrelation, kurtosis, max |
| G5 dch | 14 | ON − OFF channel profile: std, autocorrelation, max/min, skew, fraction > threshold |
| G6 blk | 24 | time × sub-band block means at 2×32, 4×16, 8×8: block std, ON−OFF block std/max/min, max(min ON − max OFF) |
| G7 glitch | 18 | dropped rows ON/OFF/edge, broadband glitch / dip rows by panel (row-median z), partial-band glitch rows, clip fractions, max byte |
| G8 corr | 21 | inter-panel correlation of channel profiles (all / 9-ch high-pass / 16-ch low-pass; ON-ON, OFF-OFF, ON-OFF, adjacent) and of the 2D residual maps |
| G9 edge | 5 | edge-channel level vs centre, ON−OFF edge level, channel 64 ON−OFF |
| G10 quant | 8 | byte-value quantisation: distinct values, histogram entropy, parity, mod-24 pattern |
| G11 sk | 19 | spectral kurtosis var/mean² of linear power, per channel over time and per row over channels |
| G12 pac | 8 | pixel-level lag-1 autocorrelation in time and in frequency (channel profile removed) |
| G13 tail | 11 | fraction of pixels > +24 / +48 bytes, < −36 bytes |

Fake-PKS renderings of the raw train cadences come from `src/rawsim.py`: a port of degrade.py with the same RNG call
order, GAINSD = 1.9, pipeline seeds 123 (deg2) and 321/GRATE 0.0085/GLO 22/GHI 60 (deg4). It is **bit-identical** to the pipeline's
lv_d2 copy: the lvl features recomputed from it match bundle/feats_deg2 with max abs diff 0.0. Plus a raw-domain port of
qmap.py: the residual after removing the smooth bandpass and panel level (bytes) is quantile-mapped to the test marginal,
and the levels are kept (`raw_d2q`, `raw_d4q`).

## How "information the model lacks" was measured
1. **Conditional AUC per confused pair** (`src/residual_scan.py`, results/scan_auc_top.txt): within each predicted-class
   stratum, AUC of true a vs true b. Top raw statistics reach about the same values as the best bundle features (lvl for scint
   pairs), so this test cannot tell "new" from "already used" information. Kept only for reference.
2. **Probability-matched AUC** (`src/residual_auc2.py`, results/scan_auc2_top.txt): cadences of a confused pair binned by
   the model's own pairwise log-odds; AUC inside the bins (0.5 = nothing beyond the model's belief).
   Strongest residuals: **radar|noise: skr_med_on 0.873, pac_f1_on 0.83** (best bundle feature 0.679; skr_med_on itself is
   noise-level dependent and was later excluded by the shift filter, so the robust carriers are pac_f1_* and the ON−OFF
   kurtosis contrasts skr_med_d / skr_max_mm); scint|radar: OFF-panel tail level px_q95_off 0.33, i.e. |AUC−0.5| 0.168 vs
   bundle best 0.121 (also excluded as shifted); scint|rfi_pers: rowP_slope_on 0.654 (bundle best 0.621).
   Pairs where raw statistics add nothing beyond the bundle: rfi_int|rfi_pers (raw max |AUC−0.5| 0.090 vs bundle 0.126),
   eti|rfi_int (0.113 vs 0.117), sat|radar (0.191 vs 0.248), radar|rfi_pers (0.342 vs 0.319, marginal).
3. **Residual stacking** (`src/residual_scan.py`, results/scan_stack.txt): LightGBM on [log oofc] vs [log oofc + group].
   Clean: probs-only 0.9280 → +all raw 0.9354, logloss 0.2128 → 0.2005. Best groups: G4 ch +0.0031, G11 sk +0.0024,
   G1 px +0.0022, G3 row +0.0021, G12 pac +0.0019. Nothing from G8 corr (+0.0003) or G10 quant (+0.0002).
4. **Robustness-aware stacking** (`src/residual_scan_fpks.py`, results/scan_fpksq.txt): the same, trained on clean + fake-PKS
   rows ([oofd, rs_d2q]) and scored on fake-PKS. All robust: +0.0071 clean / +0.0055 fPKS. Groups on fPKS:
   G5 dch +0.0028, G3 row +0.0024, G6 blk +0.0019, G8 corr +0.0018, G12 pac +0.0017, G1 px +0.0016; G4 ch only +0.0009
   (mostly a GBT-only gain); G7 glitch +0.0003, G9 edge 0.0000, G10 quant +0.0008 (dead).
5. **Domain shift** (`src/shift_check.py`, results/shift_auc.csv): per statistic, |AUC−0.5| of test vs train rendering on
   noise cadences (train true noise vs test predicted noise p>0.9). Mean shift: clean 0.177, fake-PKS degrade-only 0.081,
   **fake-PKS + raw qmap 0.052**. Statistics that stay shifted even after qmap are noise-level / bandpass /
   clipping descriptors: edge_lo/hi, byte entropy / distinct values / parity, vmax, cad_rsd/px_rsd/px_std, v1 fractions,
   absolute spectral-kurtosis levels (skr/skc med, skr max), OFF-panel quantiles and tails, chL_std.
   These were **excluded** (shift ≥ 0.1 → 36 removed, 245 kept; minus 19 dead G7/G9/G10 statistics → 226 final).
   Within-cadence ON − OFF contrasts are the safest.

## Findings worth knowing
- **Pulsar fingerprint.** `cc2d_onon` (correlation of the 2D time × frequency residual maps between ON panels) is 0.25 for
  pulsars and 0.000 ± 0.002 for every other class: the generator repeats the same pulse pattern in every ON panel. It is robust
  (fake-PKS 0.16, test-predicted pulsars 0.19) but pulsar is already at 0.99 F1, so the effect is small.
- **Radar.** Steep sawtooth sweeps raise the adjacent-channel pixel correlation (pac_f1: radar 0.12 vs noise −0.007, in
  ON *and* OFF panels) and the per-row spectral kurtosis. This is the most robust new information: radar F1 +0.016 on fake-PKS.
- **Scintillation.** Raw ON-level statistics (px_q50_d, dchL_fpos, lin_logmean_on) are strong but largely duplicate lvl.
  Row-level time slope / autocorrelation in ON panels adds some clean signal, but it does not survive the noisier
  PKS rendering. On fake-PKS, scint stays at about 0.69–0.70 F1. On test, predicted scint cadences are *brighter*
  (px_q50_d median 5.2 vs 3.8 train), consistent with the earlier finding that test scint is rarer and the visible ones are strong.
- **No artefacts** in byte quantisation, dropped-row positions or glitch-row placement: they carry no label signal.
- rfi_int vs rfi_pers (the 2nd largest confusion) is **not** helped by any raw statistic. It needs source-level
  (per-line, per-panel presence) reasoning, not global statistics.

## Verdict
**WIN, moderate: +0.0055 clean / +0.0043 fake-PKS (harness), +0.0042 / +0.0020 as a 50/50 blend with A_lvl.**
Recommendation for the final pipeline:
1. Add the 226 statistics in `feats/rs_pruned.txt` as a feature set. Compute them for train, test and all 4 fake copies:
   `COPIES=d2,d4,d6,d8 QMAP=1 python src/rawsim.py && python src/rawstats.py train test d2q d4q d6q d8q` (~30 min CPU on
   4 cores). Then select the columns with `feats/rs_pruned.txt` (see `src/make_rs_npz.py`) and retrain the 4-copy LightGBM
   (gbdt_aug.py with SETS += rs; it reads CSVs on Windows paths, so it needs a small loader change). Expected gain over A_lvl
   about +0.004 fPKS. Not done here: the bundle has no deg6/deg8 copies of the 427 base features.
2. Without a 4-copy retrain, upload `submissions/cand_rs_blend.csv`: argmax of 0.5 × A_lvl pte + 0.5 × harness rs_pruned pte,
   fPKS-val 0.8992 vs 0.8972, clean 0.9334 vs 0.9292. It agrees with submission2 on 98.8% of test rows; format checked
   against sample_submission (10,000 rows, same ID order, valid labels).
3. Do not add a subset selected by importance (top-60 / top-120 lose 35–55% of the fake-PKS gain; top-60 regresses scint),
   nor the shifted noise-level / bandpass statistics.

## Files
- `src/rawsim.py` fake-PKS raw renderer (degrade port + raw qmap); `src/rawstats.py` 281 raw statistics;
  `src/residual_scan.py`, `src/residual_auc2.py`, `src/residual_scan_fpks.py` residual-information tests;
  `src/shift_check.py` domain-shift check; `src/make_rs_npz.py` packs a feature list for bundle_eval; `src/compare_probs.py`
  paired bootstrap comparison.
- `feats/rs_pruned.txt` (226, recommended), `feats/rs_robust.txt`, `feats/rs_top60.txt`, `feats/rs_ftop120.txt` feature lists;
  `results/rs_pruned_f16.npz` = X_clean / X_test / X_deg2 / X_deg4 (float16, max rel. error 5e-4; raw-qmap fake copies) for
  `python src/bundle_eval.py results/rs_pruned_f16.npz`. (The float32 version gave 0.9331 / 0.8974; the float16 file
  was not re-run through the harness.)
- `submissions/cand_rs_blend.csv` candidate upload; `src/make_blend_sub.py` builds it from `results/probs_rs_pruned.npz`
  (a harness output, not committed; `python src/bundle_eval.py results/rs_pruned_f16.npz` writes the equivalent
  `results/probs_rs_pruned_f16.npz`).
- `results/eval_*.txt` harness logs; `results/scan_*`, `results/shift_auc.csv` analysis outputs.
