# Cloud session 4: scintillating_star

**Goal:** raise scintillating_star F1 on fake-PKS val (harness baseline 0.691) without hurting other classes.
**Outcome:** no bundle-only change moves it. The numbers say why: the remaining scint loss is **SNR-limited by panel gain
jitter**. It is not a threshold problem, and contamination is only a secondary cause. The brightening is a pure
log-domain shift, which can't be told apart from gain jitter, so the only gain-free evidence left is the scintle
texture. A raw-byte detector aimed at exactly that is written and ready, but it could not be run here (no raw data).

## 0. Constraint: no raw data in this session
`gdown` failed: the cloud environment's network policy returns **403 for drive.google.com** (and for
download.pytorch.org, so there is no torch either). Following CLOUD.md, every number below comes from the 427 committed
bundle features (clean / test / deg2 / deg4) and the harness `src/bundle_eval.py` (same folds, same LightGBM, trained on
clean + deg2 + deg4). The raw-byte detector of task step 3 (`src/scint_raw.py`, section 4) is written with portable
paths and smoke-tested on synthetic bytes. It has **not** been run on competition data, and no number in this report
comes from it.

## 1. Baseline (harness, this session)
| | clean-val | fakePKS-val | scint F1 clean | scint F1 fPKS | scint recall / precision fPKS |
|---|---|---|---|---|---|
| `python src/bundle_eval.py` | **0.9279** | **0.8931** | 0.777 | **0.691** | 0.605 / 0.805 |

Where scint goes on fPKS val (1431 true scint): noise 310, rfi_persistent 116, satellite 60, radar 44, rfi_int 33.
Predicted-scint false positives: noise 84, rfi_pers 36, radar 28, satellite 21, rfi_int 20, frb 14.

## 2. Characterising the signal (bundle features; `src/scint_char.py`, `src/scint_tail.py`; logs in `results/logs/`)
| Question | Answer (numbers) |
|---|---|
| Size of ON brightening | ON-panel level − OFF mean: **+4.0 bytes** per ON panel (mean of all 3, clean and fPKS alike). It varies a lot between cadences: lv_on_off p10/p50/p90 = 0.7 / 3.8 / 7.9 (clean). Noise class: 0 ± 1.58 (clean), ± 2.21 (fPKS) |
| Frequency shape | **Flat**: ON−OFF level by 16-channel sub-band 4.07 4.09 4.04 4.02 4.00 4.02 3.98 3.99 (clean). Low-vs-high-band tilt AUC 0.52 (none). 93% of sub-bands are brightened (noise 50%) |
| Time shape | ON time-block (4-row) level spread is barely above OFF (median 1.54 vs 1.36); no systematic in-panel ramp |
| Same in all 3 ON panels? | Mostly shared. corr(ON_i, ON_j) across scint cadences = **0.71** (clean) / 0.54 (fPKS). All-3-ON > all-3-OFF in 69% of scint (fPKS 50%; noise 10% / 6.5%; null 5%). ON level sd 3.0 vs OFF sd 1.1 means about 1.2 bytes of panel-to-panel brightness variation beyond gain jitter, as expected from scintillation between panels |
| Mean vs variance modulation | **Mean dominates**: level AUC scint-vs-noise 0.905 clean / 0.855 fPKS. The best texture (block-variance excess) is at **8 rows × 16 ch** (AUC 0.846 / 0.783), then 8×32 (0.834 / 0.773); small (2×4) and very large (16×64) scales are weak (0.74 / 0.71). So scintles are ~8 rows × 16–32 channels |
| Linear vs log domain | **Log domain.** If the star added linear power S to log-χ² noise, the lower tail would rise more than the upper (simulated +4-byte star: q10/q50/q90 shift 6.4 / 4.1 / 2.8 bytes) and the gain-free fraction of pixels above 1.5σ would drop by 0.016 in ON panels. Observed `tex_pos_d` for scint: median +0.002 (IQR −0.002…+0.006), the same as noise (0.000). The brightening is a uniform byte shift, so per panel it is **degenerate with gain jitter**. The only gain-free evidence is scintle texture and ON-ON consistency |
| SNR budget | median brightening / level-noise sd = **2.41 clean, 1.72 fPKS**. **25% (clean) / 36% (fPKS)** of scint stars brighten less than the noise class's 90th percentile. Texture correlates with level within scint (0.61 / 0.49), so faint stars are faint in texture too: texture AUC for those faint stars vs noise is 0.714 / **0.675** |
| What the level noise is made of (`src/scint_gain.py`) | In level-neutral classes the OFF panels are i.i.d. (pairwise corr −0.50, exactly as expected relative to their mean), but the **3 ON panels share a common offset** (ON-ON corr 0.47 vs 0.25 expected for i.i.d. gains). Decomposition on predicted-noise cadences: panel gain sd **1.30** + common-ON offset sd **1.22** (clean); **2.18 + 1.36** (fPKS val); **2.29 + 1.34 (test, same selection)**. The common-ON term is symmetric (it can be negative), so it is not faint stars; it is a pointing-level nuisance that no level statistic can separate from a star. The fake-PKS simulator matches test here (ON-ON corr 0.335 vs 0.325). Test panel jitter is ~5% larger (GAINSD ≈ 2.0 would match slightly better than 1.9) |
| Recall vs brightening | fPKS recall by lv_on_off quartile: **0.20 / 0.46 / 0.79 / 0.97** (clean 0.24 / 0.54 / 0.89 / 1.00). Lost scint median lv_on_off 1.7 vs found 5.5 |
| Contamination | (with lvl features the effect is now modest) fPKS recall: clean sky 0.634, ON zero-drift line 0.534, glitch row 0.596, missing rows 0.590, OFF line 0.626, steep sweep 0.417 (n=24). Contaminated scint have the same brightening (lv median 3.7–3.9), so these cases are partly lost to source priority. Most of the lost mass is faint stars in clean sky |
| Test prevalence | Harness predicts 5.8% scint on test. With 8% prevalence and the fPKS recall/precision the expected predicted share is **6.0%**, and test has slightly *more* brightened cadences than fPKS train (lv_on_off > 3: 22.3% vs 20.3%). The earlier EM estimate "test scint ≈ 4%" is most likely a recall artefact, so do not bias against scint for the LB |

## 3. Experiments (harness-identical scoring; Δ vs baseline 0.9279 / 0.8931)
EXPERIMENTS_TABLE

## 4. Raw-data detector (ready to run, not run): `src/scint_raw.py`, `src/scint_raw_run.sh`
`bash src/scint_raw_run.sh` (needs `data/signal-hunters-round2/{train,test}_cadences.npy`; `H4H_DATA` overrides the
data dir) does the following:
1. Builds the two fake-PKS renderings with a **bit-exact port of `src/degrade.py`**: same RNG call order and the
   pipeline.sh seeds/params with GAINSD=1.9, so lv_d2 / lv_d4 bytes equal the ones behind bundle deg2 / deg4 and row i is
   still cadence i. `qmap.py` is not needed because the detector works on raw bytes like `lvl.py`.
2. Computes 131 `sraw_*` features for train / test / lv_d2 / lv_d4 (~10 min CPU estimated from the smoke test), writes
   `results/feats_scint_raw.npz` and runs single-feature AUCs, then the harness baseline and `bundle_eval.py` with the
   new features.

What it measures, chosen from section 2:
* **Conservative mask** of dropped rows/pixels, broadband glitch rows, persistent narrow channels, and isolated >4.5σ
  pixels dilated ±1 channel. Scintles are ~0.1σ per pixel and survive this; tex3 masked whole detected tracks and lost
  them.
* **Overlapping matched filter at the scintle scale** (8×16, 8×32, 4×16 box on masked, gain-removed z, every window
  offset rather than one block grid). Outputs mean-square excess and p95, ON−OFF / pooled t / pair count. This is the
  gain-free evidence that section 2 says is left.
* Gain-free block-variance and positive-tail excess at 8 scales in **log and linear** power.
* Level tests in log and linear power on the masked data (difference, pooled-variance t, Mann-Whitney ON>OFF count).
* Additive-source test: slope of the channel ON−OFF level on the inverse bandpass. It is expected to be weak (log-domain
  finding) and is kept as a cheap check.
* ON-ON spectral and temporal pattern correlation (is the scintle pattern shared across the 3 ON panels?).

The smoke test on synthetic bytes (`scratch`, not committed) checked only shapes, NaN/inf and runtime. It is not
evidence of usefulness.

## 5. Verdict and recommendation
VERDICT

## Files
* `src/scint_char.py`: characterisation numbers (section 2); `src/scint_tail.py`: linear-vs-log test;
  `src/scint_auc.py`: single-feature AUCs; `src/scint_summary.py`: the table in section 3.
* `src/scint_derived.py` → `results/feats_scint_derived.npz` (48 bundle-derived scint statistics, all 4 renderings).
* `src/scint_stack.py`: nested-stacked scint specialist (MODE all / lower / confusers, FEATS all / scint) →
  `results/feats_spec_*.npz` (plain-OOF score for the harness).
* `src/scint_bias.py`: nested scint-only / all-class log-prob biases.
* `src/scint_raw.py`, `src/scint_raw_run.sh`: raw-byte detector + degrade port (section 4).
* `results/logs/*.txt`: raw outputs of every run quoted here.
