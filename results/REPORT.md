# Generator brief (cloud_prompts/2_generator.md) — interim report

Status: **interim, stopped on request before the generator was built.** Only steps 1 and the first part of step 2 are done.

## Done
- Raw data from the `data` branch (`get_data.sh`). torch can't be installed from download.pytorch.org but can from PyPI
  (2.14.1), so the existing feature scripts run unchanged.
- `src/featlib.py`: portable (Linux/Windows) pipeline, raw uint8 cadences + slew gaps -> the 427 bundle columns in bundle
  order. It reuses `phys/tex/track/rep/rep2/trk2.feats` unchanged and ports `lvl.py` and `qmap.py`.
  **Verified** against `bundle/feats_clean.npz` on the first 400 train cadences: max abs diff 6e-5 (one column,
  `phys_on_fit_resid`, differs at least-squares rounding level). Generated cadences can therefore be put in the bundle feature space.

## Baseline
`python src/bundle_eval.py` was started but **did not finish**: the process was gone and its log empty, probably killed
when the session turn was interrupted. It was not rerun. Reference value from the brief (not reproduced here):
0.9279 clean-val / 0.8931 fakePKS-val.

## Instrument measurements (preliminary)
| Quantity | GBT (train, 1,500 noise-class cadences) | PKS (test, first 3,000 cadences, all classes) |
|---|---|---|
| robust noise σ (bytes) | 11.86 | 16.31 |
| ln-power noise std | 0.364 | 0.504 |
| -> gamma shape k (χ² dof) | 8.0 (16.1) | 4.4 (8.8) |
| skew, measured / log-gamma theory at that k | −0.363 / −0.363 | −0.413 / −0.502 |
| excess kurtosis | 0.31 | 0.62 |
| dropped rows (all-zero rows) | 0.15% | 0.38% |
| glitch rows (row-median residual > 6 bytes) | 0.63% | 1.61% |
| panel gain spread (median per-cadence std of panel offsets, bytes) | 1.21 | 2.23 |

Method: residual = byte value − per-cadence channel median (over all panels and times) − per-panel median offset.
Dropped rows are excluded, as are glitch rows (row-median residual > 6 bytes) and pixels with |residual| ≥ 120.
ln-std = sqrt(var − 1/12) / (24 / ln 2). k solves trigamma(k) = ln-var. Theoretical skew = ψ''(k) / ψ'(k)^1.5.

Reading:
- **GBT noise is consistent with multiplicative gamma (log-χ²) noise.** The measured skew equals the log-gamma value
  at the fitted k.
- **PKS does not fit a single log-gamma.** Its skew is −0.41 against −0.50 expected at the fitted k, and its kurtosis
  is higher. The cause is unresolved. Either signals in the unlabelled all-class test sample inflate the variance, or the
  noise shape really differs. Separating the two is the next step.
- The ln-std values sit above SUMMARY.md's 0.343 / 0.45. This residual keeps per-panel bandpass and time structure that
  the earlier estimate removed. **Treat k as a lower bound** until a per-panel bandpass fit is done.
- PKS has 2.5× the dropped-row rate, 2.6× the glitch-row rate and 1.8× the panel gain spread of GBT. This agrees with
  SUMMARY.md (0.37 vs 0.15 dropped rows per cadence; 1.16 vs 0.64 glitch rows per cadence).

## Not done (remaining brief)
- Bandpass shape (ripple / roll-off / tilt), glitch amplitude and partial-band widths, and a clean PKS noise shape.
- Per-class signal rendering estimates.
- `src/generator.py`, its validation (LightGBM trained on generated GBT and scored on real train; adversarial AUC), and the
  PKS-generated-data experiment in `bundle_eval.py`.
