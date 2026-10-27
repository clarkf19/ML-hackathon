# Hack4Hack 2026 — Round 2 "Signal Hunters" summary (live log)

**Task:** 9-class cadence classification (6 panels ON/OFF/ON/OFF/ON/OFF × 16 time × 128 freq, uint8 log-power), macro-F1.
**Train = GBT telescope, test = PKS** (1.3× noisier, different noise *shape*, 1.8× more glitch rows, 2.5× more dropped rows,
different bandpass / channel width / tsamp / freq range, different label mix). **Uploads: 10 per day (resets midnight), round ends 28 Oct 10:00 IST. submission1 public = 0.8779** (proxy said 0.887 → proxy ≈0.01 optimistic). Code goes in as a GitHub link (not an upload).

Because test is a different telescope, every model is scored twice with the **same 5 folds**:
**clean val** (GBT) and **fake-PKS val** — the val fold re-rendered as PKS-like data (my simulator, see P-rows). The fake-PKS
number should predict the leaderboard better; adversarial AUC fake-PKS-vs-test 0.78 (clean-train-vs-test 0.96).

## (a) Current best
| Model | clean CV | fake-PKS CV | File | Upload? |
|---|---|---|---|---|
| A0 starter-style baseline | 0.631 | – | submissions/baseline_A0.csv | no (fallback only) |
| C3 features, clean train | 0.9035 | – | submissions/old_C3.csv | superseded |
| C4 features, clean + fake-PKS v3 | 0.9145 | 0.874 | submissions/old_A_deg2.csv | ok |
| C4 features, clean + fake-PKS v3 + v4 | 0.9168 | 0.8814 | submissions/cand_A_deg2_deg4.csv | superseded |
| C4 features, clean + 4 fake-PKS copies | 0.9168 | 0.8866 | submissions/cand_A_d2468.csv | superseded |
| C4 + trk2 features, clean + 4 fake-PKS copies | 0.9195 | 0.8873 | submissions/submission1.csv | **uploaded: public 0.8779** |
| **+ panel-level features from RAW data (lvl)** | **0.9292** | **0.8972** | **submissions/submission2.csv** | **uploaded: public 0.8910** (pipeline.sh reproduces it) |
| (fold-0 only) best GBDT + CNN B2 blend | 0.9253 (fold0) | 0.8950 (fold0) | needs 5-fold CNN B2 (Colab) | upload #2 after Colab |

## (b) Every idea / experiment (all numbers = macro-F1, 5-fold unless "fold0")
| # | Idea | Potential | clean | fake-PKS | Verdict | Notes |
|---|---|---|---|---|---|---|
| A0 | Starter-style per-panel stats + LGBM | low | 0.631 | – | baseline | ETI 0.13 |
| C | Physics features: per-panel drift search, global ON-track search (real global time incl. slew gap), dedispersion, persistence | high | 0.861 | – | **+0.23** | the core idea |
| C+T | + multi-scale ON−OFF block-variance texture | med | 0.879 | – | +0.018 | scint 0.62→0.68 |
| C2 | + refined ETI track: min(ON)−max(OFF) objective on clipped z, per-panel offset scan (disconnected trap), OFF-near-track, same-drift OFF lines, 2nd track | high | 0.892 | – | +0.013 | **ETI 0.78→0.875** |
| C3 | + "repeat-k" (same line, same place, k panels) + 2D autocorrelation | med | 0.9035 | – | +0.012 | radar 0.86→0.92 |
| C4 | + second-source features (mask top-2 persistent lines, recompute) | med | 0.9068 | 0.842 | +0.003 | cadences with persistent + extra intermittent line |
| P1–P4 | Fake-PKS simulator: log-gamma noise to PKS σ, extra dropped rows, glitch rows at PKS rate, quantile-map to test pixel marginal | validation | – | – | **proxy works** | adv AUC vs test 0.995 (v1/v2) → **0.78** (v3/v4) |
| A_deg2 | train on clean + fake-PKS v3 | high | 0.9145 | 0.874 | **+0.005 / +0.032** | |
| A_deg2_deg4 | + 2nd fake-PKS copy (v4: stronger glitches) | med | **0.9168** | **0.8814** | +0.002 / +0.007 | more noise realisations help |
| A_d2468 | 4 fake-PKS copies (varied noise 0.43–0.47, glitch rate/amp) | med | 0.9168 | **0.8866** | +0.000 / +0.005 | scint fPKS 0.609→0.624 |
| trk2 | In-band-aware global tracks through all 6 panels (persistent line drifting out of band ≠ intermittent) | med | **0.9195** | **0.8873** (v4-val 0.8851) | **win +0.003 / +0.001 (+0.004 v4)** | rfi_int 0.923→0.929, fPKS 0.905→0.914 |
| freq | RFI at repeated absolute frequencies ("different local interference frequencies") | high if true | – | – | **rejected** | 8 of 4,773 strong lines share a 200 Hz bin; not label-predictive |
| snr | Are test signals really fainter? (signal-tail quantiles) | high if false | – | – | test tails ≈ fake-PKS (slightly stronger), far below clean | simulator SNR assumption confirmed |
| **lvl** | **Panel power LEVEL from raw bytes** (prep.py's per-panel median subtraction erased ON-panel brightening: scint +3.67 bytes ON−OFF, pulsar +5.33, noise 0.00; AUC scint-vs-noise 0.903, missed-scint 0.833). Fake copies re-rendered with test-matched gain jitter (σ 1.9; OFF-spread median 3.45 vs test 3.55) | **high** | **0.9292** | **0.8972** | **WIN +0.010 / +0.010** | scint 0.707→**0.776** clean, 0.62→**0.698** fPKS |
| tex3 | Source-aware scint texture (mask all detected lines + glitch rows) | med | – | – | weaker single-feature AUC (0.673 vs 0.74) | over-masks scint patches; not trained |
| B | CNN on global-time canvas (6 panels placed at real time → ETI = one straight line), SNR/zoom/flip aug | high | 0.875 fold0 | 0.826 fold0 | **complementary** | radar 0.966, rfi_pers 0.943 (> GBDT), ETI 0.52 |
| B2 | B + 2000 injected ETI/trap cadences + 50% fake-PKS mixing | high | **0.886 fold0** | **0.854 fold0** | +0.011 / +0.028 over B | ETI 0.52→0.64 |
| blend | GBDT(A_deg2_deg4) + CNN 50/50, fold0 | – | 0.9205–0.9233 | 0.887–0.890 | +0.008–0.010 | oracle "either right" acc 0.9575 vs GBDT 0.925 → lots of complementarity left |
| H | Hybrid: B2 + 392 physics features into CNN head | high | final fold0 0.8817 (B2 0.8863) | 0.8372 (B2 0.8537) | **rejected** | led early, ended below B2; blend with GBDT 0.918 / 0.879 is *worse* than GBDT alone on fake-PKS (redundant with the features) |
| blend2 | best GBDT (trk2) + CNN B2, 50/50, fold0 | – | **0.9253** (GBDT 0.9193) | **0.8950** (GBDT 0.8817) | **+0.006 / +0.013** | → Colab CFG='B2' for all 5 folds |
| G | Injection (ETI + 3 stated traps) for GBDT | high for ETI | 0.9121 | 0.8744 | **neutral** for GBDT | helps CNN (B2) |
| F | EM prior-shift correction | med | 0.868 (vs 0.874) | resampled mixes 0.8376 vs 0.8386 | **rejected** | balanced argmax already robust |
| K | Pseudo-labels (proxy: unlabeled = fake-PKS fold) | med | – | 0.8421→0.8447 across a domain gap; 0.8744→0.8740 without gap | small (+0.003) | |
| K2 | Best GBDT + real PKS test pseudo-labels (p>0.9) | med | 0.9206 | 0.8876 | neutral on proxies (as expected) | real gain only visible on LB; data/pte_A_d2468_trk2_PL.npy kept as option |
| S | Scintillation: ON−OFF referenced spectra, cross-panel corr, block-median texture (tex2) | med | 0.9167 (tex2) | 0.8803 | **neutral** | single-feature AUC on missed scint 0.667→0.774 but GBDT already had it |
| D | Noise-shape matching | – | – | – | done inside P4 (quantile map) | |
| E | Drift-scale shift check | – | – | – | small (strong-line drift p90 0.60 vs 0.70) | covered by CNN zoom aug |

## Error analysis (best GBDT A_deg2_deg4)
- **macro-F1 lost per class** (clean / fake-PKS): scint **0.034 / 0.043**, ETI 0.012 / 0.015, rfi_pers 0.009 / 0.011, rfi_int 0.009 / 0.011, radar 0.007 / 0.013, satellite 0.007 / 0.010, noise 0.005 / 0.009.
- Top confusions: scint→noise 265 (fPKS 366), rfi_int→rfi_pers 183, scint→rfi_pers 151, scint→satellite 82, radar→noise (fPKS 126).
- **Scintillation recall depends on what else is in the cadence**: clean sky 0.674, with an ON line 0.423, with steep sweeps 0.279, with glitch 0.537 → contamination, not only faintness. Both GBDT and CNN ≈0.70 F1 → maybe SNR-limited, but NOT proven (contamination effect is real).
- ETI recall by track-strength quartile: 0.66 / 0.95 / 0.975 / 0.97 (clean), 0.47 / 0.93 / 0.975 / 0.975 (fake-PKS) → only the faintest quartile is lost.
- Glitch cadences are ~0.02 worse on fake-PKS (0.874 vs 0.894); slew gap / missing rows don't matter.
- CNN fixes GBDT errors mainly on rfi_int (35), scint (23), satellite (18), radar (11); GBDT fixes CNN on ETI (41), rfi_int (46), noise (25).

## Key facts from the data
- No leaks: no duplicate cadences/panels; ID / row order / target / centre freq / slew gap carry no label signal.
- Test noise robust σ 15.6 vs 11.9 bytes; ln-power noise std 0.45 vs 0.343 (log-χ² with ~11 vs ~18 dof → heavier negative tail, lighter positive tail); GBT has a centre spike at ch 64 (removed).
- Test: 1.16 glitch rows/cadence vs 0.64; 0.37 dropped rows vs 0.15.
- EM-estimated test mix ≈ train mix except scintillation (~4% vs 8%).

## Recommendation
1. **Direction to commit to:** physics-feature GBDT trained on clean + simulated-PKS copies (the backbone, 0.9195 / 0.887) **plus** a CNN trained with injection + fake-PKS mixing (B2, or H if it wins), blended. The two are strongly complementary (oracle 0.958 acc vs 0.925).
2. **First upload now:** `cand_A_d2468_trk2.csv` — calibrates the public LB against our fake-PKS proxy (expect public ≈ 0.86–0.89 if the proxy is right).
3. **Colab (Clark):** upload `colab/h4h_bundle.zip` (858 MB) to `MyDrive/h4h_r2/`, run `colab/signal_hunters_gpu.ipynb` with CFG='B2' (H rejected) → 5-fold CNN probs → blend with GBDT → second upload.
4. Keep ~70% on new information: scintillation under contamination, int-vs-pers multi-source decomposition, simulator realism.

## (c) Next 3 planned experiments
1. 5-fold CNN B2/H on Colab → GBDT+CNN blend (expected +0.008–0.010).
2. Source-aware scintillation: mask ALL detected lines/sweeps/glitches (not just zero-drift lines) before texture; scint-specialist model.
3. More/varied fake-PKS copies + pseudo-labels on real test in the final model.

## When Clark is back (needs you)
1. **Upload #1:** `submissions/cand_A_d2468_trk2.csv` (format verified: 10,000 rows, sample-submission ID order, valid labels).
2. **Colab:** upload `colab/h4h_bundle.zip` (858 MB) to `MyDrive/h4h_r2/`, open `colab/signal_hunters_gpu.ipynb`, T4 runtime, run all with CFG='B2' (H was tested and rejected). ~5 folds × 30 epochs. Copy `out/*part_*.npy` into `data/`.
3. **GitHub:** code is committed locally (2 commits, no AI attribution); `git push` to clarkf19/ML-hackathon when you're ready (data excluded by .gitignore). That repo link is also the code submission.
4. **Cloud sessions ($100):** the raw data can't go into GitHub (npy files > 100 MB), so each session would have to download the dataset from the organisers' Drive link and run `pipeline.sh` (~30 min CPU) first. Best uses: (a) scintillation specialist, (b) curved-sweep satellite features, (c) stacking/blending once Colab CNN outputs exist.

## Reproducibility
`pipeline.sh` rebuilds the features and the GBDT submission from raw data (seeded). All scripts in `src/`.
