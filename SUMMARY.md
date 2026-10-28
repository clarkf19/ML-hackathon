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
| **submission4 = lvl GBDT + CNN B3 (5 folds) 50/50, averaged with a stacked LGBM meta-model** | **0.9463** | **0.9172** (every fold +0.005…+0.017 over blend50) | submissions/submission4.csv | **uploaded: public 0.9118** (src/final_blend.py) |
| submission5 = submission4 + CNN B3pl (real-test pseudo-labels) as 2nd meta input | 0.9481 | 0.9196 (all 5 folds up) | submissions/submission5.csv | **uploaded: public 0.9170** (pseudo-label gain bigger on real test than on proxy) |
| submission6 = avg of 3 feature models (A_rs raw stats, A_lvl_PL pseudo-labels, A_trk3 ETI slope) + CNNs B3, B3pl, B3w + meta | 0.9505 | 0.9226 | submissions/submission6.csv | **uploaded: public 0.9189** |
| submission7 = 4 feature models (incl. round-3 pseudo-label A_rs_PL3) + CNNs B3pl2, B3w + larger meta | 0.9519 | 0.9230 | submissions/submission7.csv | **uploaded: public 0.9219** |
| submission8 = base (5 feature models + B3pl2, B3w, B3plw, meta) mixed 0.7/0.3 with pretrained CNNs P1 (3 folds) + E1 (2 folds); weight chosen on folds 0-1, fold 2 independent check +0.004 | — | 0.9218→0.9257 (folds 0-1) | submissions/submission8.csv | **uploaded: public 0.9216** (within noise of 0.9219) |
| submission9 = same base mixed 0.5/0.5 with 6 pretrained CNNs (P1 f0-2, E1 f0-2, E3 f0-1, E2/E4/P3 f0) | — | pooled folds 0-2: 0.9249→0.9291 | submissions/submission9.csv | **uploaded: public 0.9226 (best so far)** |
| submission10 = same base mixed 0.5/0.5 with 6 pretrained CNNs, now P1 f0-3 + E2 f0-1 (P1 f3, E2 f1 new) | — | pooled folds 0-3: 0.9248→0.9293; fold 3 (P1 only, independent) 0.9244→0.9290 | submissions/submission10.csv | **uploaded: public 0.9236 (best so far)** — 76 cadences differ from sub9 |
| **submission12 = GSEP meta with 4 complete pretrained CNNs (P1, E1, E2, E3) as separate meta inputs + B3pl2/B3w/B3plw + 6 feature models; E7 (4 folds) + E8 (3 folds) mixed at w 0.15** | 0.9624 | **5-fold base 0.9349** (prev 0.9245); with E7/E8 mix pooled folds 0-3 **0.9382** (sub11 0.9310) | submissions/submission12.csv | ready 07:00 (118 labels differ from sub11) |
| submission11 = round-5 pseudo-label feature models, each a separate meta input (GSEP) + 6 pretrained CNNs mixed (w 0.45) | — | pooled folds 0-3: 0.9310 (sub10 recipe 0.9293), all 4 folds up | submissions/submission11.csv | ready (night build) |
| submission3 = lvl GBDT + CNN B3 (4 folds) 50/50 | 0.9381 (4 folds) | 0.9089 (4 folds) | submissions/submission3.csv | **uploaded: public 0.8991** |
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

## Night 27→28 Oct (autonomous; see NIGHT_PLAN.md)
| Experiment | fake-PKS | Verdict |
|---|---|---|
| Per-class bias tuning for macro-F1 (tuned on 2 folds, scored on the other 2) | +0.0006 / +0.0016 | negligible |
| Mixing variants (geometric, log-avg, E-family ×1.5/×2, w 0.4–0.6) | 0.9286–0.9293 (ref 0.9293) | saturated |
| Scintillation F1 per model: GBDT 0.68–0.72, every CNN 0.69–0.72, blend 0.735 | — | same ceiling for all model families → SNR-limited on the proxy |
| Test cadences predicted noise: brightening (lv_on_off) quantiles = fake-PKS noise | — | no sign of extra hidden scintillation in test |
| HIP target repeats (1,230 in train): pair same-label rate 0.140 vs chance 0.142; centre freq: flat | — | no signal |
| A_rs_PL4_big (63 leaves, 1600 trees) | 0.9019 (vs 0.9024) | no gain |
| base8_4 = base8_3 + A_rsswp_PL4 in the feature average | FINAL 0.9242 (vs 0.9234); mixed with 6 pretrained 0.9297 (vs 0.9293) | tiny + |
| A_rsswp_PL5 (round-5 pseudo-labels from submission10, p>0.8) | **0.9054** (PL4 0.9036), clean 0.9367 | best feature model so far |
| A_rs_PL5 (same, without sweep features) | 0.9020 (PL4 0.9024), clean 0.9365 | neutral on proxy |
| A_trk3_PL5 (ETI slope set + round-5 pseudo-labels) | 0.9002 (A_trk3 no PL 0.8982), clean 0.9308 | +0.002 |
| A_rsswp_PL5t6 (round-5 pseudo-labels, lower threshold p>0.6 → ~9.6k test rows) | **0.9056**, clean **0.9374** | best feature model (by a hair) |
| A_rsswp_PL5w3 (real-test rows weighted ×3) | 0.9046 (v4 0.9035), clean 0.9374 | same as unweighted on proxy; LB-only question |
| A_rsswp_PL5soft (soft pseudo-labels: every test cadence once per class with p>0.05, weight 2p) | **0.8351**, clean 0.8716 | **rejected** (low-p rows act as label noise, amplified by balanced class weights); files deleted |
| A_rsswp_PL5_s11 / _s22 (seeds 11, 22) | 0.9053 / 0.9051 (seed 0: 0.9054) | stable across seeds; average them in the base |
| A_trk3_PL5t6 (ETI slope set, p>0.6) | 0.9011 (p>0.8: 0.9002) | small + |
| A_rs_PL5t6 (no sweep set, p>0.6) | 0.9032, clean 0.9374 | +0.001 over p>0.8 |
| base_b9a / base_b9b = round-5 feature models in the base, + 6 pretrained mix (cand_b9amix / cand_b9bmix.csv) | FINAL 0.9241; mixed 0.9293 / 0.9294 (sub10 recipe 0.9293) | flat on proxy (expected for pseudo-labels); LB test candidate |
| base_n1 = 7 round-5 feature models (3 seeds, t6, w3, rs_t6, trk3_t6) | FINAL 0.9241; + pretrained mix 0.9294 (cand_n1mix.csv, 62 test labels differ from sub10) | = sub10 recipe |
| base_n2 = n1 + A_rsswp_PL4, A_rs_PL4, A_rs_PL3 | FINAL 0.9247; + pretrained mix **0.9302** (cand_n2mix.csv, 57 differ) | +0.0009 vs sub10 recipe |
| GSEP=1 (each feature model its own meta input), 4 round-5 models | FINAL **0.9255** (vs 0.9241 averaged) | +0.0014 → use GSEP |
| **base_g2 = GSEP meta over A_rsswp_PL5, A_rsswp_PL5t6, A_rs_PL5t6, A_trk3_PL5t6, A_rsswp_PL4, A_rs_PL3 + 6 pretrained mix (w 0.45)** | **pooled folds 0-3 0.9310** (sub10 recipe 0.9293; every fold ≥ sub10: 0.9278/0.9306/0.9344/0.9311 vs 0.9270/0.9269/0.9341/0.9290) | **→ submissions/submission11.csv** (47 test labels differ from sub10) |
| base_g1 = GSEP over the 4 round-5 models only + mix | 0.9302 (76 differ) | second choice |
| A_rsswp_PL6t6 (round-6 pseudo-labels from submission11, p>0.6) | 0.9048 (v4 0.9038; round 5: 0.9056) | same within noise |
| A_rs_PL6t6 / A_trk3_PL6t6 (round 6) | 0.9025 / 0.9003 (round 5: 0.9032 / 0.9011) | pseudo-label rounds saturated on the proxy |
| base_g3 = g2 + the 3 round-6 models (9 separate meta inputs) + mix | 0.9305 (g2 0.9310) | no gain → keep submission11 |
| **Pretrained CNNs as meta inputs vs mixing** (folds 0-1, meta trained on 1 fold) | **0.9298 vs 0.9269** | **+0.003 → morning: complete folds go into the meta** |

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
