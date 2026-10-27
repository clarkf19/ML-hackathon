# Night plan 27→28 Oct (Clark asleep until 06:30; round ends 10:00; 10 uploads after midnight)

Best public: **submission10 = 0.9236** (base8_3 mixed 0.5 with 6 pretrained CNNs). Proxy (fake-PKS folds 0-3) 0.9293.

## Facts established tonight (numbers)
- Bias tuning for macro-F1: +0.0006 / +0.0016 held-out → negligible.
- Mixing variants (geo/log/E-weight/w 0.4-0.6): all 0.9286-0.9293 → saturated.
- Scintillation F1 ≈0.70-0.72 for EVERY model (GBDT, each CNN, blend 0.735) → SNR ceiling on the proxy.
- Target (HIP) repeats carry no label signal (pair same-label 0.140 vs chance 0.142); centre freq: no label/accuracy dependence.
- **Complete pretrained CNNs as meta inputs beat mixing**: folds 0-1, meta with P1,E1,E2,E3 (+old B3s) 0.9298 vs mix 0.9269 (+0.003, meta trained on 1 fold only).

## Laptop queue (CPU)
1. run_local_q.sh: A_rs_PL4_big (logs/local_q.out, ALLDONE)
2. run_night2.sh → logs/night2.out (NIGHT2DONE): round-5 pseudo-labels from submission10:
   A_rsswp_PL5, A_rs_PL5, A_trk3_PL5, A_rsswp_PL5t6 (PLTH 0.6)
3. run_night3.sh → logs/night3.out (NIGHT3DONE): A_rsswp_PL5w3 (test rows ×3), A_rsswp_PL5soft (soft PL over all test)
Results land in logs/aug_results.txt.

## After queues: build candidates
- base10 = final_blend with GBDT = best of {PL5 models} + A_rs_PL4,A_rs_PL3,A_rs,A_lvl_PL,A_trk3 and CNNs B3pl2g,B3wg,B3plwg:
  `GBDT=... python src/final_blend.py base10.csv B3pl2g B3wg B3plwg` (writes oofd_final_base10 / pte_base10)
- then `PTAGS=P1g,E1g,E2g,E3g,E4g,P3g ANYFOLD=1 python src/mix_partial.py base10 cand_*.csv`
- Compare pooled fold scores vs base8_3 (0.9248 → 0.9293).

## Morning (Clark): biggest expected gains
1. Copy all new Colab out/ files into data/; any run with 5 folds goes INTO the meta (final_blend SEP=1), the rest via mix_partial.
2. Upload colab/pl_test5.npy to Drive h4h_r2/; run configs E2p5 / E1p5 one fold per account (FOLDS=[k]) — ~45 min for b2 on T4.
