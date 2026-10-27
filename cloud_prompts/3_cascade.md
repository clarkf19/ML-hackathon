Context: Hack4Hack 2026 Round 2 (ends 28 Oct 10:00 IST). 9-class classification of synthetic SETI radio cadences (6 panels ON/OFF/ON/OFF/ON/OFF x 16 time x 128 freq, uint8 log power), macro-F1. Train = GBT telescope, test = PKS (noisier, more glitches, larger panel gain jitter). Our best: ~430 physics features + LightGBM on clean + simulated-PKS copies: CV 0.929 clean / 0.897 fake-PKS; public LB leader 0.912. Read SUMMARY.md and CLOUD.md in this repo first. Score every idea with `python src/bundle_eval.py` (baseline first) and report clean-val AND fakePKS-val macro-F1. Deliverable: results/REPORT.md + scripts on your own branch, committed and pushed; negative results with numbers are fine.

TASK: Better decision layer on the committed feature bundle (no raw data needed). Label = highest-priority source (eti > frb > pulsar > scint > satellite > radar > rfi_intermittent > rfi_persistent > noise); errors concentrate where a weak higher-priority source hides behind a strong lower-priority one.
Always train on clean+deg2+deg4 and score clean val and deg2 val with bundle/meta.npz['folds']. Try:
a. priority cascade of binary detectors applied top-down, thresholds by nested CV (tune 4 folds, score 5th);
b. one-vs-rest LightGBMs + calibration, argmax vs priority-aware rules;
c. pairwise specialists for scint vs noise/rfi_persistent/satellite, rfi_intermittent vs rfi_persistent, radar vs noise, overriding when confident;
d. nested per-class probability biases (a quick local test gave +0.0015) on top of a-c;
e. CatBoost / XGBoost / MLP on the same features and simple averages with the baseline (diversity gain).
Save the best variant's probabilities as results/probs_best.npz (oofc, oofd, pte).
