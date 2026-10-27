"""Variants of mixing the pretrained CNNs into the base: prob vs log-prob average, per-family weights, temperature.
Scored on fake-PKS folds 0-3 (folds with any pretrained CNN), weight shared across folds."""
import sys, os; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
y = y_train(); fo = folds()
BASE = os.environ.get('BASE', 'base8_3'); F = np.load(os.path.join(D, f'oofd_final_{BASE}.npy'))
TAGS = 'P1g,E1g,E2g,E3g,E4g,P3g'.split(',')
have = {t: [k for k in range(5) if os.path.exists(os.path.join(D, f'oofdpart_{t}_f{k}.npy'))] for t in TAGS}
ks = sorted({k for t in TAGS for k in have[t]}); idx = np.concatenate([np.where(fo == k)[0] for k in ks])
def cnn(k, mode, wt):
    cs = [(np.load(os.path.join(D, f'oofdpart_{t}_f{k}.npy')), wt.get(t[0], 1)) for t in TAGS if k in have[t]]
    W = sum(w for _, w in cs)
    if mode == 'log': return np.exp(sum(w * np.log(c + 1e-6) for c, w in cs) / W)
    return sum(w * c for c, w in cs) / W
def score(mode='prob', wt={}, w=0.5, comb='lin'):
    out = []
    for k in ks:
        v = fo == k; c = cnn(k, mode, wt); c = c / c.sum(1, keepdims=True)
        out.append(((1 - w) * F[v] + w * c) if comb == 'lin' else np.exp((1 - w) * np.log(F[v] + 1e-6) + w * np.log(c + 1e-6)))
    P = np.concatenate(out); yy = np.concatenate([y[fo == k] for k in ks]); return mf1(yy, P.argmax(1))
print('ref prob/lin w=.5', round(score(), 4))
for comb in ['lin', 'geo']:
    for mode in ['prob', 'log']:
        print(comb, mode, [(w, round(score(mode, {}, w, comb), 4)) for w in [0.4, 0.5, 0.6, 0.7]])
for we in [1.5, 2.0]:
    print('E weight', we, round(score('prob', {'E': we}, 0.5), 4), round(score('prob', {'E': we}, 0.6), 4))
