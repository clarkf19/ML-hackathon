"""Smoke test for src/scint_raw.py on SYNTHETIC bytes in a temp dir: shapes, NaN/inf, runtime only (NOT evidence of usefulness).
usage: python src/scint_raw_smoke.py"""
import numpy as np, os, sys, subprocess, tempfile
from scipy.ndimage import gaussian_filter
DD = tempfile.mkdtemp(); os.makedirs(DD + '/signal-hunters-round2', exist_ok=True)
rng = np.random.default_rng(0); N = 600
bp = 60 + 20 * np.sin(np.linspace(0, 3, 128))                      # bandpass (bytes)
x = np.zeros((N, 6, 16, 128), np.float32); y = np.zeros(N, int)
for i in range(N):
    noise = rng.chisquare(18, (6, 16, 128)) / 18
    lin = noise * 2 ** (bp / 24)
    if i % 2:                                                       # star in ON panels: additive linear power, scintillating
        y[i] = 1
        for p in (0, 2, 4):
            fld = gaussian_filter(rng.normal(size=(16, 128)), (2, 6)); fld = np.exp(2 * fld / fld.std())
            lin[p] += 0.15 * fld / fld.mean() * 2 ** (np.median(bp) / 24)
    xb = 24 * np.log2(lin) + rng.normal(0, 1.9, (6, 1, 1))
    x[i] = xb
x = np.clip(np.round(x + 40), 1, 255).astype(np.uint8)
x[3, 1, 5] = 0                                                       # a dropped row
np.save(DD + '/signal-hunters-round2/train_cadences.npy', x); np.save(DD + '/signal-hunters-round2/test_cadences.npy', x[:200])
env = dict(os.environ, H4H_DATA=DD); S = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scint_raw.py')
for a in (['degrade', 'lv_d2', '123'], ['feats', 'train', 'train'], ['feats', 'lv_d2', 'train_deg2']):
    subprocess.run([sys.executable, S, *a], env=env, check=True)
from sklearn.metrics import roc_auc_score
for nm in ('train', 'train_deg2'):
    z = np.load(DD + f'/fe_sraw_{nm}.npz'); X = z['X']; print(nm, X.shape, 'nan', int(np.isnan(X).sum()), 'inf', int(np.isinf(X).sum()))
    au = sorted([(roc_auc_score(y, X[:, j]), c) for j, c in enumerate(z['cols'])], reverse=True)
    print(' top', [(c, round(a, 3)) for a, c in au[:8]]); print(' bottom', [(c, round(a, 3)) for a, c in au[-4:]])
