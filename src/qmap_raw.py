"""Raw-domain analogue of qmap.py (optional step for fake-PKS raw renderings): monotone quantile map of the bandpass residual
r = x - per-cadence channel median (over panels+time), source marginal -> test marginal, then bandpass added back, rounded,
clipped to 1..255; dropped pixels (0) stay 0. Reference quantiles from 3000 cadences (seed 0) as in qmap.py.
usage: python src/qmap_raw.py SRCNAME OUTNAME   (H4H_DATA/{SRCNAME}_cadences.npy -> H4H_DATA/{OUTNAME}_cadences.npy)"""
import numpy as np, os, sys
from pathlib import Path
R = Path(__file__).resolve().parent.parent
RAW = Path(os.environ.get('H4H_RAW', R / 'data' / 'signal-hunters-round2')); D = Path(os.environ.get('H4H_DATA', R / 'data'))
qs = np.r_[0.0001, 0.001, np.linspace(0.01, 0.99, 99), 0.999, 0.9999] * 100

def resid(x):
    x = x.astype(np.float32); m = x == 0; x[m] = np.nan
    bp = np.nanmedian(x.reshape(len(x), -1, x.shape[-1]), axis=1)
    return x - bp[:, None, None, :], np.nan_to_num(bp), m

def q(path):
    X = np.load(path, mmap_mode='r'); idx = np.sort(np.random.default_rng(0).choice(len(X), min(3000, len(X)), replace=False))
    r = resid(X[idx])[0].ravel(); return np.percentile(r[~np.isnan(r)], qs)

if __name__ == '__main__':
    src, outn = sys.argv[1], sys.argv[2]
    qa, qb = q(D / f'{src}_cadences.npy'), q(RAW / 'test_cadences.npy')
    X = np.load(D / f'{src}_cadences.npy', mmap_mode='r'); O = np.zeros(X.shape, np.uint8)
    for s in range(0, len(X), 1000):
        r, bp, m = resid(X[s:s + 1000]); r = np.nan_to_num(r)
        y = np.interp(r, qa, qb)
        y = np.where(r > qa[-1], qb[-1] + (r - qa[-1]), y); y = np.where(r < qa[0], qb[0] + (r - qa[0]), y)
        y = np.clip(np.round(y + bp[:, None, None, :]), 1, 255); y[m] = 0; O[s:s + 1000] = y.astype(np.uint8)
    np.save(D / f'{outn}_cadences.npy', O); print('mapped', src, '->', outn, 'q50', qa[51].round(2), qb[51].round(2))
