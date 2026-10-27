"""Portable fake-PKS rendering of RAW train cadences (port of degrade.py, same RNG call order -> with the same seed/params
row i is bit-identical to the pipeline's lv_d2 / lv_d4 copies), plus an optional raw-domain port of qmap.py
(monotone quantile map of the bandpass/level-removed residual in bytes, source marginal -> test marginal; levels kept).
usage: python src/rawsim.py           -> data/raw_d2.npy, data/raw_d4.npy (GAINSD=1.9, pipeline seeds for deg2 / deg4)
       QMAP=1 python src/rawsim.py    -> also data/raw_d2q.npy, data/raw_d4q.npy"""
import numpy as np, os
from pathlib import Path
from scipy.special import polygamma
from scipy.optimize import brentq
from scipy.ndimage import median_filter
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / 'data'; SRC = D / 'signal-hunters-round2'
COPIES = {'d2': dict(seed=123), 'd4': dict(seed=321, grate=0.0085, glo=22, ghi=60)}   # pipeline.sh deg2 / deg4 settings

def degrade(X, seed=123, gainsd=1.9, grate=0.0057, glo=18, ghi=50, lnstd=0.45, bs=2000):
    K = brentq(lambda k: polygamma(1, k) - (lnstd ** 2 - 0.343 ** 2), 1, 200)
    rng = np.random.default_rng(seed); out = np.zeros(X.shape, np.uint8)
    for s in range(0, len(X), bs):
        x = X[s:s + bs].astype(np.float32); miss = x == 0
        g = rng.gamma(K, 1 / K, x.shape).astype(np.float32)
        x = x + 24 * np.log2(g) + rng.normal(0, gainsd, (len(x), 6, 1, 1)).astype(np.float32)
        gl = rng.random((len(x), 6, 16)) < grate
        amp = rng.uniform(glo, ghi, gl.shape).astype(np.float32)
        lo = np.where(rng.random(gl.shape) < 0.3, rng.integers(0, 64, gl.shape), 0); hi = np.where(lo > 0, lo + rng.integers(40, 64, gl.shape), 128)
        ch = np.arange(128)
        band = (ch[None, None, None, :] >= lo[..., None]) & (ch[None, None, None, :] < hi[..., None])
        x = x + (gl[..., None] & band) * amp[..., None]
        drop = rng.random((len(x), 6, 16, 1)) < 0.0023
        x = np.clip(np.round(x), 1, 255); x[miss | np.broadcast_to(drop, x.shape)] = 0
        out[s:s + bs] = x.astype(np.uint8)
    return out

def smooth_level(x):
    """bandpass (median over panels+time, 9-ch running median) + per-panel offset, in bytes; x float with nan = missing"""
    bp = median_filter(np.nanmedian(x, axis=(1, 2)), size=(1, 9), mode='nearest')
    r = x - bp[:, None, None, :]
    return bp[:, None, None, :] + np.nanmedian(r, axis=(2, 3), keepdims=True)

QS = np.r_[0.0001, 0.001, np.linspace(0.01, 0.99, 99), 0.999, 0.9999] * 100
def resid_quantiles(X, n=3000):
    idx = np.sort(np.random.default_rng(0).choice(len(X), n, replace=False))
    x = X[idx].astype(np.float32); x[x == 0] = np.nan
    r = (x - smooth_level(x)).ravel(); return np.percentile(r[~np.isnan(r)], QS)

def qmap_raw(X, qa, qb, bs=2000):
    out = np.zeros(X.shape, np.uint8)
    for s in range(0, len(X), bs):
        x = X[s:s + bs].astype(np.float32); m = x == 0; x[m] = np.nan
        S = smooth_level(x); r = x - S
        y = np.interp(r, qa, qb); y = np.where(r > qa[-1], qb[-1] + (r - qa[-1]), y); y = np.where(r < qa[0], qb[0] + (r - qa[0]), y)
        o = np.clip(np.round(S + y), 1, 255); o[m] = 0; out[s:s + bs] = o.astype(np.uint8)
    return out

if __name__ == '__main__':
    X = np.load(SRC / 'train_cadences.npy', mmap_mode='r')
    qt = resid_quantiles(np.load(SRC / 'test_cadences.npy', mmap_mode='r')) if os.environ.get('QMAP') else None
    for nm, kw in COPIES.items():
        Y = degrade(X, **kw); np.save(D / f'raw_{nm}.npy', Y); print(nm, 'done', flush=True)
        if qt is not None:
            qa = resid_quantiles(Y); np.save(D / f'raw_{nm}q.npy', qmap_raw(Y, qa, qt))
            print(nm, 'qmap q50/q01/q99 src', qa[[51, 2, 100]].round(2), 'test', qt[[51, 2, 100]].round(2), flush=True)
