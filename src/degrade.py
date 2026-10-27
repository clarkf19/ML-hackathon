"""Fake-PKS copy of train: add log-gamma (multiplicative gamma) noise so ln-power noise std 0.343 -> 0.45 and skew matches,
extra dropped rows (0.15% -> 0.38%), panel gain jitter. -> data/train_deg_cadences.npy"""
import numpy as np, sys, os
from scipy.special import polygamma
from scipy.optimize import brentq
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
K = float(sys.argv[1]) if len(sys.argv) > 1 else None
if K is None: K = brentq(lambda k: polygamma(1, k) - (float(os.environ.get('LNSTD', 0.45)) ** 2 - 0.343 ** 2), 1, 200)
rng = np.random.default_rng(int(os.environ.get('SEED', 123)))
X = np.load(SRC + r'\train_cadences.npy', mmap_mode='r'); out = np.zeros(X.shape, np.uint8)
for s in range(0, len(X), 2000):
    x = X[s:s + 2000].astype(np.float32); miss = x == 0
    g = rng.gamma(K, 1 / K, x.shape).astype(np.float32)
    x = x + 24 * np.log2(g) + rng.normal(0, 3, (len(x), 6, 1, 1)).astype(np.float32)     # noise + gain jitter
    gl = rng.random((len(x), 6, 16)) < float(os.environ.get('GRATE', 0.0057))                                           # extra broadband glitch rows
    amp = rng.uniform(float(os.environ.get('GLO', 18)), float(os.environ.get('GHI', 50)), gl.shape).astype(np.float32)
    lo = np.where(rng.random(gl.shape) < 0.3, rng.integers(0, 64, gl.shape), 0); hi = np.where(lo > 0, lo + rng.integers(40, 64, gl.shape), 128)
    ch = np.arange(128)
    band = (ch[None, None, None, :] >= lo[..., None]) & (ch[None, None, None, :] < hi[..., None])
    x = x + (gl[..., None] & band) * amp[..., None]
    drop = rng.random((len(x), 6, 16, 1)) < 0.0023
    x = np.clip(np.round(x), 1, 255); x[miss | np.broadcast_to(drop, x.shape)] = 0
    out[s:s + 2000] = x.astype(np.uint8)
np.save(D + '/' + os.environ.get('OUT', 'train_deg') + '_cadences.npy', out); print('K', K)
