"""Portable (Linux/Windows) port of degrade.py: fake-PKS rendering of the RAW train cadences (log-gamma noise to PKS ln-power
std, panel gain jitter, extra glitch rows, extra dropped rows). Same env knobs and RNG call order as degrade.py, so a given
SEED/GRATE/GLO/GHI/LNSTD/GAINSD reproduces the Windows output. Use GAINSD=1.9 (test-matched) for level-dependent statistics.
paths: H4H_RAW (default data/signal-hunters-round2) holds train_cadences.npy; output -> H4H_DATA/{OUT}_cadences.npy
usage: OUT=lv_d2 SEED=123 GAINSD=1.9 python src/degrade_lin.py"""
import numpy as np, os
from pathlib import Path
from scipy.special import polygamma
from scipy.optimize import brentq
R = Path(__file__).resolve().parent.parent
RAW = Path(os.environ.get('H4H_RAW', R / 'data' / 'signal-hunters-round2')); D = Path(os.environ.get('H4H_DATA', R / 'data'))
K = brentq(lambda k: polygamma(1, k) - (float(os.environ.get('LNSTD', 0.45)) ** 2 - 0.343 ** 2), 1, 200)
rng = np.random.default_rng(int(os.environ.get('SEED', 123)))
X = np.load(RAW / 'train_cadences.npy', mmap_mode='r'); out = np.zeros(X.shape, np.uint8)
for s in range(0, len(X), 2000):
    x = X[s:s + 2000].astype(np.float32); miss = x == 0
    g = rng.gamma(K, 1 / K, x.shape).astype(np.float32)
    x = x + 24 * np.log2(g) + rng.normal(0, float(os.environ.get('GAINSD', 3)), (len(x), 6, 1, 1)).astype(np.float32)
    gl = rng.random((len(x), 6, 16)) < float(os.environ.get('GRATE', 0.0057))
    amp = rng.uniform(float(os.environ.get('GLO', 18)), float(os.environ.get('GHI', 50)), gl.shape).astype(np.float32)
    lo = np.where(rng.random(gl.shape) < 0.3, rng.integers(0, 64, gl.shape), 0); hi = np.where(lo > 0, lo + rng.integers(40, 64, gl.shape), 128)
    ch = np.arange(128)
    band = (ch[None, None, None, :] >= lo[..., None]) & (ch[None, None, None, :] < hi[..., None])
    x = x + (gl[..., None] & band) * amp[..., None]
    drop = rng.random((len(x), 6, 16, 1)) < 0.0023
    x = np.clip(np.round(x), 1, 255); x[miss | np.broadcast_to(drop, x.shape)] = 0
    out[s:s + 2000] = x.astype(np.uint8)
D.mkdir(parents=True, exist_ok=True); np.save(D / f"{os.environ.get('OUT', 'train_deg')}_cadences.npy", out); print('K', K)
