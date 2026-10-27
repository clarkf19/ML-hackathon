"""Telescope-agnostic normalisation -> data/z_{train,test}.npy (float16, N x 6 x 16 x 128).
Per cadence: mask dropped rows; smooth bandpass (median over panels+time per channel, then
running median over 9 channels so narrow lines survive); per-panel gain offset; divide by robust noise std.
GBT centre spike (ch 64) is removed by its own median. Missing rows -> 0."""
import numpy as np, sys
from scipy.ndimage import median_filter
D = r'C:\Users\clark\hack4hack-r2\data'
SRC = D + r'\signal-hunters-round2'

def normalise(X, spike=False, bs=1000):
    N = len(X); out = np.zeros(X.shape, np.float16); sig = np.zeros(N, np.float32)
    for s in range(0, N, bs):
        x = X[s:s + bs].astype(np.float32)
        miss = (x == 0).all(axis=3, keepdims=True)
        x[np.broadcast_to(miss, x.shape)] = np.nan
        bp = np.nanmedian(x, axis=(1, 2))                       # (b,128)
        bps = median_filter(bp, size=(1, 9), mode='nearest')
        if spike: bps[:, 64] = bp[:, 64]
        r = x - bps[:, None, None, :]
        r -= np.nanmedian(r, axis=(2, 3), keepdims=True)        # per-panel gain
        sd = 1.4826 * np.nanmedian(np.abs(r), axis=(1, 2, 3))
        r /= sd[:, None, None, None]
        out[s:s + bs] = np.nan_to_num(np.clip(r, -20, 40), nan=0.0)
        sig[s:s + bs] = sd
    return out, sig

if __name__ == '__main__':
    for nm, spike in [(n, not n.startswith('test')) for n in (sys.argv[1:] or ['train', 'test'])]:
        X = np.load(f'{SRC}/{nm}_cadences.npy' if nm in ('train', 'test') else f'{D}/{nm}_cadences.npy', mmap_mode='r')
        z, sd = normalise(X, spike)
        np.save(f'{D}/z_{nm}.npy', z); np.save(f'{D}/sig_{nm}.npy', sd)
        print(nm, z.shape, 'sigma median', np.median(sd))
