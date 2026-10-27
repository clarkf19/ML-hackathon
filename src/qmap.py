"""Monotone quantile map of z (non-missing pixels): source marginal -> test marginal. z_{src} -> z_{out}"""
import numpy as np, sys
D = r'C:\Users\clark\hack4hack-r2\data'
src, out = sys.argv[1], sys.argv[2]
qs = np.r_[0.0001, 0.001, np.linspace(0.01, 0.99, 99), 0.999, 0.9999] * 100
def q(nm):
    z = np.load(f'{D}/z_{nm}.npy', mmap_mode='r'); idx = np.random.default_rng(0).choice(len(z), 3000, replace=False)
    v = z[np.sort(idx)].astype(np.float32).ravel(); v = v[v != 0]; return np.percentile(v, qs)
qa, qb = q(src), q('test')
Z = np.load(f'{D}/z_{src}.npy', mmap_mode='r'); O = np.zeros(Z.shape, np.float16)
for s in range(0, len(Z), 2000):
    x = Z[s:s + 2000].astype(np.float32); m = x == 0
    y = np.interp(x, qa, qb)
    y = np.where(x > qa[-1], qb[-1] + (x - qa[-1]), y); y = np.where(x < qa[0], qb[0] + (x - qa[0]), y)
    y[m] = 0; O[s:s + 2000] = y
np.save(f'{D}/z_{out}.npy', O); print('mapped', src, '->', out, 'q50', qa[51].round(3), qb[51].round(3))
