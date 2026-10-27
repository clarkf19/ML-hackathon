"""Panel LEVEL features in LINEAR power (brightening = added power -> linear ON/OFF power ratio is the natural statistic).
Reference = OFF-panel per-channel median power. Per panel: trimmed-mean power ratio; per row (time modulation); per band
at 4/16/32-channel scales. usage: lvl2.py RAWNAME OUTNAME"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
src, outn = sys.argv[1], sys.argv[2]
X = np.load(SRC + rf'\{src}_cadences.npy' if src in ('train', 'test') else D + rf'\{src}_cadences.npy', mmap_mode='r')
def tmean(a, axis):
    s = np.sort(a, axis=axis); n = a.shape[axis]; k = n // 10
    return np.take(s, range(k, n - k), axis=axis).mean(axis)
rows = []
for s0 in range(0, len(X), 1000):
    v = X[s0:s0 + 1000].astype(np.float32); miss = v == 0
    P = 2 ** (v / 24 - 4); P[miss] = np.nan; b = len(P)
    ref = np.nanmedian(P[:, 1::2].transpose(0, 3, 1, 2).reshape(b, 128, -1), axis=2)[:, None, None, :]   # (b,1,1,128)
    R = np.nan_to_num(P / ref, nan=1.0)                                   # power ratio to OFF reference
    f = {}
    pan = tmean(R.reshape(b, 6, -1), 2)                                   # (b,6)
    for p in range(6): f[f'l2_p{p}'] = np.log(pan[:, p])
    on, off = np.log(pan[:, 0::2]), np.log(pan[:, 1::2])
    f['l2_on_off'] = on.mean(1) - off.mean(1); f['l2_onmin_offmax'] = on.min(1) - off.max(1)
    rowl = np.log(tmean(R, 3))                                            # (b,6,16) per-row level
    f['l2_row_std_on'] = rowl[:, 0::2].std(2).mean(1); f['l2_row_std_off'] = rowl[:, 1::2].std(2).mean(1)
    f['l2_row_std_d'] = f['l2_row_std_on'] - f['l2_row_std_off']
    f['l2_row_max_d'] = rowl[:, 0::2].max(2).mean(1) - rowl[:, 1::2].max(2).mean(1)
    f['l2_row_frac_up'] = (rowl[:, 0::2] > 0.05).mean((1, 2)) - (rowl[:, 1::2] > 0.05).mean((1, 2))
    for w in (4, 16, 32):
        bl = np.log(tmean(R.reshape(b, 6, 16, 128 // w, w).transpose(0, 1, 3, 2, 4).reshape(b, 6, 128 // w, -1), 3))  # (b,6,nb)
        d = bl[:, 0::2].mean(1) - bl[:, 1::2].mean(1)
        f[f'l2_b{w}_std'] = d.std(1); f[f'l2_b{w}_max'] = d.max(1); f[f'l2_b{w}_min'] = d.min(1)
        f[f'l2_b{w}_fracpos'] = (d > 0.05).mean(1)
        f[f'l2_b{w}_onmin_max'] = (bl[:, 0::2].min(1) - bl[:, 1::2].max(1)).max(1)
        f[f'l2_b{w}_onstd_offstd'] = bl[:, 0::2].std(2).mean(1) - bl[:, 1::2].std(2).mean(1)
    rows.append(pd.DataFrame(f))
F = pd.concat(rows, ignore_index=True); F.to_csv(D + rf'\fe_lvl2_{outn}.csv', index=False); print(outn, F.shape, flush=True)
