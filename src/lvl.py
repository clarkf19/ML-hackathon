"""Panel LEVEL features from RAW bytes (prep.py removes per-panel level -> scintillating stars / pulsars lose their
ON-panel brightening). Bandpass flattened by cadence per-channel median; level = trimmed mean of (panel channel-median
over time - cadence channel-median). Also per sub-band (8 x 16 ch) and per time block (4 x 4 rows).
usage: lvl.py RAWNAME OUTNAME   (RAWNAME 'train'/'test' = competition files, else data/{RAWNAME}_cadences.npy)"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
src, outn = sys.argv[1], sys.argv[2]
X = np.load(SRC + rf'\{src}_cadences.npy' if src in ('train', 'test') else D + rf'\{src}_cadences.npy', mmap_mode='r')
def tmean(a, axis):  # 10% trimmed mean
    s = np.sort(a, axis=axis); n = a.shape[axis]; k = n // 10
    return np.take(s, range(k, n - k), axis=axis).mean(axis)
rows = []
for s0 in range(0, len(X), 1000):
    x = X[s0:s0 + 1000].astype(np.float32); x[x == 0] = np.nan; b = len(x)
    chm = np.nanmedian(x, axis=2)                                   # (b,6,128) per-panel channel medians over time
    ref = np.nanmedian(chm[:, 1::2], axis=1, keepdims=True)         # OFF-panel reference bandpass (b,1,128)
    rel = np.nan_to_num(chm - ref)                                  # (b,6,128)
    lv = tmean(rel, 2)                                              # (b,6)
    band = tmean(rel.reshape(b, 6, 8, 16), 3)                       # (b,6,8)
    xr = np.nan_to_num(x - ref[:, :, None, :])                      # (b,6,16,128)
    tb = tmean(xr.reshape(b, 6, 4, 4 * 128), 3)                     # (b,6,4) time blocks
    f = {}
    for p in range(6): f[f'lv_p{p}'] = lv[:, p] - lv[:, 1::2].mean(1)
    on, off = lv[:, 0::2], lv[:, 1::2]
    f['lv_on_off'] = on.mean(1) - off.mean(1); f['lv_onmin_offmax'] = on.min(1) - off.max(1)
    f['lv_onmin_offmean'] = on.min(1) - off.mean(1); f['lv_off_spread'] = off.max(1) - off.min(1); f['lv_on_spread'] = on.max(1) - on.min(1)
    bd = band[:, 0::2].mean(1) - band[:, 1::2].mean(1)              # (b,8)
    for k in range(8): f[f'lv_band{k}'] = bd[:, k]
    f['lv_band_std'] = bd.std(1); f['lv_band_max'] = bd.max(1); f['lv_band_min'] = bd.min(1)
    f['lv_band_onmin'] = (band[:, 0::2].min(1) - band[:, 1::2].max(1)).max(1)
    td = tb[:, 0::2].reshape(b, -1); to = tb[:, 1::2].reshape(b, -1)
    f['lv_tblk_std_on'] = td.std(1); f['lv_tblk_std_off'] = to.std(1); f['lv_tblk_max_d'] = td.max(1) - to.max(1)
    rows.append(pd.DataFrame(f))
F = pd.concat(rows, ignore_index=True); F.to_csv(D + rf'\fe_lvl_{outn}.csv', index=False); print(outn, F.shape)
