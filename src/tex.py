"""Multi-scale texture (scintillation) features -> fe_tex_{name}.csv. Lines/glitch rows masked (set to 0) first.
For white noise block-mean variance * blocksize == 1; scintillation raises it at mid scales in ON panels."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
SC = [(2, 4), (2, 8), (4, 8), (4, 16), (8, 16), (8, 32), (16, 32), (16, 64)]

def feats(Z, bs=2000):
    out = []
    for s in range(0, len(Z), bs):
        z = np.asarray(Z[s:s + bs], dtype=np.float32).copy(); b = len(z)
        valid = (z != 0).any(-1)
        rowm = np.median(z, -1)                                   # glitch rows: broadband
        z[np.abs(rowm) > 1.0] = 0
        chm = z.mean(2)                                           # narrow lines: time-mean per channel per panel
        lm = np.abs(chm) * 4 > 4
        lm = lm | np.roll(lm, 1, -1) | np.roll(lm, -1, -1)
        z = np.where(lm[:, :, None, :], 0, z)
        z = np.clip(z, -4, 4)
        f = {}
        for (a, c) in SC:
            bm = z.reshape(b, 6, 16 // a, a, 128 // c, c).mean((3, 5))
            v = bm.var((2, 3)) * a * c                              # (b,6)
            for p in range(6): f[f'v{a}x{c}_p{p}'] = v[:, p]
            f[f'v{a}x{c}_on'] = v[:, 0::2].mean(1); f[f'v{a}x{c}_off'] = v[:, 1::2].mean(1)
            f[f'v{a}x{c}_d'] = f[f'v{a}x{c}_on'] - f[f'v{a}x{c}_off']; f[f'v{a}x{c}_onmin'] = v[:, 0::2].min(1)
            f[f'v{a}x{c}_dmin'] = v[:, 0::2].min(1) - v[:, 1::2].max(1)
        pos = (z > 1.5).mean((2, 3)); f['pos_d'] = pos[:, 0::2].mean(1) - pos[:, 1::2].mean(1)
        mn = z.mean((2, 3)); f['mean_d'] = mn[:, 0::2].mean(1) - mn[:, 1::2].mean(1); f['mean_onmin'] = mn[:, 0::2].min(1)
        f['masked_frac'] = (z == 0).mean((1, 2, 3))
        out.append(pd.DataFrame(f))
    return pd.concat(out, ignore_index=True)

if __name__ == '__main__':
    for nm in sys.argv[1:]:
        F = feats(np.load(D + rf'\z_{nm}.npy', mmap_mode='r'))
        o = {'train': 'tex_train', 'test': 'tex_test', 'train_deg': 'texd_train', 'train_deg2': 'texd_deg2', 'train_deg4': 'texd_deg4', 'train_deg6': 'texd_deg6', 'train_deg8': 'texd_deg8'}.get(nm, f'tex_{nm}')
        F.to_csv(D + rf'\fe_{o}.csv', index=False); print(nm, F.shape)
