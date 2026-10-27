"""Robust scintillation texture: block MEDIANS (thin lines / single glitch rows can't move them).
-> fe_tex2_{name}.csv"""
import numpy as np, pandas as pd, sys, time
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
SC = [(2, 16), (4, 8), (4, 16), (4, 32), (8, 16), (8, 32), (16, 16), (16, 32)]

def feats(Z, bs=1000):
    out = []
    for s in range(0, len(Z), bs):
        z = np.clip(np.asarray(Z[s:s + bs], dtype=np.float32), -4, 4); b = len(z)
        z = z - np.median(z, axis=3, keepdims=True) * (np.abs(np.median(z, axis=3, keepdims=True)) > 0.8)   # strong broadband rows -> remove level
        f = {}
        for (a, c) in SC:
            blk = z.reshape(b, 6, 16 // a, a, 128 // c, c).transpose(0, 1, 2, 4, 3, 5).reshape(b, 6, 16 // a, 128 // c, a * c)
            md = np.median(blk, -1)                                                    # (b,6,nt,nc)
            v = md.var((2, 3)); m = md.mean((2, 3)); q = np.percentile(md.reshape(b, 6, -1), 90, axis=-1)
            for p in range(6): f[f'm{a}x{c}_v_p{p}'] = v[:, p]
            f[f'm{a}x{c}_v_on'] = v[:, 0::2].mean(1); f[f'm{a}x{c}_v_off'] = v[:, 1::2].mean(1)
            f[f'm{a}x{c}_vr'] = np.log((v[:, 0::2].mean(1) + 1e-4) / (v[:, 1::2].mean(1) + 1e-4))
            f[f'm{a}x{c}_vr_min'] = np.log((v[:, 0::2].min(1) + 1e-4) / (v[:, 1::2].max(1) + 1e-4))
            f[f'm{a}x{c}_q90d'] = q[:, 0::2].mean(1) - q[:, 1::2].mean(1)
            f[f'm{a}x{c}_md'] = m[:, 0::2].mean(1) - m[:, 1::2].mean(1)
        out.append(pd.DataFrame(f))
    return pd.concat(out, ignore_index=True)

if __name__ == '__main__':
    for nm in sys.argv[1:]:
        t = time.time(); Fd = feats(np.load(D + rf'\z_{nm}.npy', mmap_mode='r'))
        Fd.to_csv(D + rf'\fe_tex2_{nm}.csv', index=False); print(nm, Fd.shape, round(time.time() - t), 's', flush=True)
