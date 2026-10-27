"""Source-aware scintillation texture: per panel mask up to 4 detected lines (any drift, +-2 ch) and glitch rows
(full or partial band), then multi-scale ON-vs-OFF block-mean variance on what is left. -> fe_tex3_{name}.csv"""
import numpy as np, pandas as pd, torch, torch.nn.functional as F, sys, os, time
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
from phys import smooth_f, PAD
torch.set_num_threads(int(os.environ.get('NT', 6)))
DR = np.r_[np.arange(-1, 1.001, 0.125), np.arange(1.5, 8.01, 0.5), -np.arange(1.5, 8.01, 0.5)]
R = torch.arange(16) - 7.5
SC = [(2, 8), (4, 8), (4, 16), (8, 16), (8, 32), (16, 32)]

def feats(Z, bs=400):
    out = []
    for s0 in range(0, len(Z), bs):
        z = torch.from_numpy(np.asarray(Z[s0:s0 + bs], dtype=np.float32)); b = len(z)
        valid = (z != 0).any(-1)
        zc = z.clamp(-4, 4)
        lit = (zc > 1.5).float().mean(-1)                                            # glitch rows (full/partial band)
        grow = (zc.median(-1).values.abs() > 0.8) | (lit > 0.4) | ~valid
        zn = smooth_f((z - z.median(-1, keepdim=True).values) * valid[..., None]).clamp(-3, 6)
        zp = F.pad(zn, (PAD, PAD))
        L = torch.stack([torch.gather(zp, 3, (torch.arange(128)[None, :] + PAD + torch.round(float(d) * R).long()[:, None])[None, None].expand(b, 6, 16, 128)).sum(2) / 4.0 for d in DR], 2)  # (b,6,nd,128)
        mask = grow[..., None].expand(b, 6, 16, 128).clone()
        Lw = L.clone()
        for it in range(4):
            v, i = Lw.reshape(b, 6, -1).max(-1); di = i // 128; ci = i % 128
            ok = v > 4.0
            d = torch.from_numpy(DR).float()[di]                                     # (b,6)
            ch = ci[..., None].float() + torch.round(d[..., None] * R)               # (b,6,16)
            for o in range(-2, 3):
                c = (ch + o).long().clamp(0, 127)
                mm = torch.zeros(b, 6, 16, 128, dtype=torch.bool).scatter_(3, c[..., None], True)
                mask |= mm & ok[..., None, None]
            # suppress this detection (and neighbours) for the next iteration
            sup = torch.zeros_like(Lw, dtype=torch.bool)
            for o in range(-4, 5):
                cc = (ci + o).clamp(0, 127)
                sup.scatter_(3, cc[..., None, None].expand(b, 6, len(DR), 1), True)
            Lw = Lw.masked_fill(sup, -1e9)
        x = zc.masked_fill(mask, 0.0)
        frac = 1 - mask.float().mean((2, 3))                                         # kept fraction per panel
        f = {'t3_kept_on': frac[:, 0::2].mean(1), 't3_kept_off': frac[:, 1::2].mean(1)}
        for (a, c) in SC:
            xs = x.reshape(b, 6, 16 // a, a, 128 // c, c); ms = (~mask).float().reshape(b, 6, 16 // a, a, 128 // c, c)
            n = ms.sum((3, 5)); bm = xs.sum((3, 5)) / n.clamp(min=1)
            w = (n >= a * c * 0.5).float()                                           # blocks at least half unmasked
            var = ((bm ** 2) * n * w).sum((2, 3)) / w.sum((2, 3)).clamp(min=1)      # ~1 for white noise
            f[f't3_{a}x{c}_on'] = var[:, 0::2].mean(1); f[f't3_{a}x{c}_off'] = var[:, 1::2].mean(1)
            f[f't3_{a}x{c}_d'] = f[f't3_{a}x{c}_on'] - f[f't3_{a}x{c}_off']
            f[f't3_{a}x{c}_onmin_offmax'] = var[:, 0::2].min(1).values - var[:, 1::2].max(1).values
            mean = ((bm * n * w).sum((2, 3)) / (n * w).sum((2, 3)).clamp(min=1))
            f[f't3_{a}x{c}_mean_d'] = mean[:, 0::2].mean(1) - mean[:, 1::2].mean(1)
        out.append(pd.DataFrame({k: v.numpy() for k, v in f.items()}))
    return pd.concat(out, ignore_index=True)

if __name__ == '__main__':
    for nm in sys.argv[1:]:
        t = time.time(); Fd = feats(np.load(D + rf'\z_{nm}.npy', mmap_mode='r'))
        Fd.to_csv(D + rf'\fe_tex3_{nm}.csv', index=False); print(nm, Fd.shape, round(time.time() - t), 's', flush=True)
