"""Bidirectional fast-sweep search (radar chirps / satellites / dispersed pulses) -> fe_swp_{name}.csv.
A sweep crossing the band in s rows: arrival row(c) = r0 + sign * s * shape(c), for sign = +1 (later at HIGHER freq,
'unphysical' for dispersion) and -1 (later at LOWER freq, like dispersion), linear and curved shapes, s = 1..15 rows.
Per panel: best SNR per sign, number of separate sweeps (sawtooth repeats), regularity of their spacing."""
import numpy as np, pandas as pd, torch, torch.nn.functional as F, sys, os, time
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
torch.set_num_threads(int(os.environ.get('NT', 6)))
U = torch.arange(128) / 127.0

def feats(Z, bs=400):
    out = []
    for s0 in range(0, len(Z), bs):
        z = torch.from_numpy(np.asarray(Z[s0:s0 + bs], dtype=np.float32)); b = len(z)
        valid = (z != 0).any(-1)
        z = (z - z.median(-1, keepdim=True).values * (z.median(-1, keepdim=True).values.abs() > 0.8)) * valid[..., None]   # drop glitch-row level
        z = z - z.mean(2, keepdim=True)                                         # remove persistent (zero-drift) lines
        zp = F.pad(z.clamp(-3, 6), (0, 0, 16, 16))                              # (b,6,48,128)
        res = {}
        for sign in (1, -1):
            best = torch.full((b, 6), -1e9); bestser = torch.zeros(b, 6, 32)
            for curv in (0.0, 1.0):
                shp = (U + curv * U ** 2) / (1 + curv) if sign > 0 else ((1 - U) + curv * (1 - U) ** 2) / (1 + curv)
                for s in range(1, 16):
                    rows = torch.arange(32)[:, None] + torch.round(s * shp).long()[None, :]      # (32,128)
                    ser = torch.gather(zp, 2, rows[None, None].expand(b, 6, 32, 128)).sum(-1) / np.sqrt(128)  # (b,6,32)
                    v = ser.amax(-1); u = v > best; best = torch.where(u, v, best)
                    bestser = torch.where(u[..., None], ser, bestser)
            pk = (bestser > 4) & (bestser >= F.pad(bestser, (1, 0))[..., :-1]) & (bestser >= F.pad(bestser, (0, 1))[..., 1:])
            npk = pk.sum(-1).float()                                            # repeats within a panel
            nm = 'up' if sign > 0 else 'dn'
            for p in range(6): res[f'sw_{nm}_p{p}'] = best[:, p]
            res[f'sw_{nm}_onmax'] = best[:, 0::2].max(1).values; res[f'sw_{nm}_offmax'] = best[:, 1::2].max(1).values
            res[f'sw_{nm}_allmin'] = best.min(1).values; res[f'sw_{nm}_2ndmin'] = best.sort(1).values[:, 1]
            res[f'sw_{nm}_npanel4'] = (best > 4).sum(1).float(); res[f'sw_{nm}_npanel6'] = (best > 6).sum(1).float()
            res[f'sw_{nm}_npk_tot'] = npk.sum(1); res[f'sw_{nm}_npk_max'] = npk.max(1).values
            res[f'sw_{nm}_multi_panels'] = (npk >= 2).sum(1).float()            # sawtooth: >=2 sweeps in a panel
            # spacing regularity of peaks in the strongest panel
            pmax = best.argmax(1); ser = bestser[torch.arange(b), pmax]
            acf = torch.stack([(ser[:, :32 - l] * ser[:, l:]).mean(1) for l in range(2, 12)], 1) / (ser.pow(2).mean(1, keepdim=True) + 1e-3)
            res[f'sw_{nm}_acf_max'] = acf.max(1).values; res[f'sw_{nm}_acf_lag'] = acf.argmax(1).float() + 2
        res['sw_up_minus_dn'] = res['sw_up_onmax'] - res['sw_dn_onmax']
        res['sw_up_minus_dn_all'] = res['sw_up_2ndmin'] - res['sw_dn_2ndmin']
        out.append(pd.DataFrame({k: v.numpy() for k, v in res.items()}))
    return pd.concat(out, ignore_index=True)

if __name__ == '__main__':
    for nm in sys.argv[1:]:
        t = time.time(); Fd = feats(np.load(D + rf'\z_{nm}.npy', mmap_mode='r'))
        Fd.to_csv(D + rf'\fe_swp_{nm}.csv', index=False); print(nm, Fd.shape, round(time.time() - t), 's', flush=True)
