"""C2: refined ETI track features -> fe_trk_{name}.csv.
Objective: min(ON track sums) - max(OFF track sums, 0) on clipped z (bright crossing lines can't dominate).
Then for the best track: per-panel offset scan (+-8 ch) -> connection / disconnection; OFF energy near track;
same-drift line anywhere in OFF / ON; second-best separated track."""
import numpy as np, pandas as pd, torch, torch.nn.functional as F, sys, os, time
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
from phys import smooth_f
torch.set_num_threads(int(os.environ.get('NT', 8)))
D2 = np.linspace(-1.2, 1.2, 193)
CREF = torch.arange(-96, 224)
OFFS = torch.arange(-8, 9)

def track_sums(zp, gaps, d, cref, W):
    """zp: (b,6,16,W) padded by 128 left. returns (b,6,len(cref))"""
    b = zp.shape[0]; P = torch.zeros(b, 6, len(cref))
    for gval in np.unique(gaps):
        m = torch.from_numpy(gaps == gval); n = int(m.sum())
        tref = 2 * (16 + gval) + 7.5
        for p in range(6):
            t = p * (16 + gval) + torch.arange(16) - tref
            ch = (torch.round(cref[None, :] + d * t[:, None]).long() + 128).clamp(0, W - 1)
            P[m, p] = torch.gather(zp[m, p], 2, ch[None].expand(n, 16, len(cref))).sum(1) / 4.0
    return P

def feats(Z, md, bs=400):
    gaps_all = md.slew_gap_rows.values; out = []
    for s0 in range(0, len(Z), bs):
        t0 = time.time()
        z = torch.from_numpy(np.asarray(Z[s0:s0 + bs], dtype=np.float32)); b = len(z); g = gaps_all[s0:s0 + bs]
        valid = (z != 0).any(-1)
        zn = smooth_f((z - z.median(-1, keepdim=True).values) * valid[..., None]).clamp(-3, 3)
        W = 128 + 256; zp = F.pad(zn, (128, 128))
        best = torch.full((b,), -1e9); bd = torch.zeros(b); bc = torch.zeros(b, dtype=torch.long)
        allbest = []
        for d in D2:
            P = track_sums(zp, g, float(d), CREF, W)
            obj = P[:, 0::2].min(1).values - P[:, 1::2].clamp(min=0).max(1).values     # (b,ncref)
            v, i = obj.max(-1); allbest.append(obj)
            u = v > best; best[u] = v[u]; bd[u] = float(d); bc[u] = i[u]
        allobj = torch.stack(allbest, 1)                                              # (b,nd,ncref)
        f = {'t_obj': best.numpy(), 't_drift': bd.abs().numpy()}
        # second-best track away from the best (|dc|>6 at ref time)
        cgrid = CREF[None, None, :].expand_as(allobj)
        far = (cgrid - CREF[bc][:, None, None]).abs() > 6
        f['t_obj2'] = torch.where(far, allobj, torch.tensor(-1e9)).amax((1, 2)).numpy()
        # per-panel offset scan around best track
        pan = torch.zeros(b, 6, len(OFFS))
        for j, o in enumerate(OFFS):
            for gval in np.unique(g):
                m = torch.from_numpy(g == gval); n = int(m.sum()); tref = 2 * (16 + gval) + 7.5
                for p in range(6):
                    t = p * (16 + gval) + torch.arange(16) - tref
                    ch = (torch.round(CREF[bc[m]][:, None] + o + bd[m][:, None] * t[None, :]).long() + 128).clamp(0, W - 1)  # (n,16)
                    pan[m, p, j] = torch.gather(zp[m, p], 2, ch[:, :, None]).squeeze(-1).sum(1) / 4.0
        c0 = len(OFFS) // 2
        at0 = pan[:, :, c0]; bo = OFFS[pan.argmax(-1)].float(); bv = pan.max(-1).values
        for p in range(6): f[f't_p{p}'] = at0[:, p].numpy(); f[f't_off{p}'] = bo[:, p].numpy(); f[f't_best{p}'] = bv[:, p].numpy()
        f['t_on_offset_absmax'] = bo[:, 0::2].abs().max(1).values.numpy()
        f['t_on_offset_spread'] = bo[:, 0::2].std(1).numpy()
        f['t_on_gain_from_shift'] = (bv[:, 0::2] - at0[:, 0::2]).max(1).values.numpy()
        f['t_off_near_max'] = bv[:, 1::2].max(1).values.numpy()
        f['t_on_min'] = at0[:, 0::2].min(1).values.numpy(); f['t_off_max0'] = at0[:, 1::2].max(1).values.numpy()
        # same-drift line anywhere in each panel (drift = best track drift)
        r = torch.arange(16) - 7.5
        anyd = torch.zeros(b, 6)
        zp2 = F.pad(zn, (70, 70))
        for i in range(b):
            off = torch.round(bd[i] * r).long()
            idx = torch.arange(128)[None, :] + 70 + off[:, None]
            anyd[i] = torch.gather(zp2[i], 2, idx[None].expand(6, 16, 128)).sum(1).max(-1).values / 4.0
        for p in range(6): f[f't_samed_p{p}'] = anyd[:, p].numpy()
        f['t_samed_offmax'] = anyd[:, 1::2].max(1).values.numpy()
        f['t_samed_on_minus_track'] = (anyd[:, 0::2] - at0[:, 0::2]).max(1).values.numpy()
        out.append(pd.DataFrame(f))
        if s0 == 0: print('batch', round(time.time() - t0, 1), 's', flush=True)
    return pd.concat(out, ignore_index=True)

if __name__ == '__main__':
    tr, te = meta()
    for nm in sys.argv[1:]:
        md = te if nm.startswith('test') else (pd.read_csv(D + rf'\{nm}_info.csv') if nm.startswith('inj') else tr)
        Fd = feats(np.load(D + rf'\z_{nm}.npy', mmap_mode='r'), md)
        o = {'train': 'trk_train', 'test': 'trk_test', 'train_deg': 'trkd_train', 'train_deg2': 'trkd_deg2', 'train_deg4': 'trkd_deg4', 'train_deg6': 'trkd_deg6', 'train_deg8': 'trkd_deg8', 'train_deg10': 'trkd_deg10', 'train_deg12': 'trkd_deg12', 'train_deg14': 'trkd_deg14', 'train_deg16': 'trkd_deg16'}.get(nm, f'trk_{nm}')
        Fd.to_csv(D + rf'\fe_{o}.csv', index=False); print(nm, Fd.shape, flush=True)
