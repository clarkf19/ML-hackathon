"""C: physics features -> data/fe_phys_{train,test}.csv"""
import numpy as np, pandas as pd, torch, torch.nn.functional as F, sys, os, time
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
torch.set_num_threads(int(os.environ.get('NT', 8)))
D1 = np.r_[np.arange(-1, 1.001, 0.125), np.arange(1.5, 8.01, 0.5), -np.arange(1.5, 8.01, 0.5)]
D2 = np.linspace(-1.2, 1.2, 97)
PAD = 70

def smooth_f(z):  # [0.5,1,0.5] along frequency
    return (z + 0.5 * F.pad(z, (1, 0))[..., :-1] + 0.5 * F.pad(z, (0, 1))[..., 1:]) / 2.0

def panel_drift(zn):
    """zn: (b,6,16,128). returns score (b,6,nd) = max_c line SNR, and argmax c."""
    b = zn.shape[0]; zp = F.pad(zn, (PAD, PAD)); r = torch.arange(16) - 7.5
    S = torch.zeros(b, 6, len(D1)); C = torch.zeros(b, 6, len(D1), dtype=torch.long)
    for k, d in enumerate(D1):
        off = torch.round(d * r).long()
        idx = (torch.arange(128)[None, :] + PAD + off[:, None])            # (16,128)
        g = torch.gather(zp, 3, idx[None, None].expand(b, 6, 16, 128)).sum(2) / 4.0
        S[:, :, k], C[:, :, k] = g.max(-1)
    return S, C

def global_track(zn, gaps):
    """ON-track search in global time. For each drift d and channel c at t_ref (centre of panel 2):
    per-panel sums along the straight track. Returns features dict arrays."""
    b = zn.shape[0]; W = 128 + 2 * 128
    zp = F.pad(zn, (128, 128))                                              # (b,6,16,384)
    best_min = torch.full((b,), -1e9); best_diff = torch.full((b,), -1e9)
    rec_min = torch.zeros(b, 8); rec_diff = torch.zeros(b, 8)
    out_all = torch.full((b,), -1e9)
    cref = torch.arange(-64, 192)                                           # 256 candidate positions
    for d in D2:
        P = torch.zeros(b, 6, len(cref))
        for gval in np.unique(gaps):
            m = torch.from_numpy(gaps == gval)
            if not m.any(): continue
            tref = 2 * (16 + gval) + 7.5
            for p in range(6):
                t = p * (16 + gval) + torch.arange(16) - tref
                ch = torch.round(cref[None, :] + d * t[:, None]).long() + 128   # (16,256)
                ch = ch.clamp(0, W - 1)
                g = torch.gather(zp[m, p], 2, ch[None].expand(int(m.sum()), 16, len(cref))).sum(1) / 4.0
                P[m, p] = g
        on, off = P[:, 0::2], P[:, 1::2]
        mn = on.min(1).values                                               # all three ON present
        diff = on.sum(1) / 1.732 - off.clamp(min=0).max(1).values * 1.732
        allp = P.min(1).values
        for score, best, rec in [(mn, best_min, rec_min), (diff, best_diff, rec_diff)]:
            v, i = score.max(-1); upd = v > best
            best[upd] = v[upd]
            sel = P[torch.arange(b), :, i]                                 # (b,6)
            rec[upd] = torch.cat([sel, torch.full((b, 1), float(d)), cref[i][:, None].float()], 1)[upd]
        out_all = torch.maximum(out_all, allp.max(-1).values)
    return best_min, rec_min, best_diff, rec_diff, out_all

def dedisperse(z):
    """broadband dispersed pulses. delay(c) = s*shape(u), u=1-c/127 (low freq later). returns (b,6,ns,na,32) series"""
    b = z.shape[0]; u = 1 - torch.arange(128) / 127.0
    zp = F.pad(z, (0, 0, 16, 16))                                           # time pad -> (b,6,48,128)
    res = {}
    for a in (0.0, 1.5):
        shp = (u + a * u ** 2) / (1 + a)
        for s in range(0, 15):
            dl = torch.round(s * shp).long()                                # (128,)
            rows = torch.arange(32)[:, None] + dl[None, :]                  # ref time -16..15 at top freq
            g = torch.gather(zp, 2, rows[None, None].expand(b, 6, 32, 128)).sum(-1) / np.sqrt(128)
            res[(a, s)] = g
    return res

def feats(Z, meta_df, bs=400):
    gaps = meta_df.slew_gap_rows.values; rows = []
    for s0 in range(0, len(Z), bs):
        t = time.time()
        z = torch.from_numpy(np.asarray(Z[s0:s0 + bs], dtype=np.float32)); b = len(z); g = gaps[s0:s0 + bs]
        valid = (z != 0).any(-1)                                            # (b,6,16)
        rowmed = z.median(-1, keepdim=True).values
        zn = smooth_f((z - rowmed) * valid[..., None])                      # narrowband, broadband removed
        f = {}
        S, C = panel_drift(zn)
        ad = torch.from_numpy(np.abs(D1)).float()
        grp = {'d0': ad <= 0.13, 'dslow': (ad > 0.13) & (ad <= 1.0), 'dmid': (ad > 1.0) & (ad <= 3.0), 'dsteep': ad > 3.0}
        for gname, gm in grp.items():
            sc = S[:, :, gm].max(-1).values                                 # (b,6)
            for p in range(6): f[f'{gname}_p{p}'] = sc[:, p]
            on, off = sc[:, 0::2], sc[:, 1::2]
            f[f'{gname}_onmin'] = on.min(1).values; f[f'{gname}_onmax'] = on.max(1).values
            f[f'{gname}_offmax'] = off.max(1).values; f[f'{gname}_offmin'] = off.min(1).values
            f[f'{gname}_allmin'] = sc.min(1).values; f[f'{gname}_2ndmin'] = sc.sort(1).values[:, 1]
            for th in (3, 5, 8):
                f[f'{gname}_non{th}'] = (on > th).sum(1); f[f'{gname}_noff{th}'] = (off > th).sum(1)
        best = S.max(-1); bd = torch.from_numpy(D1).float()[best.indices]   # best drift per panel
        for p in range(6): f[f'bestd_p{p}'] = bd[:, p]; f[f'best_p{p}'] = best.values[:, p]
        onb = bd[:, 0::2]; f['on_drift_spread'] = onb.std(1); f['on_absdrift_mean'] = onb.abs().mean(1)
        # ON panels best slow-drift detection: positions -> track consistency
        slow = grp['dslow'] | grp['d0']
        Ss = S[:, :, slow]; ki = Ss.argmax(-1); dsl = torch.from_numpy(D1[slow.numpy()]).float()[ki]
        csl = torch.gather(C[:, :, slow], 2, ki[..., None])[..., 0].float()
        tt = torch.stack([torch.from_numpy(p * (16 + g) + 7.5).float() for p in range(6)], 1)
        on_t, on_c, on_d = tt[:, 0::2], csl[:, 0::2], dsl[:, 0::2]
        pred = on_c[:, 0:1] + on_d.mean(1, keepdim=True) * (on_t - on_t[:, 0:1])
        f['on_track_resid'] = (on_c - pred).abs().max(1).values
        A = torch.stack([torch.ones_like(on_t), on_t], -1)
        coef = torch.linalg.lstsq(A, on_c[..., None]).solution
        f['on_fit_resid'] = (A @ coef - on_c[..., None]).abs().max(1).values[:, 0]
        f['on_fit_slope_vs_drift'] = (coef[:, 1, 0] - on_d.mean(1)).abs()
        f['on_drift_mean'] = on_d.mean(1).abs()
        # global ON track
        bm, rm, bdf, rdf, ball = global_track(zn, g)
        f['gt_min'] = bm; f['gt_diff'] = bdf; f['gt_all'] = ball
        for j in range(6): f[f'gtm_p{j}'] = rm[:, j]; f[f'gtd_p{j}'] = rdf[:, j]
        f['gtm_drift'] = rm[:, 6].abs(); f['gtd_drift'] = rdf[:, 6].abs()
        f['gtm_offmax'] = rm[:, [1, 3, 5]].max(1).values; f['gtd_offmax'] = rdf[:, [1, 3, 5]].max(1).values
        f['gtm_onsum'] = rm[:, [0, 2, 4]].sum(1)
        # persistent zero-drift line: time-avg spectrum
        spec = zn.mean(2) * 4.0                                             # (b,6,128) SNR-ish
        srt = spec.sort(1).values
        f['pers_min6'] = srt[:, 0].max(-1).values; f['pers_min5'] = srt[:, 1].max(-1).values; f['pers_min4'] = srt[:, 2].max(-1).values
        f['spec_max'] = spec.amax((1, 2))
        # broadband / dispersion
        zb = z * valid[..., None]
        dd = dedisperse(zb)
        keys = list(dd.keys()); stack = torch.stack([dd[k] for k in keys], 2)   # (b,6,nk,32)
        s_of = torch.tensor([k[1] for k in keys])
        ser0 = stack[:, :, s_of == 0].amax((2, 3)); serD = stack[:, :, s_of >= 2].amax((2, 3))
        for p in range(6): f[f'bb0_p{p}'] = ser0[:, p]; f[f'bbD_p{p}'] = serD[:, p]
        f['bbD_onmax'] = serD[:, 0::2].max(1).values; f['bbD_offmax'] = serD[:, 1::2].max(1).values
        f['bbD_onmin'] = serD[:, 0::2].min(1).values; f['bbD_on2nd'] = serD[:, 0::2].sort(1).values[:, 1]
        f['bb0_max'] = ser0.max(1).values; f['bbD_minus_bb0'] = serD.max(1).values - ser0.max(1).values
        # best dispersion per cadence: pulse count & DM consistency
        bestk = stack[:, 0::2].amax((1, 3)).argmax(-1)                       # (b,)
        ser = stack[torch.arange(b), :, bestk]                              # (b,6,32)
        f['best_s'] = s_of[bestk].float()
        for th in (4, 6):
            pk = (ser > th) & (ser >= F.pad(ser, (1, 0))[..., :-1]) & (ser >= F.pad(ser, (0, 1))[..., 1:])
            f[f'npulse_on{th}'] = pk[:, 0::2].sum((1, 2)); f[f'npulse_off{th}'] = pk[:, 1::2].sum((1, 2))
            f[f'npanel_on_pulse{th}'] = pk[:, 0::2].any(-1).sum(1)
        sD = stack[:, 0::2].amax(3)                                         # (b,3,nk)
        f['dm_consist'] = (s_of[sD.argmax(-1)].float().std(1))
        # scintillation texture: smoothed power
        k = torch.ones(1, 1, 4, 16) / 64
        sm = F.conv2d(zb.reshape(b * 6, 1, 16, 128), k, stride=(2, 4)).reshape(b, 6, -1) * 8.0
        f_s = sm.std(-1); f_m = sm.amax(-1); f_pos = (sm > 2).float().mean(-1)
        for p in range(6): f[f'tex_std_p{p}'] = f_s[:, p]; f[f'tex_max_p{p}'] = f_m[:, p]
        f['tex_std_on_off'] = f_s[:, 0::2].mean(1) - f_s[:, 1::2].mean(1)
        f['tex_pos_on_off'] = f_pos[:, 0::2].mean(1) - f_pos[:, 1::2].mean(1)
        f['tex_std_onmin'] = f_s[:, 0::2].min(1).values; f['tex_std_offmax'] = f_s[:, 1::2].max(1).values
        f['mean_on_off'] = zb[:, 0::2].mean((1, 2, 3)) - zb[:, 1::2].mean((1, 2, 3))
        f['n_missing_rows'] = (~valid).sum((1, 2))
        f['gap'] = torch.from_numpy(g).float()
        rows.append(pd.DataFrame({k2: (v.numpy() if torch.is_tensor(v) else v) for k2, v in f.items()}))
        if s0 == 0: print('batch time', round(time.time() - t, 1), 'nfeat', len(f), flush=True)
    return pd.concat(rows, ignore_index=True)

if __name__ == '__main__':
    tr, te = meta()
    n = None
    for nm in (sys.argv[1:] or ['train', 'test']):
        md = te if nm.startswith('test') else (pd.read_csv(D + rf'\{nm}_info.csv') if nm.startswith('inj') else tr)
        Z = np.load(D + rf'\z_{nm}.npy', mmap_mode='r')
        if n: Z = Z[:n]; md = md.iloc[:n]
        Fd = feats(Z, md)
        Fd.to_csv(D + rf'\fe_phys_{nm}{"_smoke" if n else ""}.csv', index=False); print(nm, Fd.shape, flush=True)
