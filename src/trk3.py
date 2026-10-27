"""trk3: SUB-CHANNEL ETI track connection test.
Find the best ON-only track (same objective as track.py), then per ON panel fit the line position to sub-channel
precision (weighted centroid per row in a +-4 ch window around the track, then weighted least squares over the 16 rows),
and measure how far each panel's line sits from ONE global straight track, in channels and in sigmas.
-> fe_trk3_{name}.csv"""
import numpy as np, pandas as pd, torch, torch.nn.functional as F, sys, os, time
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
from phys import smooth_f
from track import track_sums, D2, CREF
torch.set_num_threads(int(os.environ.get('NT', 8)))
W = 384

def feats(Z, md, bs=400):
    gaps_all = md.slew_gap_rows.values; out = []
    for s0 in range(0, len(Z), bs):
        t0 = time.time()
        z = torch.from_numpy(np.asarray(Z[s0:s0 + bs], dtype=np.float32)); b = len(z); g = gaps_all[s0:s0 + bs]
        valid = (z != 0).any(-1)
        zr = (z - z.median(-1, keepdim=True).values) * valid[..., None]
        zn = smooth_f(zr).clamp(-3, 3); zp = F.pad(zn, (128, 128))
        best = torch.full((b,), -1e9); bd = torch.zeros(b); bc = torch.zeros(b)
        for d in D2:
            P = track_sums(zp, g, float(d), CREF, W)
            obj = P[:, 0::2].min(1).values - P[:, 1::2].clamp(min=0).max(1).values
            v, i = obj.max(-1); u = v > best; best[u] = v[u]; bd[u] = float(d); bc[u] = CREF[i[u]].float()
        # sub-channel per-row centroids along the best track, ON panels
        wpos = zr.clamp(min=0)                                                   # unsmoothed, positive part
        tref = torch.from_numpy(2 * (16 + g) + 7.5).float()
        offs = torch.arange(-4, 5).float()
        res = {}
        pan_c, pan_s, pan_e, pan_slope, pan_se = [], [], [], [], []
        for p in (0, 2, 4):
            t = torch.from_numpy(p * (16 + g)).float()[:, None] + torch.arange(16).float()[None, :] - tref[:, None]   # (b,16)
            cpred = bc[:, None] + bd[:, None] * t                                                                   # (b,16)
            chs = torch.round(cpred)[..., None] + offs                                                              # (b,16,9)
            inb = (chs >= 0) & (chs <= 127)
            wv = torch.gather(wpos[:, p], 2, chs.clamp(0, 127).long()) * inb                                        # (b,16,9)
            wsum = wv.sum(-1); cen = (wv * chs).sum(-1) / wsum.clamp(min=1e-3)                                      # (b,16)
            rw = wsum * valid[:, p]                                                                                # row weights
            # weighted LS of centroid vs t within the panel: cen = a + s*(t - tm)
            tm = (rw * t).sum(1) / rw.sum(1).clamp(min=1e-3)
            tt = t - tm[:, None]
            s_ = (rw * tt * (cen - (rw * cen).sum(1, keepdim=True) / rw.sum(1, keepdim=True).clamp(min=1e-3))).sum(1) / (rw * tt ** 2).sum(1).clamp(min=1e-3)
            a_ = (rw * cen).sum(1) / rw.sum(1).clamp(min=1e-3)                                                       # position at tm
            resid = cen - (a_[:, None] + s_[:, None] * tt)
            se = torch.sqrt((rw * resid ** 2).sum(1) / rw.sum(1).clamp(min=1e-3) / torch.clamp((rw > 0).sum(1).float() - 2, min=1)) + 0.05
            pred_at_tm = bc + bd * tm
            pan_c.append(a_ - pred_at_tm); pan_e.append(se); pan_slope.append(s_ - bd); pan_s.append(rw.sum(1))
        dc = torch.stack(pan_c, 1); se = torch.stack(pan_e, 1); dsl = torch.stack(pan_slope, 1); wt = torch.stack(pan_s, 1)
        # connection chi2: offsets of the 3 panel positions after refitting ONE global line through them (2 dof removed -> 1 dof)
        f = {'t3_obj': best, 't3_drift': bd.abs()}
        for k, p in enumerate((0, 2, 4)):
            f[f't3_off_p{p}'] = dc[:, k]; f[f't3_se_p{p}'] = se[:, k]; f[f't3_slopedev_p{p}'] = dsl[:, k]
        f['t3_off_absmax'] = dc.abs().max(1).values; f['t3_off_sig_max'] = (dc.abs() / se).max(1).values
        f['t3_off_rms'] = dc.pow(2).mean(1).sqrt(); f['t3_off_chi2'] = (dc / se).pow(2).sum(1)
        f['t3_slopedev_absmax'] = dsl.abs().max(1).values; f['t3_slope_spread'] = dsl.std(1)
        f['t3_wmin'] = wt.min(1).values
        # within-panel drift of each ON line (independent of the global track): zero-drift trap indicator
        f['t3_inpanel_drift_mean'] = (dsl + bd[:, None]).mean(1).abs(); f['t3_inpanel_drift_min'] = (dsl + bd[:, None]).abs().min(1).values
        out.append(pd.DataFrame({k2: v.numpy() for k2, v in f.items()}))
        if s0 == 0: print('batch', round(time.time() - t0, 1), 's', flush=True)
    return pd.concat(out, ignore_index=True)

if __name__ == '__main__':
    tr, te = meta()
    for nm in sys.argv[1:]:
        md = te if nm.startswith('test') else tr
        Fd = feats(np.load(D + rf'\z_{nm}.npy', mmap_mode='r'), md)
        Fd.to_csv(D + rf'\fe_trk3_{nm}.csv', index=False); print(nm, Fd.shape, flush=True)
