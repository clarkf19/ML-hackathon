"""In-band-aware global tracks through ALL 6 panels (persistent drifting RFI that exits the band stays 'persistent').
For each straight global track: per-panel sum and in-band flag; present = sum>3. -> fe_trk2_{name}.csv"""
import numpy as np, pandas as pd, torch, torch.nn.functional as F, sys, os, time
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
from phys import smooth_f
torch.set_num_threads(int(os.environ.get('NT', 6)))
D2 = np.linspace(-1.2, 1.2, 97)
CREF = torch.arange(-96, 224)

def feats(Z, md, bs=400):
    gaps_all = md.slew_gap_rows.values; out = []
    for s0 in range(0, len(Z), bs):
        z = torch.from_numpy(np.asarray(Z[s0:s0 + bs], dtype=np.float32)); b = len(z); g = gaps_all[s0:s0 + bs]
        valid = (z != 0).any(-1)
        zn = smooth_f((z - z.median(-1, keepdim=True).values) * valid[..., None]).clamp(-3, 6)
        W = 384; zp = F.pad(zn, (128, 128))
        best = {k: torch.full((b,), -1e9) for k in ['inmin', 'in2nd', 'npres', 'absent_in']}
        rec = torch.zeros(b, 4)
        for d in D2:
            P = torch.zeros(b, 6, len(CREF)); IB = torch.zeros(b, 6, len(CREF))
            for gval in np.unique(g):
                m = torch.from_numpy(g == gval); n = int(m.sum()); tref = 2 * (16 + gval) + 7.5
                for p in range(6):
                    t = p * (16 + gval) + torch.arange(16) - tref
                    chf = torch.round(CREF[None, :] + float(d) * t[:, None])                     # (16,ncref)
                    inb = ((chf >= 0) & (chf <= 127)).float().mean(0)                           # (ncref,)
                    ch = (chf.long() + 128).clamp(0, W - 1)
                    P[m, p] = torch.gather(zp[m, p], 2, ch[None].expand(n, 16, len(CREF))).sum(1) / 4.0
                    IB[m, p] = inb[None].expand(n, -1)
            inb = IB > 0.6; ninb = inb.sum(1)                                                   # (b,ncref)
            Pin = torch.where(inb, P, torch.tensor(1e9)); srt = Pin.sort(1).values
            inmin = torch.where(ninb >= 3, srt[:, 0], torch.tensor(-1e9))
            in2nd = torch.where(ninb >= 3, srt[:, 1], torch.tensor(-1e9))
            pres = (P > 3) & inb; npres = pres.sum(1).float()
            absent = (inb & ~(P > 3)).sum(1).float()
            # score tracks by strength of 2nd-weakest in-band panel (persistent w/ one fade)
            v, i = in2nd.max(-1); upd = v > best['in2nd']
            best['in2nd'][upd] = v[upd]
            ar = torch.arange(b)
            cand = torch.stack([inmin[ar, i], npres[ar, i], absent[ar, i], ninb[ar, i].float()], 1)
            rec[upd] = cand[upd]
            best['inmin'] = torch.maximum(best['inmin'], inmin.max(-1).values)
            # strongest single track (by sum of present in-band panels): how many in-band panels miss it?
            strength = torch.where(pres, P, torch.tensor(0.0)).sum(1)
            v2, i2 = strength.max(-1); upd2 = v2 > best['npres']
            best['npres'][upd2] = v2[upd2]
            if not hasattr(feats, '_r2'): pass
            best['absent_in'][upd2] = torch.where(upd2, absent[ar, i2] * 100 + npres[ar, i2], best['absent_in'])[upd2]
        f = {'g2_inmin': best['inmin'], 'g2_in2nd': best['in2nd'], 'g2_rec_inmin': rec[:, 0], 'g2_rec_npres': rec[:, 1],
             'g2_rec_absent': rec[:, 2], 'g2_rec_ninb': rec[:, 3], 'g2_strong_sum': best['npres'],
             'g2_strong_absent': torch.div(best['absent_in'], 100, rounding_mode='floor'), 'g2_strong_npres': best['absent_in'] % 100}
        out.append(pd.DataFrame({k: v.numpy() for k, v in f.items()}))
    return pd.concat(out, ignore_index=True)

if __name__ == '__main__':
    tr, te = meta()
    for nm in sys.argv[1:]:
        md = te if nm.startswith('test') else tr
        t = time.time(); Fd = feats(np.load(D + rf'\z_{nm}.npy', mmap_mode='r'), md)
        Fd.to_csv(D + rf'\fe_trk2_{nm}.csv', index=False); print(nm, Fd.shape, round(time.time() - t), 's', flush=True)
