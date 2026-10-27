"""rep: how many panels contain the SAME line (same drift, same channel +-2) -> persistent vs intermittent.
acf: per-panel 2D autocorrelation summaries (radar sawtooth periodicity, scintillation blobs, drifting lines).
-> fe_rep_{name}.csv"""
import numpy as np, pandas as pd, torch, torch.nn.functional as F, sys, os, time
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
from phys import smooth_f, PAD
torch.set_num_threads(int(os.environ.get('NT', 8)))
DR = np.r_[np.arange(-1, 1.001, 0.125), np.arange(1.5, 8.01, 0.5), -np.arange(1.5, 8.01, 0.5)]

def feats(Z, bs=500):
    out = []
    for s0 in range(0, len(Z), bs):
        z = torch.from_numpy(np.asarray(Z[s0:s0 + bs], dtype=np.float32)); b = len(z)
        valid = (z != 0).any(-1)
        zn = smooth_f((z - z.median(-1, keepdim=True).values) * valid[..., None]).clamp(-3, 6)
        zp = F.pad(zn, (PAD, PAD)); r = torch.arange(16) - 7.5
        f = {}
        kth = torch.full((b, 6), -1e9); kth_on = torch.full((b, 3), -1e9); kth_off = torch.full((b, 3), -1e9)
        steep_sum = torch.zeros(b); slow_sum = torch.zeros(b)
        for d in DR:
            off = torch.round(float(d) * r).long()
            idx = torch.arange(128)[None, :] + PAD + off[:, None]
            L = torch.gather(zp, 3, idx[None, None].expand(b, 6, 16, 128)).sum(2) / 4.0     # (b,6,128)
            Lm = F.max_pool1d(L.reshape(b * 6, 1, 128), 5, 1, 2).reshape(b, 6, 128)          # +-2 ch tolerance
            srt = Lm.sort(1, descending=True).values                                          # (b,6,128)
            kth = torch.maximum(kth, srt.max(-1).values)
            kth_on = torch.maximum(kth_on, Lm[:, 0::2].sort(1, descending=True).values.max(-1).values)
            kth_off = torch.maximum(kth_off, Lm[:, 1::2].sort(1, descending=True).values.max(-1).values)
            pm = L.max(-1).values.sum(1)
            if abs(d) > 2: steep_sum = torch.maximum(steep_sum, pm)
            else: slow_sum = torch.maximum(slow_sum, pm)
        for k in range(6): f[f'rep_k{k + 1}'] = kth[:, k].numpy()
        for k in range(3): f[f'rep_on_k{k + 1}'] = kth_on[:, k].numpy(); f[f'rep_off_k{k + 1}'] = kth_off[:, k].numpy()
        f['steep_sum6'] = steep_sum.numpy(); f['slow_sum6'] = slow_sum.numpy()
        # 2D autocorrelation per panel (masked-row zeros ok), averaged ON / OFF
        x = (z * valid[..., None]).clamp(-4, 8)
        x = x - x.mean((2, 3), keepdim=True)
        Fx = torch.fft.rfft2(F.pad(x, (0, 128, 0, 16)))
        ac = torch.fft.irfft2(Fx * Fx.conj(), s=(32, 256))                                  # (b,6,32,256)
        ac = ac / ac[..., :1, :1].clamp(min=1e-3)
        for nm, sl in [('on', slice(0, 6, 2)), ('off', slice(1, 6, 2))]:
            a = ac[:, sl].mean(1)
            for dt in (1, 2, 3, 4, 6, 8):
                row = torch.cat([a[:, dt, 252:], a[:, dt, :5]], -1)                          # dc -4..4
                wide = torch.cat([a[:, dt, 236:252], a[:, dt, 5:21]], -1)                    # 4<|dc|<=20
                f[f'acf_{nm}_t{dt}_c0'] = a[:, dt, 0].numpy(); f[f'acf_{nm}_t{dt}_near'] = row.max(-1).values.numpy()
                f[f'acf_{nm}_t{dt}_wide'] = wide.max(-1).values.numpy()
            f[f'acf_{nm}_t0_c1'] = a[:, 0, 1].numpy(); f[f'acf_{nm}_t0_c4'] = a[:, 0, 4].numpy(); f[f'acf_{nm}_t0_c12'] = a[:, 0, 12].numpy()
            f[f'acf_{nm}_t1_c1'] = a[:, 1, 1].numpy(); f[f'acf_{nm}_t2_c4'] = a[:, 2, 4].numpy()
        out.append(pd.DataFrame(f))
    return pd.concat(out, ignore_index=True)

if __name__ == '__main__':
    for nm in sys.argv[1:]:
        t = time.time(); Fd = feats(np.load(D + rf'\z_{nm}.npy', mmap_mode='r'))
        o = {'train': 'rep_train', 'test': 'rep_test', 'train_deg': 'repd_train'}.get(nm, f'rep_{nm}')
        Fd.to_csv(D + rf'\fe_{o}.csv', index=False); print(nm, Fd.shape, round(time.time() - t), 's', flush=True)
