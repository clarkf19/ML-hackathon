"""Second-source features: mask the top-2 repeated (persistent-pattern) lines, then recompute line stats on the rest.
-> fe_rep2_{name}.csv"""
import numpy as np, pandas as pd, torch, torch.nn.functional as F, sys, os, time
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
from phys import smooth_f, PAD
torch.set_num_threads(int(os.environ.get('NT', 8)))
DR = np.r_[np.arange(-1, 1.001, 0.125), np.arange(1.5, 8.01, 0.5), -np.arange(1.5, 8.01, 0.5)]
R = torch.arange(16) - 7.5

def linemaps(zn):
    b = zn.shape[0]; zp = F.pad(zn, (PAD, PAD)); Ls = []
    for d in DR:
        off = torch.round(float(d) * R).long()
        idx = torch.arange(128)[None, :] + PAD + off[:, None]
        Ls.append(torch.gather(zp, 3, idx[None, None].expand(b, 6, 16, 128)).sum(2) / 4.0)
    return torch.stack(Ls, 2)                                                  # (b,6,nd,128)

def stats(L, pre):
    b = L.shape[0]; f = {}
    Lm = F.max_pool1d(L.reshape(-1, 1, 128), 5, 1, 2).reshape(L.shape)
    srt = Lm.sort(1, descending=True).values.amax((2, 3))                      # (b,6) k-th largest panel at best (d,c)
    for k in range(6): f[f'{pre}k{k + 1}'] = srt[:, k]
    pm = L.amax((2, 3))                                                        # per-panel best line anywhere
    for p in range(6): f[f'{pre}p{p}'] = pm[:, p]
    for th in (4, 6):
        f[f'{pre}non{th}'] = (pm[:, 0::2] > th).sum(1).float(); f[f'{pre}noff{th}'] = (pm[:, 1::2] > th).sum(1).float()
    on = Lm[:, 0::2].sort(1, descending=True).values.amax((2, 3)); off = Lm[:, 1::2].sort(1, descending=True).values.amax((2, 3))
    for k in range(3): f[f'{pre}on_k{k + 1}'] = on[:, k]; f[f'{pre}off_k{k + 1}'] = off[:, k]
    return f, Lm

def mask_best(zn, Lm):
    b = zn.shape[0]
    k5 = Lm.sort(1, descending=True).values[:, 4]                              # (b,nd,128) 5th best panel
    v, i = k5.reshape(b, -1).max(-1); di = i // 128; ci = i % 128
    d = torch.from_numpy(DR).float()[di]
    ch = ci[:, None].float() + torch.round(d[:, None] * R[None, :])            # (b,16)
    m = torch.zeros(b, 16, 128, dtype=torch.bool)
    for o in range(-3, 4):
        c = (ch + o).long().clamp(0, 127)
        m.scatter_(2, c[..., None], True)
    zn = zn.masked_fill(m[:, None], 0.0)
    return zn, v

def feats(Z, bs=400):
    out = []
    for s0 in range(0, len(Z), bs):
        z = torch.from_numpy(np.asarray(Z[s0:s0 + bs], dtype=np.float32))
        valid = (z != 0).any(-1)
        zn = smooth_f((z - z.median(-1, keepdim=True).values) * valid[..., None]).clamp(-3, 6)
        f = {}
        L = linemaps(zn); _, Lm = stats(L, 'a_')
        zn, v1 = mask_best(zn, Lm); f['m1_k5'] = v1
        L = linemaps(zn); s1, Lm = stats(L, 'b_'); f.update(s1)
        zn, v2 = mask_best(zn, Lm); f['m2_k5'] = v2
        L = linemaps(zn); s2, _ = stats(L, 'c_'); f.update(s2)
        out.append(pd.DataFrame({k: v.numpy() for k, v in f.items()}))
    return pd.concat(out, ignore_index=True)

if __name__ == '__main__':
    for nm in sys.argv[1:]:
        t = time.time(); Fd = feats(np.load(D + rf'\z_{nm}.npy', mmap_mode='r'))
        o = {'train': 'rep2_train', 'test': 'rep2_test', 'train_deg': 'rep2d_train'}.get(nm, f'rep2_{nm}')
        Fd.to_csv(D + rf'\fe_{o}.csv', index=False); print(nm, Fd.shape, round(time.time() - t), 's', flush=True)
