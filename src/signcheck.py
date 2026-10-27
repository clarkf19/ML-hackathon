"""Is drift SIGN informative? Signed drift of the best ON-only global track (track.py objective) and of the strongest
per-panel line, by class; plus dispersion direction for broadband sweeps."""
import sys, os; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
import torch, torch.nn.functional as F
from phys import smooth_f, panel_drift, D1
from track import track_sums, D2, CREF
torch.set_num_threads(8)
tr, te = meta(); y = y_train()
def signed(Z, g, bs=400):
    out = []
    for s0 in range(0, len(Z), bs):
        z = torch.from_numpy(np.asarray(Z[s0:s0 + bs], dtype=np.float32)); b = len(z); gg = g[s0:s0 + bs]
        valid = (z != 0).any(-1); zn = smooth_f((z - z.median(-1, keepdim=True).values) * valid[..., None]).clamp(-3, 3)
        zp = F.pad(zn, (128, 128)); best = torch.full((b,), -1e9); bd = torch.zeros(b)
        for d in D2:
            P = track_sums(zp, gg, float(d), CREF, 384); obj = P[:, 0::2].min(1).values - P[:, 1::2].clamp(min=0).max(1).values
            v, _ = obj.max(-1); u = v > best; best[u] = v[u]; bd[u] = float(d)
        S, _ = panel_drift(zn); k = S.reshape(b, -1).argmax(-1) % len(D1)          # strongest line anywhere (any panel)
        out.append(np.c_[bd.numpy(), best.numpy(), D1[k.numpy()], S.reshape(b, -1).max(-1).values.numpy()])
    return np.concatenate(out)
idx = np.where(np.isin(y, [0, 4, 5, 6, 7]))[0]
R = signed(np.load(D + r'\z_train.npy', mmap_mode='r')[np.sort(idx)], tr.slew_gap_rows.values[np.sort(idx)]); yy = y[np.sort(idx)]
for c in [0, 6, 7, 4, 5]:
    m = yy == c
    strong = m & (R[:, 1] > 3) if c in (0, 6) else m & (R[:, 3] > 6)
    col = 0 if c in (0, 6) else 2
    d = R[strong, col]; nz = d[np.abs(d) > 0.02]
    print(f'{LABELS[c]:16s} n={strong.sum():5d}  {"track" if col == 0 else "line "} drift: frac negative {np.mean(nz < 0):.3f}  frac positive {np.mean(nz > 0):.3f}  zero-ish {np.mean(np.abs(d) <= 0.02):.3f}  median {np.median(d):+.3f}')
