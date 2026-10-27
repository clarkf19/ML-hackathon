"""G: inject ETI tracks / trap tracks into raw train cadences (byte domain, linear power add).
kinds: eti (ON-only connected drifting track), disc (ON-only, same drift, panels offset), zero (ON-only zero drift),
onoff (track also in one OFF panel). Label: eti -> eti_candidate; traps -> higher priority of (base, rfi_intermittent).
usage: inject.py N SEED OUTNAME [ampscale]   -> data/{OUT}_cadences.npy, data/{OUT}_info.csv"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
N, SEED, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
AS = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
rng = np.random.default_rng(SEED)
X = np.load(SRC + r'\train_cadences.npy', mmap_mode='r'); tr = meta()[0]; y = y_train()
kinds = rng.choice(['eti', 'disc', 'zero', 'onoff'], N, p=[0.5, 0.2, 0.15, 0.15])
base = rng.integers(0, len(X), N)
out = np.zeros((N, 6, 16, 128), np.uint8); info = []
ch = np.arange(128)
for n in range(N):
    x = X[base[n]].astype(np.float32); miss = x == 0
    g = tr.slew_gap_rows.values[base[n]]; k = kinds[n]
    P = 2 ** (x / 24 - 4)
    bp = np.median(np.where(miss, np.nan, P), axis=1, keepdims=True); bp = np.nan_to_num(np.nanmedian(bp, 0, keepdims=True), nan=1.0)
    d = rng.uniform(0.04, 0.8) * rng.choice([-1, 1]) if k != 'zero' else 0.0
    A = AS * np.exp(rng.uniform(np.log(0.25), np.log(2.5)))
    sig = rng.uniform(0.5, 1.3)
    tref = 2 * (16 + g) + 7.5
    span = abs(d) * (2 * (16 + g) + 8)
    c0 = rng.uniform(5 + span, 123 - span) if span < 55 else rng.uniform(5, 123)
    panels = [0, 2, 4] + ([int(rng.choice([1, 3, 5]))] if k == 'onoff' else [])
    offs = {p: 0.0 for p in panels}
    if k == 'disc':
        for p in (0, 2, 4): offs[p] = rng.choice([-1, 1]) * rng.uniform(4, 30)
        offs[int(rng.choice([0, 2, 4]))] = 0.0
    for p in panels:
        t = p * (16 + g) + np.arange(16) - tref
        c = c0 + d * t + offs[p]                                      # (16,)
        prof = np.exp(-0.5 * ((ch[None, :] - c[:, None]) / sig) ** 2)
        P[p] += A * prof * bp[0]
    v = np.clip(np.round(24 * (np.log2(P) + 4)), 1, 255); v[miss] = 0
    out[n] = v.astype(np.uint8)
    lab = L2I['eti_candidate'] if k == 'eti' else min(y[base[n]], L2I['rfi_intermittent'])
    info.append((base[n], k, lab, d, A, sig, g))
np.save(D + rf'\{OUT}_cadences.npy', out)
pd.DataFrame(info, columns=['base', 'kind', 'y', 'drift', 'amp', 'sig', 'slew_gap_rows']).to_csv(D + rf'\{OUT}_info.csv', index=False)
print('done', N, pd.Series(kinds).value_counts().to_dict())
