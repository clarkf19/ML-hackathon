"""Panel presence patterns per class from REAL bundle features: which panels carry a narrowband line (phys_best_p > TH),
as 6-bit patterns (ON panels 0,2,4 / OFF 1,3,5), plus per-class rates of drift classes among detected panels.
usage: python src/gen_patterns.py [GEN.npz] [--th=5]"""
import sys, numpy as np, pandas as pd
from pathlib import Path
B = Path(__file__).resolve().parent.parent / 'bundle'
m = np.load(B / 'meta.npz', allow_pickle=True); L, COLS = list(m['labels']), list(m['cols'])
TH = float(next((a.split('=')[1] for a in sys.argv if a.startswith('--th=')), 5))

def report(X, y, name):
    X = pd.DataFrame(X, columns=COLS); print(f'===== {name}')
    bp = np.stack([X[f'phys_best_p{p}'] for p in range(6)], 1) > TH
    bd = np.abs(np.stack([X[f'phys_bestd_p{p}'] for p in range(6)], 1))
    pat = [''.join('x' if b else '.' for b in r) for r in bp]
    for k in range(9):
        s = pd.Series(pat)[y == k].value_counts(normalize=True).head(8).round(3)
        non, noff = bp[y == k][:, 0::2].sum(1), bp[y == k][:, 1::2].sum(1)
        d = bd[y == k][bp[y == k]]
        dq = np.round(np.quantile(d, [.1, .5, .9]), 2).tolist() if len(d) else '-'
        print(f'{L[k]:20s} n_on/off mean {non.mean():.2f}/{noff.mean():.2f}  |drift| of detected q10/50/90 {dq}  frac|d|>1 {np.mean(d > 1) if len(d) else 0:.2f}')
        print('     ', dict(s))

def glitch(X, name):
    X = pd.DataFrame(X, columns=COLS); b = X.phys_bb0_max.values
    pp = np.stack([X[f'phys_bb0_p{p}'] for p in range(6)], 1)
    print(f'{name:8s} bb0|>10 q05..95', np.round(np.quantile(b[b > 10], [.05, .1, .25, .5, .75, .9, .95]), 1).tolist(),
          ' #panels bb0>10:', np.bincount((pp > 10).sum(1), minlength=7).tolist())

if __name__ == '__main__':
    for w in ['clean', 'deg2', 'test']: glitch(np.load(B / f'feats_{w}.npz')['X'], w)
    report(np.load(B / 'feats_clean.npz')['X'], m['y'], 'REAL train')
    for f in [a for a in sys.argv[1:] if not a.startswith('--')]:
        z = np.load(f); report(z['X'], z['y'], f)
