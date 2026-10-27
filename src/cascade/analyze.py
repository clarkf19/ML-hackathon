"""Where does a decision-layer variant gain or lose vs the harness baseline? Per-class F1 deltas and the confusion cells that
changed most (fakePKS-val and clean-val), from results/cache/.  usage: python src/cascade/analyze.py "<variant>" ["<reference>"]"""
import sys, re, numpy as np
from common import y, K, R, SHORT, conf, f1_per


def lab(name, w):
    return np.load(R / 'cache' / (re.sub(r'[^A-Za-z0-9_.+-]+', '_', name) + '.npz'))[f'lab_{w}']


v = sys.argv[1]; ref = sys.argv[2] if len(sys.argv) > 2 else 'H argmax (harness baseline)'
for w, nm in [('d', 'fakePKS-val'), ('c', 'clean-val')]:
    a, b = lab(ref, w), lab(v, w)
    d = f1_per(y, b) - f1_per(y, a)
    print(f'{nm}: per-class F1 delta', dict(zip(SHORT, np.round(d, 4))))
    dc = conf(y, b) - conf(y, a); np.fill_diagonal(dc, 0)
    cells = sorted(((dc[i, j], SHORT[i], SHORT[j]) for i in range(K) for j in range(K) if dc[i, j]), key=lambda t: -abs(t[0]))
    print(f'  biggest changes in errors (true -> pred: change in count):', ', '.join(f'{t}->{p}: {c:+d}' for c, t, p in cells[:10]))
    print(f'  rows changed {int((a != b).sum())}, fixed {int(((a != y) & (b == y)).sum())}, broken {int(((a == y) & (b != y)).sum())}')
