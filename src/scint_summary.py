"""Markdown table of every results/probs_*.npz: clean / fake-PKS macro-F1, scint F1 / recall / precision, deltas vs the
harness baseline (results/probs_baseline.npz).  usage: python src/scint_summary.py"""
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
R = Path(__file__).resolve().parent.parent
m = np.load(R / 'bundle/meta.npz', allow_pickle=True); y = m['y']; S = 3
def stats(fn):
    z = np.load(fn); o = {}
    for nm in ('oofc', 'oofd'):
        p = z[nm].argmax(1)
        o[nm] = (f1_score(y, p, average='macro'), f1_score(y == S, p == S), recall_score(y == S, p == S), precision_score(y == S, p == S))
    return o
base = stats(R / 'results/probs_baseline.npz')
print('| run | clean-val | Δ | fakePKS-val | Δ | scint F1 clean | scint F1 fPKS | Δ | scint rec / prec fPKS |')
print('|---|---|---|---|---|---|---|---|---|')
for fn in sorted(R.glob('results/probs_*.npz'), key=lambda f: (f.stem != 'probs_baseline', f.stat().st_mtime)):
    s = stats(fn); c, d = s['oofc'], s['oofd']
    print(f'| {fn.stem[6:]} | {c[0]:.4f} | {c[0] - base["oofc"][0]:+.4f} | {d[0]:.4f} | {d[0] - base["oofd"][0]:+.4f} | {c[1]:.3f} | {d[1]:.3f} '
          f'| {d[1] - base["oofd"][1]:+.3f} | {d[2]:.3f} / {d[3]:.3f} |')
