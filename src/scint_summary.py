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
base = stats(R / 'results/probs_baseline.npz'); pb = np.load(R / 'results/probs_baseline.npz')['oofd'].argmax(1)
rng = np.random.default_rng(0); BOOT = [rng.integers(0, len(y), len(y)) for _ in range(200)]
def ci(p):  # paired bootstrap 95% interval of the fakePKS macro-F1 difference vs baseline
    d = [f1_score(y[i], p[i], average='macro') - f1_score(y[i], pb[i], average='macro') for i in BOOT]
    return np.percentile(d, [2.5, 97.5])
print('| run | clean-val | Δ | fakePKS-val | Δ (95% paired bootstrap) | scint F1 clean | scint F1 fPKS | Δ | scint rec / prec fPKS |')
print('|---|---|---|---|---|---|---|---|---|')
for fn in sorted(R.glob('results/probs_*.npz'), key=lambda f: (f.stem != 'probs_baseline', f.stat().st_mtime)):
    s = stats(fn); c, d = s['oofc'], s['oofd']; lo, hi = ci(np.load(fn)['oofd'].argmax(1))
    print(f'| {fn.stem[6:]} | {c[0]:.4f} | {c[0] - base["oofc"][0]:+.4f} | {d[0]:.4f} | {d[0] - base["oofd"][0]:+.4f} ({lo:+.4f}…{hi:+.4f}) '
          f'| {c[1]:.3f} | {d[1]:.3f} | {d[1] - base["oofd"][1]:+.3f} | {d[2]:.3f} / {d[3]:.3f} |', flush=True)
