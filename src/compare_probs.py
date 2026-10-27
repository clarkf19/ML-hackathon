"""Paired comparison of two bundle_eval probability files: macro-F1 clean / fake-PKS, per-class F1 deltas, paired bootstrap
(resample cadences, 1000x) of the macro-F1 difference.  usage: python src/compare_probs.py results/probs_A.npz results/probs_B.npz"""
import numpy as np, sys
from pathlib import Path
from sklearn.metrics import f1_score
ROOT = Path(__file__).resolve().parent.parent
m = np.load(ROOT / 'bundle/meta.npz', allow_pickle=True); y = m['y']
S = ['eti', 'frb', 'puls', 'scint', 'sat', 'radar', 'rint', 'rpers', 'noise']
A, B = np.load(sys.argv[1]), np.load(sys.argv[2]); rng = np.random.default_rng(0)
for k, nm in [('oofc', 'clean'), ('oofd', 'fakePKS')]:
    a, b = A[k].argmax(1), B[k].argmax(1)
    fa, fb = f1_score(y, a, average=None), f1_score(y, b, average=None)
    d = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y)); d.append(f1_score(y[i], b[i], average='macro') - f1_score(y[i], a[i], average='macro'))
    d = np.array(d)
    print(f'{nm:8s} {fa.mean():.4f} -> {fb.mean():.4f} ({fb.mean() - fa.mean():+.4f})  bootstrap 90% CI [{np.percentile(d, 5):+.4f}, {np.percentile(d, 95):+.4f}]  P(gain>0) {np.mean(d > 0):.2f}')
    print('   per-class delta', ' '.join(f'{s}:{x:+.3f}' for s, x in zip(S, fb - fa)))
ta, tb = A['pte'].argmax(1), B['pte'].argmax(1)
print('test argmax agreement', (ta == tb).mean().round(4), ' test mix B', np.round(np.bincount(tb, minlength=9) / len(tb), 3))
