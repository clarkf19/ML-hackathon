"""Write results/probs_best.npz for one variant of decide.py (from results/cache/): oofc / oofd = nested OOF probabilities on
clean-val / fakePKS-val, pte = test probabilities with the decision layer tuned on all 5 folds. The decision layer is folded
into the probabilities so that argmax(probs) == the variant's labels (rule-based variants: 0.5 * (probs + one-hot label)).
Also stores lab_{c,d,t} (labels) and p_{c,d,t} (the variant's probabilities before folding in a rule).
usage: python src/cascade/save_best.py "<variant name>" [--csv submissions/<file>.csv]"""
import sys, re, numpy as np
from common import y, K, R, f1m, _m

name = sys.argv[1]
z = np.load(R / 'cache' / (re.sub(r'[^A-Za-z0-9_.+-]+', '_', name) + '.npz'))
out = {}
for w, key in zip('cdt', ['oofc', 'oofd', 'pte']):
    lab, p = z[f'lab_{w}'], z[f'p_{w}']
    if not p.any(): p = np.full_like(p, 1.0 / K)
    p = p / p.sum(1, keepdims=True)
    q = p if (p.argmax(1) == lab).all() else 0.5 * (p + np.eye(K)[lab])
    assert (q.argmax(1) == lab).all()
    out[key] = q; out[f'lab_{w}'] = lab; out[f'p_{w}'] = p
np.savez_compressed(R / 'probs_best.npz', **out, variant=np.array(name))
print(f'{name}: clean-val {f1m(y, out["oofc"].argmax(1)):.4f}  fakePKS-val {f1m(y, out["oofd"].argmax(1)):.4f}  '
      f'test mix {np.round(np.bincount(out["lab_t"], minlength=K) / len(out["lab_t"]), 3)}')
if '--csv' in sys.argv:
    import pandas as pd
    f = sys.argv[sys.argv.index('--csv') + 1]
    pd.DataFrame({'id': _m['test_ids'], 'label': np.array([str(s) for s in _m['labels']])[out['lab_t']]}).to_csv(f, index=False)
    print('wrote', f)
