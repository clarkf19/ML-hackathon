"""E3: decision-level scint fixes on a probability file, honest (nested over the 5 folds: tune on 4 folds' fake-PKS OOF,
apply to the 5th; the same biases are applied to clean val).
  scint-bias : one log-prob bias on scintillating_star only
  all-bias   : coordinate ascent over all 9 class biases (portable copy of src/bias_tune.py)
usage: python src/scint_bias.py results/probs_baseline.npz [...]"""
import sys, numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
R = Path(__file__).resolve().parent.parent
m = np.load(R / 'bundle/meta.npz', allow_pickle=True); y, fo = m['y'], m['folds']; S = 3
mf = lambda yy, p: f1_score(yy, p, average='macro')
GRID = np.linspace(-1.5, 1.5, 31)

def tune(Lg, yy, classes):
    b = np.zeros(9)
    for it in range(3 if len(classes) > 1 else 1):
        for c in classes:
            best = (mf(yy, (Lg + b).argmax(1)), b[c])
            for v in GRID:
                bb = b.copy(); bb[c] = v; s = mf(yy, (Lg + bb).argmax(1))
                if s > best[0] + 1e-6: best = (s, v)
            b[c] = best[1]
    return b

for fn in sys.argv[1:]:
    z = np.load(fn); Pd = np.log(z['oofd'] + 1e-9); Pc = np.log(z['oofc'] + 1e-9)
    print(f'== {Path(fn).stem}: argmax clean {mf(y, Pc.argmax(1)):.4f} fakePKS {mf(y, Pd.argmax(1)):.4f}'
          f'  scint F1 clean {f1_score(y == S, Pc.argmax(1) == S):.3f} fPKS {f1_score(y == S, Pd.argmax(1) == S):.3f}')
    for nm, cl in [('scint-bias', [S]), ('all-bias', list(range(9)))]:
        pd_, pc_ = np.zeros(len(y), int), np.zeros(len(y), int); Bs = []
        for k in range(5):
            t, v = fo != k, fo == k
            b = tune(Pd[t], y[t], cl); Bs.append(b); pd_[v] = (Pd[v] + b).argmax(1); pc_[v] = (Pc[v] + b).argmax(1)
        print(f'   {nm:10s} nested: clean {mf(y, pc_):.4f} fakePKS {mf(y, pd_):.4f}  scint F1 clean {f1_score(y == S, pc_ == S):.3f}'
              f' fPKS {f1_score(y == S, pd_ == S):.3f}  mean scint bias {np.mean(Bs, 0)[S]:+.2f} (fold sd {np.std(Bs, 0)[S]:.2f})', flush=True)
