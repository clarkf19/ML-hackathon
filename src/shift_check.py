"""Per-statistic domain shift: AUC of test vs clean train and test vs fake-PKS (deg2) for every column of a stats .npz
(|AUC-.5| near 0 = fake-PKS reproduces the test distribution). Use it to drop statistics the simulator does not render.
usage: python src/shift_check.py stats.npz [tag]   -> results/shift_{tag}.csv"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
z = np.load(sys.argv[1], allow_pickle=True); tag = sys.argv[2] if len(sys.argv) > 2 else Path(sys.argv[1]).stem
names = list(z['names']); rng = np.random.default_rng(0)
def auc(a, b):
    a = a[rng.choice(len(a), min(len(a), 8000), replace=False)]; b = b[rng.choice(len(b), min(len(b), 8000), replace=False)]
    X = np.nan_to_num(np.vstack([a, b]), nan=-1e9); r = rankdata(X, axis=0); n1 = len(a)
    return (r[:n1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * len(b))
S = pd.DataFrame({'stat': names, 'auc_test_vs_clean': auc(z['X_test'], z['X_clean']), 'auc_test_vs_deg2': auc(z['X_test'], z['X_deg2']),
                  'auc_test_vs_deg4': auc(z['X_test'], z['X_deg4'])})
S['shift_deg'] = np.maximum(np.abs(S.auc_test_vs_deg2 - .5), np.abs(S.auc_test_vs_deg4 - .5)) + .5
S.sort_values('shift_deg', ascending=False).to_csv(Path('results') / f'shift_{tag}.csv', index=False, float_format='%.4f')
print(f'{tag}: median |AUC-.5| test-vs-clean {np.median(np.abs(S.auc_test_vs_clean - .5)):.3f}  test-vs-fakePKS {np.median(S.shift_deg - .5):.3f}; '
      f'stats with fakePKS shift > .75: {(S.shift_deg > .75).sum()} / {len(S)}')
print(S.sort_values('shift_deg', ascending=False).head(10).to_string(index=False))
