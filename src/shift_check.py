"""Domain-shift check of raw statistics: per feature, AUC(test vs train rendering) for clean / fake-PKS (degrade only) /
fake-PKS + raw qmap, on noise cadences (train: true noise; test: current model pte noise p>0.9) so label mix does not
confound. |AUC-0.5| near 0 = the simulator reproduces the test distribution of that statistic.
usage: python src/shift_check.py -> results/shift_auc.csv"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parent.parent; D = ROOT / 'data'
m = np.load(ROOT / 'bundle/meta.npz', allow_pickle=True); y, pte = m['y'], m['pte']
ld = lambda n: np.load(D / f'rs_{n}.npz')
cols = list(ld('train')['cols']); T = ld('test')['X'][(pte.argmax(1) == 8) & (pte.max(1) > 0.9)]
rng = np.random.default_rng(0); nz = rng.permutation(np.where(y == 8)[0])[:3000]; T = T[rng.permutation(len(T))[:3000]]
res = {'feat': cols}
for n in ['train', 'd2', 'd4', 'd2q', 'd4q']:
    if not (D / f'rs_{n}.npz').exists(): continue
    A = ld(n)['X'][nz]; a = []
    for j in range(len(cols)):
        u = np.r_[A[:, j], T[:, j]]; u = np.nan_to_num(u, nan=np.nanmedian(u))
        a.append(roc_auc_score(np.r_[np.zeros(len(A)), np.ones(len(T))], u))
    res[f'shift_{n}'] = np.abs(np.array(a) - 0.5)
R = pd.DataFrame(res); R.to_csv(ROOT / 'results/shift_auc.csv', index=False)
print('noise cadences: test', len(T), 'train', len(nz)); print(R.describe().round(3).to_string())
print(R.sort_values('shift_d2', ascending=False).head(30).round(3).to_string(index=False))
