"""rs feature set: robust raw-domain statistics (src/rawstats.py over raw fake-PKS copies from src/rawsim.py), keeping only
columns whose fake-PKS-vs-clean shift AUC is < 0.1 above chance (results/shift_auc.csv) -> data/fe_rs_{train,test,train_degN}.csv
usage (after the raw copies + stats):  QMAP=1 COPIES=d2,d4,d6,d8 python src/rawsim.py && python src/rawstats.py train test d2q d4q d6q d8q && python src/build_rs.py"""
import numpy as np, pandas as pd, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.join(ROOT, 'data')
shift = pd.read_csv(os.path.join(ROOT, 'results', 'shift_auc.csv')).set_index('feat')['shift_d2q']
z = np.load(os.path.join(D, 'rs_train.npz')); cols = list(z['cols']); keep = [j for j, c in enumerate(cols) if shift.get(c, 1) < 0.1]
print('robust cols', len(keep), 'of', len(cols))
for src, dst in [('train', 'fe_rs_train.csv'), ('test', 'fe_rs_test.csv'), ('d2q', 'fe_rs_train_deg2.csv'), ('d4q', 'fe_rs_train_deg4.csv'), ('d6q', 'fe_rs_train_deg6.csv'), ('d8q', 'fe_rs_train_deg8.csv')]:
    X = np.load(os.path.join(D, f'rs_{src}.npz'))['X'][:, keep]
    pd.DataFrame(X, columns=[cols[j] for j in keep]).to_csv(os.path.join(D, dst), index=False); print(dst, X.shape)
