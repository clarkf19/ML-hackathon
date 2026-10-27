import sys, os; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
import lightgbm as lgb
y = y_train(); fo = folds(); ks = [k for k in range(5) if os.path.exists(D + rf'\oofdpart_B3g_f{k}.npy')]
idx = np.concatenate([np.where(fo == k)[0] for k in ks]); yy = y[idx]; ff = fo[idx]
G = np.load(D + r'\oofd_A_lvl.npy')[idx]; C = np.concatenate([np.load(D + rf'\oofdpart_B3g_f{k}.npy') for k in ks])
Gc = np.load(D + r'\oofc_A_lvl.npy')[idx]; Cc = np.concatenate([np.load(D + rf'\oofpart_B3g_f{k}.npy') for k in ks])
X = np.hstack([np.log(G + 1e-6), np.log(C + 1e-6)]); Xc = np.hstack([np.log(Gc + 1e-6), np.log(Cc + 1e-6)])
B, Bc = (G + C) / 2, (Gc + Cc) / 2
def run(train_rows):
    pd_, pc_ = np.zeros((len(yy), 9)), np.zeros((len(yy), 9))
    for k in ks:
        t, v = ff != k, ff == k
        XT = {'both': np.vstack([X[t], Xc[t]]), 'clean': Xc[t], 'fake': X[t]}[train_rows]
        YT = np.r_[yy[t], yy[t]] if train_rows == 'both' else yy[t]
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, class_weight='balanced', verbose=-1, num_threads=8).fit(XT, YT)
        pd_[v] = m.predict_proba(X[v]); pc_[v] = m.predict_proba(Xc[v])
    return pd_, pc_
print(f'blend50          fPKS {mf1(yy, B.argmax(1)):.4f}  clean {mf1(yy, Bc.argmax(1)):.4f}')
for tr in ['both', 'clean', 'fake']:
    pd_, pc_ = run(tr)
    print(f'meta trained on {tr:5s}: meta fPKS {mf1(yy, pd_.argmax(1)):.4f} clean {mf1(yy, pc_.argmax(1)):.4f} | avg(meta,blend) fPKS {mf1(yy, (pd_ + B).argmax(1)):.4f} clean {mf1(yy, (pc_ + Bc).argmax(1)):.4f}')
    print('    per-class fPKS avg:', per_class(yy, (pd_ + B).argmax(1)))
print('    per-class fPKS blend50:', per_class(yy, B.argmax(1)))
