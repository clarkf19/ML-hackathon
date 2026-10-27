"""Final-blend score restricted to folds where all given CNNs exist (nested over those folds)."""
import sys, os; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
import lightgbm as lgb
y = y_train(); fo = folds()
GTS = os.environ.get('GBDT', 'A_rs,A_lvl_PL,A_trk3').split(',')
G = np.mean([np.load(D + rf'\oofd_{t}.npy') for t in GTS], 0); Gc = np.mean([np.load(D + rf'\oofc_{t}.npy') for t in GTS], 0)
L = lambda a: np.log(a + 1e-6)
def score(tags, ks):
    idx = np.concatenate([np.where(fo == k)[0] for k in ks]); yy, ff = y[idx], fo[idx]
    Cd = [np.concatenate([np.load(D + rf'\oofdpart_{t}_f{k}.npy') for k in ks]) for t in tags]
    Cc = [np.concatenate([np.load(D + rf'\oofpart_{t}_f{k}.npy') for k in ks]) for t in tags]
    X = np.hstack([L(G[idx])] + [L(c) for c in Cd]); Xc = np.hstack([L(Gc[idx])] + [L(c) for c in Cc])
    B = (G[idx] + np.mean(Cd, 0)) / 2; pm = np.zeros((len(yy), 9))
    for k in ks:
        t, v = ff != k, ff == k
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, class_weight='balanced', verbose=-1, num_threads=8)
        m.fit(np.vstack([X[t], Xc[t]]), np.r_[yy[t], yy[t]]); pm[v] = m.predict_proba(X[v])
    return mf1(yy, ((pm + B) / 2).argmax(1)), mf1(yy, np.mean(Cd, 0).argmax(1))
ks = [int(k) for k in sys.argv[1].split(',')]
for tags in [a.split(',') for a in sys.argv[2:]]:
    f, c = score(tags, ks); print(f'folds {ks} CNNs {tags}: CNN-avg {c:.4f} | FINAL {f:.4f}', flush=True)
