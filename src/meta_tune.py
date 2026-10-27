"""Decision-layer variants on complete 5-fold OOFs (all nested over folds): meta size, meta weight vs blend, LR meta."""
import sys, os; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
y = y_train(); fo = folds()
GTS = os.environ.get('GBDT', 'A_rs,A_lvl_PL,A_trk3').split(','); TAGS = sys.argv[1:]
G = np.mean([np.load(D + rf'\oofd_{t}.npy') for t in GTS], 0); Gc = np.mean([np.load(D + rf'\oofc_{t}.npy') for t in GTS], 0)
def cnn(t, pre):
    o = np.zeros((len(y), 9))
    for k in range(5): o[fo == k] = np.load(D + rf'\{pre}_{t}_f{k}.npy')
    return o
Cd = [cnn(t, 'oofdpart') for t in TAGS]; Cc = [cnn(t, 'oofpart') for t in TAGS]
L = lambda a: np.log(a + 1e-6)
X = np.hstack([L(G)] + [L(c) for c in Cd]); Xc = np.hstack([L(Gc)] + [L(c) for c in Cc]); B = (G + np.mean(Cd, 0)) / 2
def meta(mk):
    pm = np.zeros((len(y), 9))
    for k in range(5):
        t, v = fo != k, fo == k; m = mk().fit(np.vstack([X[t], Xc[t]]), np.r_[y[t], y[t]]); pm[v] = m.predict_proba(X[v])
    return pm
res = {}
for nm, mk in [('lgb300_15', lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, class_weight='balanced', verbose=-1, num_threads=8)),
               ('lgb600_7', lambda: lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=7, min_child_samples=60, class_weight='balanced', verbose=-1, num_threads=8)),
               ('lgb300_31', lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=30, colsample_bytree=0.6, class_weight='balanced', verbose=-1, num_threads=8)),
               ('logreg', lambda: LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced'))]:
    pm = meta(mk); res[nm] = pm
    print(f'{nm:10s} meta {mf1(y, pm.argmax(1)):.4f} | ' + ' '.join(f'w={w}:{mf1(y, (w * pm + (1 - w) * B).argmax(1)):.4f}' for w in (0.3, 0.5, 0.7)), flush=True)
avg = (res['lgb300_15'] + res['logreg']) / 2
print(f'avg(lgb,logreg) meta {mf1(y, avg.argmax(1)):.4f} | w=0.5 with blend {mf1(y, (0.5 * avg + 0.5 * B).argmax(1)):.4f}')
