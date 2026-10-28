"""Meta-model variants with the complete pretrained CNNs inside (5-fold nested, fake-PKS macro-F1).
Same inputs as final_blend.py GSEP=1 SEP=1; varies which CNNs go in, training rows, meta size and the meta/blend50 mix."""
import sys, os; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
import lightgbm as lgb
y = y_train(); fo = folds(); L = lambda a: np.log(a + 1e-6)
GTS = os.environ.get('GBDT', 'A_rsswp_PL5,A_rsswp_PL5t6,A_rs_PL5t6,A_trk3_PL5t6,A_rsswp_PL4,A_rs_PL3').split(',')
G = [np.load(D + rf'\oofd_{t}.npy') for t in GTS]; Gc = [np.load(D + rf'\oofc_{t}.npy') for t in GTS]
def cnn(t):
    od, oc = np.zeros((len(y), 9)), np.zeros((len(y), 9))
    for k in range(5): v = fo == k; od[v] = np.load(D + rf'\oofdpart_{t}_f{k}.npy'); oc[v] = np.load(D + rf'\oofpart_{t}_f{k}.npy')
    return od, oc
OLD, NEW = ['B3pl2g', 'B3wg', 'B3plwg'], [t for t in os.environ.get('NEW', 'E1g,E2g,E3g,P1g').split(',')]
C = {t: cnn(t) for t in OLD + NEW}
def run(name, cn, rows='both', leaves=31, nest=300, lr=0.03, gsep=True):
    gd = G if gsep else [np.mean(G, 0)]; gc = Gc if gsep else [np.mean(Gc, 0)]
    X = np.hstack([L(a) for a in gd] + [L(C[t][0]) for t in cn]); Xc = np.hstack([L(a) for a in gc] + [L(C[t][1]) for t in cn])
    B = (np.mean(G, 0) + np.mean([C[t][0] for t in cn], 0)) / 2; pm = np.zeros((len(y), 9))
    for k in range(5):
        t, v = fo != k, fo == k
        m = lgb.LGBMClassifier(n_estimators=nest, learning_rate=lr, num_leaves=leaves, min_child_samples=30, colsample_bytree=0.6, class_weight='balanced', verbose=-1, num_threads=6)
        Xtr, ytr = (np.vstack([X[t], Xc[t]]), np.r_[y[t], y[t]]) if rows == 'both' else (X[t], y[t])
        m.fit(Xtr, ytr); pm[v] = m.predict_proba(X[v])
    out = {a: round(mf1(y, (a * pm + (1 - a) * B).argmax(1)), 4) for a in (0.5, 0.7, 0.85, 1.0)}
    print(f'{name:34s} meta/blend mix a -> fPKS {out}', flush=True); return pm, B
run('V0 current (old+4 new, both rows)', OLD + NEW)
run('V1 only 4 pretrained', NEW)
run('V2 fPKS rows only', OLD + NEW, rows='fpks')
run('V3 leaves 15', OLD + NEW, leaves=15)
run('V4 600 trees lr .02', OLD + NEW, nest=600, lr=0.02)
run('V5 GBDTs averaged (no GSEP)', OLD + NEW, gsep=False)
