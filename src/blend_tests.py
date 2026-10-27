"""Decision-layer tests on folds with CNN OOF: nested per-class bias, stacked LGBM meta-model, blend weight per class."""
import sys, os; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
import lightgbm as lgb
y = y_train(); fo = folds(); TAGC = sys.argv[1] if len(sys.argv) > 1 else 'B3g'
ks = [k for k in range(5) if os.path.exists(D + rf'\oofdpart_{TAGC}_f{k}.npy')]
idx = np.concatenate([np.where(fo == k)[0] for k in ks]); yy = y[idx]; ff = fo[idx]
G = np.load(D + r'\oofd_A_lvl.npy')[idx]; C = np.concatenate([np.load(D + rf'\oofdpart_{TAGC}_f{k}.npy') for k in ks])
Gc = np.load(D + r'\oofc_A_lvl.npy')[idx]; Cc = np.concatenate([np.load(D + rf'\oofpart_{TAGC}_f{k}.npy') for k in ks])
B = (G + C) / 2; Bc = (Gc + Cc) / 2
print(f'folds {ks}  blend50 fPKS {mf1(yy, B.argmax(1)):.4f}  clean {mf1(yy, Bc.argmax(1)):.4f}')
def tune_bias(L, t):
    b = np.zeros(9)
    for _ in range(3):
        for c in range(9):
            best = (mf1(t, (L + b).argmax(1)), b[c])
            for v in np.linspace(-1.5, 1.5, 31):
                bb = b.copy(); bb[c] = v; s = mf1(t, (L + bb).argmax(1))
                if s > best[0] + 1e-6: best = (s, v)
            b[c] = best[1]
    return b
LB = np.log(B + 1e-9); pr = np.zeros(len(yy), int)
for k in ks:
    t, v = ff != k, ff == k; pr[v] = (LB[v] + tune_bias(LB[t], yy[t])).argmax(1)
print(f'nested bias on blend: fPKS {mf1(yy, pr):.4f}')
# stacked meta-model on [log G, log C] trained on fake-PKS OOF of other folds (and clean OOF as extra rows)
XS = np.hstack([np.log(G + 1e-6), np.log(C + 1e-6)]); XSc = np.hstack([np.log(Gc + 1e-6), np.log(Cc + 1e-6)])
pm = np.zeros((len(yy), 9))
for k in ks:
    t, v = ff != k, ff == k
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, class_weight='balanced', verbose=-1, num_threads=8)
    m.fit(np.vstack([XS[t], XSc[t]]), np.r_[yy[t], yy[t]]); pm[v] = m.predict_proba(XS[v])
print(f'stacked meta LGBM: fPKS {mf1(yy, pm.argmax(1)):.4f}   avg(meta, blend) {mf1(yy, (pm + B).argmax(1)):.4f}')
# per-class CNN weight (nested)
pw = np.zeros(len(yy), int)
for k in ks:
    t, v = ff != k, ff == k; w = np.full(9, 0.5)
    for c in range(9):
        best = (-1, 0.5)
        for wc in np.linspace(0.2, 0.8, 7):
            ww = w.copy(); ww[c] = wc; s = mf1(yy[t], ((1 - ww) * G[t] + ww * C[t]).argmax(1))
            if s > best[0]: best = (s, wc)
        w[c] = best[1]
    pw[v] = ((1 - w) * G[v] + w * C[v]).argmax(1)
print(f'nested per-class weights: fPKS {mf1(yy, pw):.4f}')
