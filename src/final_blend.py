"""Final decision layer: GBDT (A_lvl) + CNN (5-fold) -> 50/50 blend + stacked LGBM meta-model (trained on clean+fake-PKS OOF).
Nested CV score, then fit meta on all OOF and apply to test. usage: final_blend.py OUTFILE CNNTAG [CNNTAG2 ...]"""
import sys, os; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
import lightgbm as lgb
OUT, TAGS = sys.argv[1], sys.argv[2:]
y = y_train(); fo = folds()
GT = os.environ.get('GBDT', 'A_lvl')            # feature-model version for OOF/test
GTEST = os.environ.get('GBDT_TEST', GT)          # e.g. a pseudo-label variant used only for test predictions
GTS = GT.split(','); GTT = GTEST.split(',')        # several feature models -> averaged
G = np.mean([np.load(D + rf'\oofd_{t}.npy') for t in GTS], 0); Gc = np.mean([np.load(D + rf'\oofc_{t}.npy') for t in GTS], 0)
Gt = np.mean([np.load(D + rf'\pte_{t}.npy') for t in GTT], 0)
SEP = bool(os.environ.get('SEP', '1') == '1')    # each CNN its own meta input
def cnn(tag):
    od, oc = np.zeros((len(y), 9)), np.zeros((len(y), 9))
    for k in range(5):
        v = fo == k; od[v] = np.load(D + rf'\oofdpart_{tag}_f{k}.npy'); oc[v] = np.load(D + rf'\oofpart_{tag}_f{k}.npy')
    return od, oc, np.mean([np.load(D + rf'\ptepart_{tag}_f{k}.npy') for k in range(5)], 0)
CS = [cnn(t) for t in TAGS]
C = np.mean([c[0] for c in CS], 0); Cc = np.mean([c[1] for c in CS], 0); Ct = np.mean([c[2] for c in CS], 0)
B, Bc, Bt = (G + C) / 2, (Gc + Cc) / 2, (Gt + Ct) / 2
L = lambda a: np.log(a + 1e-6)
if SEP and len(CS) > 1:
    X = np.hstack([L(G)] + [L(c[0]) for c in CS]); Xc = np.hstack([L(Gc)] + [L(c[1]) for c in CS]); Xt = np.hstack([L(Gt)] + [L(c[2]) for c in CS])
else:
    X, Xc, Xt = np.hstack([L(G), L(C)]), np.hstack([L(Gc), L(Cc)]), np.hstack([L(Gt), L(Ct)])
mk = lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=int(os.environ.get('META_LEAVES', 31)), min_child_samples=30, colsample_bytree=0.6, class_weight='balanced', verbose=-1, num_threads=8)
pm, pmc = np.zeros((len(y), 9)), np.zeros((len(y), 9))
for k in range(5):
    t, v = fo != k, fo == k
    m = mk().fit(np.vstack([X[t], Xc[t]]), np.r_[y[t], y[t]]); pm[v] = m.predict_proba(X[v]); pmc[v] = m.predict_proba(Xc[v])
F, Fc = (pm + B) / 2, (pmc + Bc) / 2
np.save(os.path.join(D, f'oofd_final_{OUT[:-4]}.npy'), F); np.save(os.path.join(D, f'oofc_final_{OUT[:-4]}.npy'), Fc)
for t, c in zip(TAGS, CS): print(f'  CNN {t}: fPKS {mf1(y, c[0].argmax(1)):.4f} clean {mf1(y, c[1].argmax(1)):.4f} | blend50 with GBDT fPKS {mf1(y, ((G + c[0]) / 2).argmax(1)):.4f}')
print(f'GBDT={GT} (test: {GTEST}) SEP={SEP} CNN {TAGS}: GBDT fPKS {mf1(y, G.argmax(1)):.4f} | CNN {mf1(y, C.argmax(1)):.4f} | blend50 {mf1(y, B.argmax(1)):.4f} | meta {mf1(y, pm.argmax(1)):.4f} | FINAL avg(meta,blend) fPKS {mf1(y, F.argmax(1)):.4f}  clean {mf1(y, Fc.argmax(1)):.4f}')
print('per-class final fPKS', per_class(y, F.argmax(1)))
for k in range(5): print(f'  fold{k}: blend50 {mf1(y[fo == k], B[fo == k].argmax(1)):.4f} -> final {mf1(y[fo == k], F[fo == k].argmax(1)):.4f}')
m = mk().fit(np.vstack([X, Xc]), np.r_[y, y]); Ft = (m.predict_proba(Xt) + Bt) / 2
np.save(D + rf'\pte_{OUT[:-4]}.npy', Ft)
if OUT != 'none.csv': print(write_sub(Ft.argmax(1), OUT))
