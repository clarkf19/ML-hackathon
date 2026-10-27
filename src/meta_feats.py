"""Meta-model that sees probabilities AND physics features (nested 5-fold, train rows = clean + fake-PKS OOF)."""
import sys, os; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
import lightgbm as lgb
y = y_train(); fo = folds()
SETS = ['phys', 'tex', 'trk', 'rep', 'rep2', 'trk2', 'lvl']
FN = {'clean': lambda s: f'fe_{s}_train.csv', 'deg2': lambda s: {'tex': 'fe_texd_deg2.csv', 'trk': 'fe_trkd_deg2.csv', 'trk2': 'fe_trk2_train_deg2.csv'}.get(s, f'fe_{s}_train_deg2.csv')}
Fc = pd.concat([pd.read_csv(os.path.join(D, FN['clean'](s))) for s in SETS], axis=1).values.astype(np.float32)
Fd = pd.concat([pd.read_csv(os.path.join(D, FN['deg2'](s))) for s in SETS], axis=1).values.astype(np.float32)
G, Gc = np.load(D + r'\oofd_A_lvl.npy'), np.load(D + r'\oofc_A_lvl.npy')
C = np.zeros_like(G); Cc = np.zeros_like(G)
for k in range(5): v = fo == k; C[v] = np.load(D + rf'\oofdpart_B3g_f{k}.npy'); Cc[v] = np.load(D + rf'\oofpart_B3g_f{k}.npy')
L = lambda a: np.log(a + 1e-6); B = (G + C) / 2
TOPK = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if TOPK:   # most important GBDT features (by a quick model) to keep the meta small
    m0 = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, verbose=-1, num_threads=8).fit(Fd, y)
    keep = np.argsort(m0.booster_.feature_importance('gain'))[::-1][:TOPK]; Fc, Fd = Fc[:, keep], Fd[:, keep]
X, Xc = np.hstack([L(G), L(C), Fd]), np.hstack([L(Gc), L(Cc), Fc])
pm = np.zeros_like(G)
for k in range(5):
    t, v = fo != k, fo == k
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40, colsample_bytree=0.5, class_weight='balanced', verbose=-1, num_threads=8)
    m.fit(np.vstack([X[t], Xc[t]]), np.r_[y[t], y[t]]); pm[v] = m.predict_proba(X[v])
print(f'TOPK={TOPK or "all"}: meta+feats fPKS {mf1(y, pm.argmax(1)):.4f} | avg(meta+feats, blend) {mf1(y, (pm + B).argmax(1)):.4f}   (reference: prob-only final 0.9172)')
print('  per-class', per_class(y, ((pm + B) / 2).argmax(1)))
