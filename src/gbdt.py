"""LightGBM on feature CSVs. usage: gbdt.py TAG feat1 [feat2 ...]  (names: data/fe_{name}_{train,test}.csv)
env DROP=comma list of feature substrings to drop; ADV=1 adversarial check."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
from sklearn.metrics import confusion_matrix, roc_auc_score
TAG = sys.argv[1]; names = sys.argv[2:]
Ftr = pd.concat([pd.read_csv(D + rf'\fe_{n}_train.csv').add_prefix(n + '_') for n in names], axis=1)
Fte = pd.concat([pd.read_csv(D + rf'\fe_{n}_test.csv').add_prefix(n + '_') for n in names], axis=1)
drop = [c for c in Ftr.columns if any(s and s in c for s in os.environ.get('DROP', '').split(','))]
Ftr = Ftr.drop(columns=drop); Fte = Fte.drop(columns=drop)
y = y_train(); fo = folds()
prm = dict(n_estimators=3000, learning_rate=0.04, num_leaves=31, min_child_samples=20, colsample_bytree=0.6,
           subsample=0.8, subsample_freq=1, class_weight='balanced', verbose=-1, num_threads=int(os.environ.get('NT', 8)))
oof = np.zeros((len(y), 9)); pte = np.zeros((len(Fte), 9)); imp = 0
for k in range(5):
    tri, vai = fo != k, fo == k
    m = lgb.LGBMClassifier(**prm).fit(Ftr[tri], y[tri], eval_set=[(Ftr[vai], y[vai])], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[vai] = m.predict_proba(Ftr[vai]); pte += m.predict_proba(Fte) / 5; imp = imp + m.booster_.feature_importance('gain')
np.save(D + rf'\oof_{TAG}.npy', oof); np.save(D + rf'\pte_{TAG}.npy', pte)
p = oof.argmax(1)
with open(ROOT + rf'\logs\gbdt_{TAG}.txt', 'w') as o:
    print(TAG, names, 'nfeat', Ftr.shape[1], 'CV mF1', round(mf1(y, p), 4), file=o)
    print(per_class(y, p), file=o)
    print('confusion (rows true, cols pred, order', [l[:6] for l in LABELS], ')\n', confusion_matrix(y, p), file=o)
    print('test pred mix', dict(zip(LABELS, np.round(np.bincount(pte.argmax(1), minlength=9) / len(pte), 3))), file=o)
    print('top feats', pd.Series(imp, Ftr.columns).sort_values(ascending=False).head(25).round(0).to_dict(), file=o)
    if os.environ.get('ADV'):
        A = pd.concat([Ftr, Fte]); ya = np.r_[np.zeros(len(Ftr)), np.ones(len(Fte))]
        rng = np.random.default_rng(0); f5 = rng.integers(0, 5, len(A)); pa = np.zeros(len(A)); ai = 0
        for k in range(5):
            ma = lgb.LGBMClassifier(n_estimators=200, verbose=-1, num_threads=8).fit(A[f5 != k], ya[f5 != k])
            pa[f5 == k] = ma.predict_proba(A[f5 == k])[:, 1]; ai = ai + ma.booster_.feature_importance('gain')
        print('adversarial AUC', round(roc_auc_score(ya, pa), 4), file=o)
        print('adv top', pd.Series(ai, A.columns).sort_values(ascending=False).head(15).round(0).to_dict(), file=o)
print(open(ROOT + rf'\logs\gbdt_{TAG}.txt').read())
