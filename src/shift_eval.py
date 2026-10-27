"""Proxy-test: does the model survive the GBT->PKS shift? Same folds.
Trains on {clean, deg, both} feature sets of train; evaluates on clean-val and deg-val. Also adv AUC deg-train vs test.
usage: shift_eval.py clean_name deg_name   (fe_{name}_train.csv)"""
import numpy as np, pandas as pd, lightgbm as lgb, sys
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
from sklearn.metrics import roc_auc_score
cn, dn = sys.argv[1], sys.argv[2]
A = pd.read_csv(D + rf'\fe_{cn}_train.csv'); B = pd.read_csv(D + rf'\fe_{dn}_train.csv'); T = pd.read_csv(D + rf'\fe_{cn}_test.csv')
y = y_train(); fo = folds()
prm = dict(n_estimators=600, learning_rate=0.06, num_leaves=31, colsample_bytree=0.6, subsample=0.8, subsample_freq=1,
           class_weight='balanced', verbose=-1, num_threads=8)
res = {}; oofs = {}
for trn in ['clean', 'deg', 'both']:
    oc = np.zeros((len(y), 9)); od = np.zeros((len(y), 9))
    for k in range(5):
        tri, vai = fo != k, fo == k
        Xt = {'clean': A[tri], 'deg': B[tri], 'both': pd.concat([A[tri], B[tri]])}[trn]
        yt = np.r_[y[tri], y[tri]] if trn == 'both' else y[tri]
        m = lgb.LGBMClassifier(**prm).fit(Xt, yt)
        oc[vai] = m.predict_proba(A[vai]); od[vai] = m.predict_proba(B[vai])
    res[trn] = (round(mf1(y, oc.argmax(1)), 4), round(mf1(y, od.argmax(1)), 4)); oofs[trn] = od
    print(f'train={trn:5s}  val clean {res[trn][0]}  val fakePKS {res[trn][1]}  perclass(fakePKS) {per_class(y, od.argmax(1))}', flush=True)
for nm, S in [('clean-train', A), ('fakePKS-train', B)]:
    Z = pd.concat([S, T]); ya = np.r_[np.zeros(len(S)), np.ones(len(T))]; f5 = np.random.default_rng(0).integers(0, 5, len(Z)); pa = np.zeros(len(Z))
    for k in range(5):
        pa[f5 == k] = lgb.LGBMClassifier(n_estimators=200, verbose=-1, num_threads=8).fit(Z[f5 != k], ya[f5 != k]).predict_proba(Z[f5 == k])[:, 1]
    print(f'adversarial AUC {nm} vs test: {roc_auc_score(ya, pa):.4f}', flush=True)
