"""Portable (Linux/Windows) evaluation harness for cloud experiments.
Baseline = LightGBM on bundle features, trained on clean + fake-PKS deg2 + deg4, scored on clean val AND fake-PKS (deg2) val,
same 5 folds as the main pipeline. Add your own features: pass .npz files with arrays X_clean (18000,k), X_test (10000,k),
X_deg2 (18000,k), X_deg4 (18000,k) [deg arrays may be omitted -> clean copy used, which is WRONG for robustness: compute them].
usage: python src/bundle_eval.py [extra1.npz ...]     -> prints macro-F1 clean / fake-PKS and per-class F1"""
import sys, numpy as np, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import f1_score
B = Path(__file__).resolve().parent.parent / 'bundle'
m = np.load(B / 'meta.npz', allow_pickle=True); y, fo, L = m['y'], m['folds'], list(m['labels'])
X = {w: np.load(B / f'feats_{w}.npz')['X'] for w in ['clean', 'test', 'deg2', 'deg4']}
for e in sys.argv[1:]:
    z = np.load(e)
    for w in X: X[w] = np.hstack([X[w], z['X_' + w] if 'X_' + w in z else z['X_clean' if w != 'test' else 'X_test']])
prm = dict(n_estimators=900, learning_rate=0.05, num_leaves=31, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
           class_weight='balanced', verbose=-1)
oc = np.zeros((len(y), 9)); od = np.zeros((len(y), 9)); pt = np.zeros((10000, 9))
for k in range(5):
    t, v = fo != k, fo == k
    mdl = lgb.LGBMClassifier(**prm).fit(np.vstack([X['clean'][t], X['deg2'][t], X['deg4'][t]]), np.r_[y[t], y[t], y[t]])
    oc[v] = mdl.predict_proba(X['clean'][v]); od[v] = mdl.predict_proba(X['deg2'][v]); pt += mdl.predict_proba(X['test']) / 5
f = lambda a: f1_score(y, a.argmax(1), average='macro')
print(f'n_features {X["clean"].shape[1]}  clean-val {f(oc):.4f}  fakePKS-val {f(od):.4f}')
print('per-class fakePKS', dict(zip(L, np.round(f1_score(y, od.argmax(1), average=None), 3))))
out = Path('results'); out.mkdir(exist_ok=True); tag = '_'.join(Path(a).stem for a in sys.argv[1:]) or 'baseline'
np.savez_compressed(out / f'probs_{tag}.npz', oofc=oc, oofd=od, pte=pt)
