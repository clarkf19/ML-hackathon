"""K proxy: per fold, labeled = clean+deg2 of other folds; unlabeled = fake-PKS val fold (labels never used).
Round 0: train, predict unlabeled. Round 1: add confident pseudo-labels (p>TH), retrain, re-predict. Report macro-F1 both."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
SETS = ['phys', 'tex', 'trk', 'rep', 'rep2']
FN = {'clean': lambda s: f'fe_{s}_train.csv', 'deg2': lambda s: {'tex': 'fe_texd_deg2.csv', 'trk': 'fe_trkd_deg2.csv'}.get(s, f'fe_{s}_train_deg2.csv')}
def load(w): return pd.concat([pd.read_csv(os.path.join(D, FN[w](s))).add_prefix(s + '_') for s in SETS], axis=1)
Xc, Xd = load('clean'), load('deg2'); y = y_train(); fo = folds()
TH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
prm = dict(n_estimators=900, learning_rate=0.05, num_leaves=31, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
           class_weight='balanced', verbose=-1, num_threads=int(os.environ.get('NT', 8)))
p0 = np.zeros((len(y), 9)); p1 = np.zeros((len(y), 9)); used = 0
for k in range(5):
    tri, vai = fo != k, fo == k
    Xl = pd.concat([Xc[tri], Xd[tri]]) if os.environ.get('LAB', 'both') == 'both' else Xc[tri]; yl = np.r_[y[tri], y[tri]] if os.environ.get('LAB', 'both') == 'both' else y[tri]
    m = lgb.LGBMClassifier(**prm).fit(Xl, yl); pu = m.predict_proba(Xd[vai]); p0[vai] = pu
    conf = pu.max(1) > TH; used += conf.sum()
    m = lgb.LGBMClassifier(**prm).fit(pd.concat([Xl, Xd[vai][conf]]), np.r_[yl, pu.argmax(1)[conf]]); p1[vai] = m.predict_proba(Xd[vai])
    print(k, flush=True)
msg = f'PL proxy LAB={os.environ.get("LAB", "both")} TH={TH}: round0 {mf1(y, p0.argmax(1)):.4f} -> round1 {mf1(y, p1.argmax(1)):.4f}  (pseudo used {used}/{len(y)}, their acc {np.mean(p0.argmax(1)[p0.max(1) > TH] == y[p0.max(1) > TH]):.4f})\n  r1 {per_class(y, p1.argmax(1))}'
print(msg); open(ROOT + r'\logs\aug_results.txt', 'a').write(msg + '\n')
