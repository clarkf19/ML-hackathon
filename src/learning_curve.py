"""Is the feature model data-limited? Train on a fraction of each training fold (same folds, same clean+4 fake copies),
score full val folds on clean and fake-PKS."""
import sys, os; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
import lightgbm as lgb
SETS = ['phys', 'tex', 'trk', 'rep', 'rep2', 'trk2', 'lvl']
def fn(w, s):
    if w == 'clean': return f'fe_{s}_train.csv'
    return {'tex': f'fe_texd_{w}.csv', 'trk': f'fe_trkd_{w}.csv', 'trk2': f'fe_trk2_train_{w}.csv'}.get(s, f'fe_{s}_train_{w}.csv')
X = {w: pd.concat([pd.read_csv(os.path.join(D, fn(w, s))) for s in SETS], axis=1).values.astype(np.float32) for w in ['clean', 'deg2', 'deg4', 'deg6', 'deg8']}
y = y_train(); fo = folds(); rng = np.random.default_rng(0); keep_r = rng.random(len(y))
for frac in [float(a) for a in sys.argv[1:]]:
    oc, od = np.zeros((len(y), 9)), np.zeros((len(y), 9))
    for k in range(5):
        t = (fo != k) & (keep_r < frac); v = fo == k
        m = lgb.LGBMClassifier(n_estimators=900, learning_rate=0.05, num_leaves=31, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
                               class_weight='balanced', verbose=-1, num_threads=6)
        m.fit(np.vstack([X[w][t] for w in X]), np.tile(y[t], len(X)))
        oc[v] = m.predict_proba(X['clean'][v]); od[v] = m.predict_proba(X['deg2'][v])
    print(f'train fraction {frac:.2f} ({int((keep_r < frac).sum() * 0.8)} cadences/fold): clean {mf1(y, oc.argmax(1)):.4f}  fakePKS {mf1(y, od.argmax(1)):.4f}', flush=True)
