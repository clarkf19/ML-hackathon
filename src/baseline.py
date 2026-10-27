"""A0: starter-style per-panel stats on normalised z + LightGBM, 5-fold CV; adversarial validation."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, json
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *

def feats(z):
    out = []
    for s in range(0, len(z), 2000):
        x = z[s:s + 2000].astype(np.float32)
        prof = x.mean(2)                     # (b,6,128) time-avg spectrum
        rowp = x.mean(3)                     # (b,6,16) broadband
        f = {}
        for p in range(6):
            f[f'p{p}_std'] = x[:, p].std((1, 2)); f[f'p{p}_max'] = x[:, p].max((1, 2))
            f[f'p{p}_peak'] = prof[:, p].max(1); f[f'p{p}_peak_ch'] = prof[:, p].argmax(1)
            f[f'p{p}_rowmax'] = rowp[:, p].max(1); f[f'p{p}_rowstd'] = rowp[:, p].std(1)
            f[f'p{p}_q99'] = np.quantile(x[:, p].reshape(len(x), -1), 0.99, axis=1)
        df = pd.DataFrame(f)
        for k in ['std', 'max', 'peak', 'rowmax', 'rowstd', 'q99']:
            on = df[[f'p{p}_{k}' for p in (0, 2, 4)]]; off = df[[f'p{p}_{k}' for p in (1, 3, 5)]]
            df[f'on_off_{k}'] = on.mean(1) - off.mean(1); df[f'onmin_offmax_{k}'] = on.min(1) - off.max(1)
        out.append(df)
    return pd.concat(out, ignore_index=True)

if __name__ == '__main__':
    tr, te = meta(); y = y_train(); fo = folds()
    Ftr = feats(np.load(D + r'\z_train.npy', mmap_mode='r')); Fte = feats(np.load(D + r'\z_test.npy', mmap_mode='r'))
    Ftr.to_csv(D + r'\fe_base_train.csv', index=False); Fte.to_csv(D + r'\fe_base_test.csv', index=False)
    P = dict(objective='multiclass', num_class=9, learning_rate=0.05, num_leaves=31, min_child_samples=20,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=16, class_weight=None)
    oof = np.zeros((len(y), 9)); pte = np.zeros((len(te), 9))
    for k in range(5):
        tri, vai = fo != k, fo == k
        m = lgb.LGBMClassifier(n_estimators=2000, class_weight='balanced', **{k2: v for k2, v in P.items() if k2 not in ('objective', 'num_class', 'class_weight')})
        m.fit(Ftr[tri], y[tri], eval_set=[(Ftr[vai], y[vai])], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[vai] = m.predict_proba(Ftr[vai]); pte += m.predict_proba(Fte) / 5
    print('A0 CV macroF1', round(mf1(y, oof.argmax(1)), 4), per_class(y, oof.argmax(1)))
    np.save(D + r'\oof_A0.npy', oof); np.save(D + r'\pte_A0.npy', pte)
    print('test pred dist', write_sub(pte.argmax(1), 'submission1.csv'))
    # adversarial validation
    A = pd.concat([Ftr, Fte]); ya = np.r_[np.zeros(len(Ftr)), np.ones(len(Fte))]
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    pa = cross_val_predict(lgb.LGBMClassifier(n_estimators=300, verbose=-1, num_threads=16), A, ya, cv=5, method='predict_proba')[:, 1]
    print('adversarial AUC', round(roc_auc_score(ya, pa), 4))
    m = lgb.LGBMClassifier(n_estimators=300, verbose=-1, num_threads=16).fit(A, ya)
    imp = pd.Series(m.booster_.feature_importance('gain'), A.columns).sort_values(ascending=False)
    print('adv top', imp.head(8).round(0).to_dict())
