"""Level-1 models on the bundle features, trained like src/bundle_eval.py: clean + deg2 + deg4 rows of the 4 training folds,
scored on the clean val fold (oofc) and its fake-PKS deg2 rendering (oofd); test (pte) = mean of the 5 fold models.
Saves results/probs_<name>.npz.  usage: python src/cascade/train_base.py <name> [<name> ...]
  lgb    multiclass LightGBM, harness params (a re-run of the harness baseline that also keeps the models' raw scores)
  lgbx   multiclass LightGBM with extra-trees splits, 63 leaves, 30% columns, other seed (diversity for averaging)
  ovr    9 one-vs-rest binary LightGBMs (p_k = P(class k present as label | x)), unweighted
  cond   8 conditional binary LightGBMs for the priority cascade: stage k is trained only on rows whose label is k or
         lower priority (the rows a top-down cascade still sees), target = (label == k); column 8 (noise) = 1
  pairs  pairwise specialists, trained only on rows of the two classes, p = P(first class | x)
  xgb / cat / mlp  XGBoost / CatBoost / MLP multiclass on the same rows (balanced class weights where supported)"""
import sys, time, numpy as np
from common import y, folds, K, LGB, R, load_X

PAIRS = [(3, 8), (3, 7), (3, 4), (6, 7), (5, 8)]  # scint-noise, scint-rfi_pers, scint-sat, rfi_int-rfi_pers, radar-noise


def bal_w(yy):
    c = np.bincount(yy, minlength=K); return (len(yy) / (K * c))[yy]


def lgb_bin(Xtr, t, n=900):
    import lightgbm as lgb
    return lgb.LGBMClassifier(**{**LGB, 'n_estimators': n}).fit(Xtr, t.astype(int))


def fit_predict(name, Xtr, ytr, Xs):
    """returns a list (one per array in Xs) of (n, c) score arrays"""
    if name == 'lgb':
        import lightgbm as lgb
        m = lgb.LGBMClassifier(**LGB, class_weight='balanced').fit(Xtr, ytr)
        return [m.predict_proba(X) for X in Xs]
    if name == 'lgbx':  # a more randomised LightGBM (extra-trees splits, other seed / column sampling) for blend diversity
        import lightgbm as lgb
        m = lgb.LGBMClassifier(**{**LGB, 'colsample_bytree': 0.3, 'num_leaves': 63, 'n_estimators': 1200},
                               extra_trees=True, random_state=1, class_weight='balanced').fit(Xtr, ytr)
        return [m.predict_proba(X) for X in Xs]
    if name == 'ovr':
        out = [np.zeros((len(X), K)) for X in Xs]
        for k in range(K):
            m = lgb_bin(Xtr, ytr == k)
            for o, X in zip(out, Xs): o[:, k] = m.predict_proba(X)[:, 1]
        return out
    if name == 'cond':
        out = [np.ones((len(X), K)) for X in Xs]
        for k in range(K - 1):
            s = ytr >= k
            m = lgb_bin(Xtr[s], ytr[s] == k)
            for o, X in zip(out, Xs): o[:, k] = m.predict_proba(X)[:, 1]
        return out
    if name == 'pairs':
        out = [np.zeros((len(X), len(PAIRS))) for X in Xs]
        for j, (a, b) in enumerate(PAIRS):
            s = (ytr == a) | (ytr == b)
            m = lgb_bin(Xtr[s], ytr[s] == a, n=600)
            for o, X in zip(out, Xs): o[:, j] = m.predict_proba(X)[:, 1]
        return out
    if name == 'xgb':
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.5,
                              min_child_weight=2, tree_method='hist', n_jobs=4, random_state=0)
        m.fit(Xtr, ytr, sample_weight=bal_w(ytr))
        return [m.predict_proba(X) for X in Xs]
    if name == 'cat':
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=1500, learning_rate=0.1, depth=6, loss_function='MultiClass',
                               auto_class_weights='Balanced', thread_count=4, random_seed=0, verbose=0)
        m.fit(Xtr, ytr)
        return [m.predict_proba(X) for X in Xs]
    if name == 'mlp':
        from sklearn.preprocessing import QuantileTransformer
        from sklearn.neural_network import MLPClassifier
        med = np.nanmedian(Xtr, 0)
        fill = lambda X: np.where(np.isnan(X), med, X)
        qt = QuantileTransformer(n_quantiles=200, output_distribution='normal', subsample=50000, random_state=0)
        Z = qt.fit_transform(fill(Xtr))
        # class balance by resampling (MLPClassifier has no sample weights)
        rng = np.random.default_rng(0); n = len(ytr) // K
        idx = np.concatenate([rng.choice(np.flatnonzero(ytr == k), n, replace=True) for k in range(K)])
        ps = []
        for seed in range(3):
            m = MLPClassifier(hidden_layer_sizes=(512, 256), alpha=1e-3, batch_size=256, learning_rate_init=1e-3,
                              max_iter=60, early_stopping=True, validation_fraction=0.1, n_iter_no_change=8,
                              random_state=seed).fit(Z[idx], ytr[idx])
            ps.append([m.predict_proba(qt.transform(fill(X))) for X in Xs])
        return [np.mean([p[i] for p in ps], 0) for i in range(len(Xs))]
    raise ValueError(name)


if __name__ == '__main__':
    X = load_X()
    for name in sys.argv[1:]:
        t0 = time.time(); oc = od = pt = None
        for k in range(5):
            t, v = folds != k, folds == k
            Xtr = np.vstack([X['clean'][t], X['deg2'][t], X['deg4'][t]]); ytr = np.r_[y[t], y[t], y[t]]
            a, b, c = fit_predict(name, Xtr, ytr, [X['clean'][v], X['deg2'][v], X['test']])
            if oc is None: oc, od, pt = np.zeros((len(y), a.shape[1])), np.zeros((len(y), a.shape[1])), 0
            oc[v], od[v] = a, b; pt = pt + c / 5
            print(f'{name} fold {k} done {time.time() - t0:.0f}s', flush=True)
        np.savez_compressed(R / f'probs_{name}.npz', oofc=oc, oofd=od, pte=pt)
        if oc.shape[1] == K:
            from common import f1m
            print(f'{name}: argmax clean-val {f1m(y, oc.argmax(1)):.4f} fakePKS-val {f1m(y, od.argmax(1)):.4f} '
                  f'({time.time() - t0:.0f}s)', flush=True)
