"""Validate the generator in the 427-feature space.
1. LightGBM trained ONLY on generated GBT cadences, scored on REAL train (all rows + per fixed fold).
2. Adversarial AUC generated-vs-real (5-fold LGBM), overall and per class, with the most separating features.
3. (optional) generated PKS vs real test adversarial AUC, with fake-PKS deg2 vs test as reference.
usage: python src/gen_validate.py GEN_GBT.npz [GEN_PKS.npz]     (npz: X (n,427), y (n,))"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score
B = Path(__file__).resolve().parent.parent / 'bundle'
m = np.load(B / 'meta.npz', allow_pickle=True); y, fo, L, COLS = m['y'], m['folds'], list(m['labels']), list(m['cols'])
NT = 4
PRM = dict(n_estimators=900, learning_rate=0.05, num_leaves=31, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
           class_weight='balanced', verbose=-1, num_threads=NT)


def adv_auc(A, Bm, n_est=200, seed=0, top=6):
    Z = np.vstack([A, Bm]); ya = np.r_[np.zeros(len(A)), np.ones(len(Bm))]
    f5 = np.random.default_rng(seed).integers(0, 5, len(Z)); pa = np.zeros(len(Z)); imp = 0
    for k in range(5):
        mdl = lgb.LGBMClassifier(n_estimators=n_est, verbose=-1, num_threads=NT).fit(Z[f5 != k], ya[f5 != k])
        pa[f5 == k] = mdl.predict_proba(Z[f5 == k])[:, 1]; imp = imp + mdl.booster_.feature_importance('gain')
    return roc_auc_score(ya, pa), pd.Series(imp, COLS).sort_values(ascending=False).head(top).round(0).to_dict()


def gen_on_real(Xg, yg, Xr, n_est=900):
    mdl = lgb.LGBMClassifier(**{**PRM, 'n_estimators': n_est}).fit(Xg, yg)
    p = mdl.predict_proba(Xr); pr = p.argmax(1)
    return p, f1_score(y, pr, average='macro'), dict(zip(L, np.round(f1_score(y, pr, average=None, labels=range(9)), 3))), \
        [round(f1_score(y[fo == k], pr[fo == k], average='macro'), 4) for k in range(5)]


if __name__ == '__main__':
    Xc = np.load(B / 'feats_clean.npz')['X']
    g = np.load(sys.argv[1]); Xg, yg = g['X'], g['y']
    out = []
    P = lambda *a: (print(*a, flush=True), out.append(' '.join(map(str, a))))
    P(f'generated GBT: {len(yg)} cadences, mix {dict(zip(L, np.bincount(yg, minlength=9)))}')
    p, f, pc, pf = gen_on_real(Xg, yg, Xc)
    P(f'[1] train on GENERATED only -> REAL train: macro-F1 {f:.4f}  per fixed fold {pf}')
    P(f'    per-class F1 {pc}')
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, p.argmax(1), labels=range(9))
    P('    confusion (rows=true real, cols=pred):\n' + pd.DataFrame(cm, index=[l[:9] for l in L], columns=[l[:6] for l in L]).to_string())
    auc, top = adv_auc(Xg, Xc); P(f'[2] adversarial AUC generated-GBT vs real train: {auc:.4f}  top {top}')
    for k in range(9):
        a, t = adv_auc(Xg[yg == k], Xc[y == k], n_est=100, top=3); P(f'    {L[k]:20s} AUC {a:.4f}  top {t}')
    if len(sys.argv) > 2:
        Xt = np.load(B / 'feats_test.npz')['X']; Xd = np.load(B / 'feats_deg2.npz')['X']
        gp = np.load(sys.argv[2])
        a1, t1 = adv_auc(gp['X'], Xt); a2, t2 = adv_auc(Xd, Xt)
        P(f'[3] adversarial AUC generated-PKS vs real test: {a1:.4f} top {t1}')
        P(f'    reference fake-PKS deg2 vs real test: {a2:.4f} top {t2}')
    Path('results').mkdir(exist_ok=True)
    open(f'results/validate_{Path(sys.argv[1]).stem}.txt', 'w').write('\n'.join(out) + '\n')
