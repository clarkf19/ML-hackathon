"""Which candidate statistics carry information the current model LACKS? Uses bundle/meta.npz oofc / oofd (current best
model's OOF probabilities on clean val / fake-PKS val) and the fixed folds.
 A  pair AUC: for each confused pair (true a -> predicted b, >= MINN cases): AUC of every statistic separating the MISSED a's
    from the CORRECT b's they were mistaken for (AUC_M), next to AUC on correct a vs correct b (AUC_C). Features the model
    already uses are pushed toward AUC_M ~ 0.5 by selection, so the reference is the same AUC_M distribution of the 427
    bundle features (null): a statistic is a candidate when |AUC_M - .5| beats the null's 95th percentile.
 B  stacker: LightGBM on logit(oof) alone vs logit(oof) + one group of statistics (fixed folds) -> OOF logloss / macro-F1 gain.
 C  error predictor: LightGBM predicting (argmax != y) from confidence alone vs + statistics; AUC and top-gain statistics.
 Each part runs on clean (oofc + X_clean) and fake-PKS (oofd + X_deg2).
usage: python src/resid_scan.py stats.npz [tag]   (npz: X_clean, X_deg2, names) -> results/resid_{tag}_*.csv"""
import os, sys, re, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import f1_score, log_loss, roc_auc_score
R = Path(__file__).resolve().parent.parent
m = np.load(R / 'bundle/meta.npz', allow_pickle=True); y, fo, L, cols = m['y'], m['folds'], list(m['labels']), list(m['cols'])
src = Path(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else src.stem
z = np.load(src, allow_pickle=True); names = list(z['names'])
OUT = R / 'results'; OUT.mkdir(exist_ok=True)
MINN = 30
NT = int(os.environ.get('NT', 4))   # threads per LightGBM fit (use NT=1 OMP_WAIT_POLICY=PASSIVE next to another job)
SH = {'eti_candidate': 'eti', 'frb': 'frb', 'pulsar': 'psr', 'scintillating_star': 'scint', 'satellite_pass': 'sat',
      'radar_chirp': 'radar', 'rfi_intermittent': 'rfi_int', 'rfi_persistent': 'rfi_pers', 'noise': 'noise'}

def auc_cols(X, pos):
    """column-wise ROC AUC of X (n,k) for binary pos (n,), ties averaged."""
    r = rankdata(X, axis=0); n1 = pos.sum(); n0 = len(pos) - n1
    return (r[pos].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)

def pair_auc(dom, P, Xn, Xb):
    pred = P.argmax(1); rows = []
    for a in range(9):
        for b in range(9):
            miss = (y == a) & (pred == b)
            if a == b or miss.sum() < MINN: continue
            cb, ca = (y == b) & (pred == b), (y == a) & (pred == a)
            sm, sc = np.r_[np.where(miss)[0], np.where(cb)[0]], np.r_[np.where(ca)[0], np.where(cb)[0]]
            am, ac = auc_cols(Xn[sm], y[sm] == a), auc_cols(Xn[sc], y[sc] == a)
            null = np.abs(auc_cols(Xb[sm], y[sm] == a) - .5)
            q95 = np.quantile(null, .95)
            for j in range(len(names)):
                rows.append((dom, f'{SH[L[a]]}->{SH[L[b]]}', miss.sum(), names[j], am[j], ac[j], abs(am[j] - .5), q95, null.max()))
    return pd.DataFrame(rows, columns=['dom', 'pair', 'n_missed', 'stat', 'auc_missed_vs_correct_pred', 'auc_correct',
                                       'effect', 'null_q95', 'null_max'])

def logit(P): P = np.clip(P, 1e-6, 1); return np.log(P)

def cv_lgb(F, target, prm):
    o = np.zeros((len(target), 9 if prm.get('objective') == 'multiclass' else 1))
    for k in range(5):                      # capacity chosen by early stopping on the next fold (inside the training part)
        t, v, s = (fo != k) & (fo != (k + 1) % 5), fo == k, fo == (k + 1) % 5
        mdl = lgb.LGBMModel(**prm).fit(F[t], target[t], eval_set=[(F[s], target[s])],
                                       callbacks=[lgb.early_stopping(40, verbose=False)])
        p = mdl.predict(F[v]); o[v] = p if p.ndim == 2 else p[:, None]
    return o

SPRM = dict(objective='multiclass', num_class=9, n_estimators=1000, learning_rate=0.04, num_leaves=15, min_child_samples=40,
            colsample_bytree=0.8, subsample=0.8, subsample_freq=1, verbose=-1, num_threads=NT)
EPRM = dict(objective='binary', n_estimators=1000, learning_rate=0.03, num_leaves=15, min_child_samples=40, colsample_bytree=0.5,
            subsample=0.8, subsample_freq=1, verbose=-1, num_threads=NT)

KINDS = ['on2_offmax', 'zadjmin', 'adjmin', 'adjn', 'onrank', 'zalt', 'onslope', 'offslope', 'slope', 'bump', 'cv', 'argmax']

def groups():
    g = {}
    for j, nm in enumerate(names):          # bs_{family}_{kind} (bundle_stats.py) or {prefix}_{group}_{detail} (raw_stats.py)
        kind = next((k for k in KINDS if nm.endswith('_' + k)), None) if nm.startswith('bs_') else None
        fam_ = nm[3:-len(kind) - 1] if kind else nm.split('_')[1]
        if nm.startswith('bs_corr_'): fam_, kind = 'corr_lv_tex', None
        g.setdefault('fam:' + fam_, []).append(j)
        if kind: g.setdefault('kind:' + kind, []).append(j)
    g['ALL'] = list(range(len(names)))
    return {k: v for k, v in g.items() if len(v) >= 1}

if __name__ == '__main__':
    doms = [('clean', m['oofc'], z['X_clean'], np.load(R / 'bundle/feats_clean.npz')['X']),
            ('fakePKS', m['oofd'], z['X_deg2'], np.load(R / 'bundle/feats_deg2.npz')['X'])]
    # A
    A = pd.concat([pair_auc(d, P, Xn, Xb) for d, P, Xn, Xb in doms])
    A.to_csv(OUT / f'resid_{tag}_pairauc.csv', index=False, float_format='%.4f')
    for (d, pr), g in A.groupby(['dom', 'pair'], sort=False):
        g = g.sort_values('effect', ascending=False)
        top = ', '.join(f'{r.stat}={r.auc_missed_vs_correct_pred:.3f}' for r in g.head(3).itertuples())
        print(f'A {d:7s} {pr:16s} n={g.n_missed.iloc[0]:4d} null q95 {g.null_q95.iloc[0]:.3f} max {g.null_max.iloc[0]:.3f} | '
              f'n>q95 {(g.effect > g.null_q95).sum():3d} n>max {(g.effect > g.null_max).sum():3d} | top {top}', flush=True)
    # B + C
    G = groups(); rowsB, rowsC = [], []
    for d, P, Xn, _ in doms:
        base = logit(P)
        conf = np.c_[P.max(1), np.sort(P, 1)[:, -1] - np.sort(P, 1)[:, -2], P.argmax(1)]
        err = (P.argmax(1) != y).astype(int)
        res = {}
        for gname, idx in [('NONE', [])] + list(G.items()):
            o = cv_lgb(np.c_[base, Xn[:, idx]], y, SPRM)
            res[gname] = (log_loss(y, o), f1_score(y, o.argmax(1), average='macro'))
            rowsB.append((d, gname, len(idx), *res[gname], res[gname][0] - res['NONE'][0], res[gname][1] - res['NONE'][1]))
            print(f'B {d:7s} {gname:28s} k={len(idx):3d} logloss {res[gname][0]:.4f} mF1 {res[gname][1]:.4f} '
                  f'd_ll {res[gname][0] - res["NONE"][0]:+.4f} d_f1 {res[gname][1] - res["NONE"][1]:+.4f}', flush=True)
        e0 = cv_lgb(conf, err, EPRM)[:, 0]; Fe = np.c_[conf, Xn]
        e1 = cv_lgb(Fe, err, EPRM)[:, 0]
        gain = lgb.LGBMModel(**{**EPRM, 'n_estimators': 300}).fit(Fe, err).booster_.feature_importance('gain')[3:]
        o = np.argsort(-gain)[:25]
        rowsC += [(d, names[j], gain[j] / gain.sum()) for j in o]
        print(f'C {d:7s} error AUC conf-only {roc_auc_score(err, e0):.4f}  + stats {roc_auc_score(err, e1):.4f}  top: '
              + ', '.join(names[j] for j in o[:8]), flush=True)
    pd.DataFrame(rowsB, columns=['dom', 'group', 'k', 'logloss', 'macroF1', 'd_logloss', 'd_macroF1']).to_csv(
        OUT / f'resid_{tag}_stacker.csv', index=False, float_format='%.5f')
    pd.DataFrame(rowsC, columns=['dom', 'stat', 'gain_share']).to_csv(OUT / f'resid_{tag}_errpred.csv', index=False, float_format='%.4f')
