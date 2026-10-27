"""Which raw statistics carry information the current model (bundle/meta.npz oofc = A_lvl OOF probs) lacks?
A) conditional AUC per confused class pair (a,b): within each predicted-class stratum (pred==a, pred==b), AUC of the
   statistic for true a vs true b, averaged over the two strata (= separation the model's decision does NOT explain),
   vs the unconditional AUC among correctly classified a and b. Bundle features scored the same way as a reference;
   split-half (folds 0-1 vs 2-4) stability reported.
B) residual stacking: LightGBM on [log oofc] vs [log oofc + statistic group], fixed folds -> macro-F1 / logloss gain,
   and an error predictor (is the current argmax wrong?) with / without the group.
usage: python src/residual_scan.py [rs_train.npz]   -> results/scan_auc.csv, results/scan_stack.txt"""
import numpy as np, pandas as pd, lightgbm as lgb, sys
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, log_loss
ROOT = Path(__file__).resolve().parent.parent
m = np.load(ROOT / 'bundle/meta.npz', allow_pickle=True); y, fo, L, oofc = m['y'], m['folds'], list(m['labels']), m['oofc']
z = np.load(ROOT / 'data' / (sys.argv[1] if len(sys.argv) > 1 else 'rs_train.npz')); F, cols = z['X'], list(z['cols'])
Bx = np.load(ROOT / 'bundle/feats_clean.npz')['X']; bcols = list(m['cols'])
out = ROOT / 'results'; out.mkdir(exist_ok=True)
pred = oofc.argmax(1); S = ['eti', 'frb', 'puls', 'scint', 'sat', 'radar', 'rfi_int', 'rfi_pers', 'noise']
cm = np.zeros((9, 9), int); np.add.at(cm, (y, pred), 1)
pairs = sorted({tuple(sorted((a, b))) for a in range(9) for b in range(9) if a != b}, key=lambda p: -(cm[p] + cm[p[::-1]]))[:10]
print('top confused pairs:', [(S[a], S[b], cm[a, b], cm[b, a]) for a, b in pairs])

def auc(x, t):
    if t.sum() < 5 or (~t).sum() < 5: return np.nan
    x = np.nan_to_num(x, nan=np.nanmedian(x)); return roc_auc_score(t, x)

def cond_auc(x, a, b, sel):
    """mean over strata pred==a / pred==b of AUC(true a vs true b); weights = min class count in the stratum"""
    num = den = 0.
    for s in (a, b):
        k = sel & (pred == s) & ((y == a) | (y == b)); t = y[k] == a
        w = min(t.sum(), (~t).sum())
        if w >= 5: num += w * (auc(x[k], t) - 0.5); den += w
    return num / den + 0.5 if den else np.nan

rows = []
half = fo < 2
for a, b in pairs:
    corr = (pred == y) & ((y == a) | (y == b))
    for src, M, cn in [('raw', F, cols), ('bundle', Bx, bcols)]:
        for j, c in enumerate(cn):
            x = M[:, j]
            ca = cond_auc(x, a, b, np.ones(len(y), bool))
            rows.append(dict(pair=f'{S[a]}|{S[b]}', n_err=cm[a, b] + cm[b, a], src=src, feat=c, cond=ca, cond_dev=abs(ca - 0.5),
                             c_h1=cond_auc(x, a, b, half), c_h2=cond_auc(x, a, b, ~half), corr_auc=auc(x[corr], y[corr] == a)))
R = pd.DataFrame(rows); R['stable'] = ((R.c_h1 - 0.5) * (R.c_h2 - 0.5) > 0) & (np.minimum(abs(R.c_h1 - 0.5), abs(R.c_h2 - 0.5)) > 0.03)
R.to_csv(out / 'scan_auc.csv', index=False)
with open(out / 'scan_auc_top.txt', 'w') as fh:
    for p, g in R.groupby('pair', sort=False):
        ref = g[g.src == 'bundle'].cond_dev; q = np.nanpercentile(ref, [50, 90, 99])
        top = g[g.src == 'raw'].sort_values('cond_dev', ascending=False).head(12)
        fh.write(f'\n== {p}  (errors {g.n_err.iloc[0]})  bundle-feature cond|AUC-0.5| p50/p90/p99 = {q.round(3)}  max {ref.max():.3f}\n')
        fh.write(top[['feat', 'cond', 'c_h1', 'c_h2', 'stable', 'corr_auc']].round(3).to_string(index=False) + '\n')
print(open(out / 'scan_auc_top.txt').read())

# B) residual stacking
lo = np.log(np.clip(oofc, 1e-6, 1))
G = {}
for c in cols:
    g = c.split('_')[0]
    g = {'px': 'G1px', 'cad': 'G1px', 'lin': 'G2lin', 'rowL': 'G3row', 'rowP': 'G3row', 'chL': 'G4ch', 'chP': 'G4ch', 'dchL': 'G5dch',
         'dchP': 'G5dch', 'drop': 'G7glitch', 'glitch5': 'G7glitch', 'glitch10': 'G7glitch', 'dip5': 'G7glitch', 'glitchamp': 'G7glitch',
         'rowq75': 'G7glitch', 'v1': 'G7glitch', 'vmax': 'G7glitch', 'cc': 'G8corr', 'cchp': 'G8corr', 'cclp': 'G8corr', 'cc2d': 'G8corr',
         'edge': 'G9edge', 'edgeD': 'G9edge', 'c64': 'G9edge', 'q': 'G10quant', 'skc': 'G11sk', 'skr': 'G11sk', 'pac': 'G12pac',
         'fup24': 'G13tail', 'fup48': 'G13tail', 'fdn36': 'G13tail'}.get(g, 'G6blk' if g.startswith('blk') else 'other')
    G.setdefault(g, []).append(cols.index(c))
prm = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, colsample_bytree=0.5, subsample=0.8,
           subsample_freq=1, class_weight='balanced', verbose=-1, num_threads=4)
def stack(Xs):
    o = np.zeros((len(y), 9)); e = np.zeros(len(y)); err = (pred != y).astype(int); imp = np.zeros(Xs.shape[1])
    for k in range(5):
        t, v = fo != k, fo == k
        md = lgb.LGBMClassifier(**prm).fit(Xs[t], y[t]); o[v] = md.predict_proba(Xs[v]); imp += md.booster_.feature_importance('gain')
        me = lgb.LGBMClassifier(**{**prm, 'class_weight': None}).fit(Xs[t], err[t]); e[v] = me.predict_proba(Xs[v])[:, 1]
    return f1_score(y, o.argmax(1), average='macro'), log_loss(y, o), roc_auc_score(err, e), imp
lines = []
f0, l0, e0, _ = stack(lo); lines.append(f'stack [oofc only]           mF1 {f0:.4f}  logloss {l0:.4f}  err-AUC {e0:.4f}   (raw oofc argmax mF1 {f1_score(y, pred, average="macro"):.4f})')
print(lines[-1], flush=True)
f1a, l1a, e1a, imp = stack(np.hstack([lo, F])); lines.append(f'stack [oofc + ALL raw {F.shape[1]}]  mF1 {f1a:.4f}  logloss {l1a:.4f}  err-AUC {e1a:.4f}')
print(lines[-1], flush=True)
I = pd.Series(imp[9:], index=cols).sort_values(ascending=False); I.to_csv(out / 'scan_stack_importance.csv')
for g, ix in G.items():
    fg, lg, eg, _ = stack(np.hstack([lo, F[:, ix]]))
    lines.append(f'stack [oofc + {g:9s} {len(ix):3d}]  mF1 {fg:.4f} ({fg - f0:+.4f})  logloss {lg:.4f} ({lg - l0:+.4f})  err-AUC {eg:.4f} ({eg - e0:+.4f})')
    print(lines[-1], flush=True)
lines.append('top-40 raw features by stacking gain: ' + ', '.join(I.index[:40]))
open(out / 'scan_stack.txt', 'w').write('\n'.join(lines) + '\n'); print(lines[-1])
