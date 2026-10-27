"""Probability-matched residual AUC: for each confused pair (a,b), cadences with true in {a,b} are binned by the current
model's pairwise log-odds log(p_a/p_b) (8 quantile bins of the ambiguous range |log-odds| < 4); within each bin, AUC of the
statistic for true a vs true b; weighted mean over bins. If the statistic adds nothing beyond the model's belief the
expected value is 0.5. Bundle features scored the same way as reference. -> results/scan_auc2.csv, scan_auc2_top.txt"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parent.parent
m = np.load(ROOT / 'bundle/meta.npz', allow_pickle=True); y, oofc = m['y'], m['oofc']
z = np.load(ROOT / 'data/rs_train.npz'); F, cols = z['X'], list(z['cols'])
Bx = np.load(ROOT / 'bundle/feats_clean.npz')['X']; bcols = list(m['cols'])
S = ['eti', 'frb', 'puls', 'scint', 'sat', 'radar', 'rfi_int', 'rfi_pers', 'noise']
pairs = [(3, 8), (6, 7), (3, 7), (3, 4), (0, 6), (3, 5), (4, 5), (5, 7), (4, 7), (5, 8)]
rows = []
for a, b in pairs:
    k = np.where(((y == a) | (y == b)))[0]; lo = np.log(oofc[k, a] + 1e-6) - np.log(oofc[k, b] + 1e-6)
    k, lo = k[np.abs(lo) < 4], lo[np.abs(lo) < 4]
    bins = np.digitize(lo, np.quantile(lo, np.linspace(0, 1, 9)[1:-1])); t = y[k] == a
    for src, M, cn in [('raw', F, cols), ('bundle', Bx, bcols)]:
        for j, c in enumerate(cn):
            num = den = 0.
            for q in range(8):
                s = bins == q; tq = t[s]; w = min(tq.sum(), (~tq).sum())
                if w >= 10 and M[k[s], j].std() > 0: num += w * (roc_auc_score(tq, M[k[s], j]) - 0.5); den += w
            rows.append(dict(pair=f'{S[a]}|{S[b]}', n=len(k), src=src, feat=c, auc=num / den + 0.5 if den else np.nan))
R = pd.DataFrame(rows); R['dev'] = (R.auc - 0.5).abs(); R.to_csv(ROOT / 'results/scan_auc2.csv', index=False)
with open(ROOT / 'results/scan_auc2_top.txt', 'w') as fh:
    for p, g in R.groupby('pair', sort=False):
        bd = g[g.src == 'bundle'].sort_values('dev', ascending=False); r = g[g.src == 'raw'].sort_values('dev', ascending=False)
        fh.write(f'\n== {p} (n ambiguous {g.n.iloc[0]})  bundle dev p50/p90/max {bd.dev.median():.3f}/{bd.dev.quantile(.9):.3f}/{bd.dev.max():.3f} '
                 f'top bundle {list(zip(bd.feat[:3], bd.auc[:3].round(3)))}\n')
        fh.write(r.head(10)[['feat', 'auc']].round(3).to_string(index=False) + '\n')
print(open(ROOT / 'results/scan_auc2_top.txt').read())
