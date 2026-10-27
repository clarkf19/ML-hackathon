"""Robustness-aware residual stacking: stacker inputs = [log current-model probs, raw statistics]; trained on clean rows
([oofc, rs_train]) + fake-PKS rows ([oofd, rs_d2q]) of folds != k, scored on fold k clean AND fake-PKS rows.
Per group gain vs probs-only stacker, and per-feature gain importance -> results/scan_fpks.txt, scan_fpks_importance.csv
usage: python src/residual_scan_fpks.py [DEGSUFFIX(q|'')]"""
import numpy as np, pandas as pd, lightgbm as lgb, sys
from pathlib import Path
from sklearn.metrics import f1_score, log_loss
ROOT = Path(__file__).resolve().parent.parent; D = ROOT / 'data'
sfx = sys.argv[1] if len(sys.argv) > 1 else 'q'
m = np.load(ROOT / 'bundle/meta.npz', allow_pickle=True); y, fo = m['y'], m['folds']
Fc = np.load(D / 'rs_train.npz'); cols = list(Fc['cols']); Fc = Fc['X']; Fd = np.load(D / f'rs_d2{sfx}.npz')['X']
lc, ld = np.log(np.clip(m['oofc'], 1e-6, 1)), np.log(np.clip(m['oofd'], 1e-6, 1))
shift = pd.read_csv(ROOT / 'results/shift_auc.csv').set_index('feat')[f'shift_d2{sfx}']
grp = lambda c: next((g for p, g in [('px_', 'G1px'), ('cad_', 'G1px'), ('lin_', 'G2lin'), ('row', 'G3row'), ('chL', 'G4ch'), ('chP', 'G4ch'),
                                      ('dch', 'G5dch'), ('blk', 'G6blk'), ('cc', 'G8corr'), ('edge', 'G9edge'), ('c64', 'G9edge'), ('q_', 'G10quant'),
                                      ('sk', 'G11sk'), ('pac', 'G12pac'), ('fup', 'G13tail'), ('fdn', 'G13tail')] if c.startswith(p)), 'G7glitch')
rob = [j for j, c in enumerate(cols) if shift[c] < 0.1]
G = {}
for j in rob: G.setdefault(grp(cols[j]), []).append(j)
prm = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, colsample_bytree=0.5, subsample=0.8,
           subsample_freq=1, class_weight='balanced', verbose=-1, num_threads=4)
def stack(ix):
    oc = np.zeros((len(y), 9)); od = np.zeros((len(y), 9)); imp = np.zeros(9 + len(ix))
    Xc, Xd = np.hstack([lc, Fc[:, ix]]), np.hstack([ld, Fd[:, ix]])
    for k in range(5):
        t, v = fo != k, fo == k
        md = lgb.LGBMClassifier(**prm).fit(np.vstack([Xc[t], Xd[t]]), np.r_[y[t], y[t]])
        oc[v] = md.predict_proba(Xc[v]); od[v] = md.predict_proba(Xd[v]); imp += md.booster_.feature_importance('gain')
    f = lambda o: f1_score(y, o.argmax(1), average='macro')
    return f(oc), f(od), log_loss(y, oc), log_loss(y, od), imp[9:]
lines = []
c0, d0, lc0, ld0, _ = stack([]); lines.append(f'probs only            clean {c0:.4f} fPKS {d0:.4f}  logloss {lc0:.4f} / {ld0:.4f}'); print(lines[-1], flush=True)
ca, da, lca, lda, imp = stack(rob); lines.append(f'+ all robust {len(rob):3d}      clean {ca:.4f} ({ca - c0:+.4f}) fPKS {da:.4f} ({da - d0:+.4f})  logloss {lca - lc0:+.4f} / {lda - ld0:+.4f}')
print(lines[-1], flush=True)
pd.Series(imp, index=[cols[j] for j in rob]).sort_values(ascending=False).to_csv(ROOT / 'results/scan_fpks_importance.csv')
for g, ix in sorted(G.items()):
    cg, dg, lcg, ldg, _ = stack(ix)
    lines.append(f'+ {g:9s} {len(ix):3d}        clean {cg:.4f} ({cg - c0:+.4f}) fPKS {dg:.4f} ({dg - d0:+.4f})  logloss {lcg - lc0:+.4f} / {ldg - ld0:+.4f}')
    print(lines[-1], flush=True)
open(ROOT / f'results/scan_fpks{sfx}.txt', 'w').write('\n'.join(lines) + '\n')
