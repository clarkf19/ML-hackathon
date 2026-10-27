"""Scintillating-star characterisation from the committed feature bundle (raw data unavailable in this cloud session).
Prints numbers used in results/REPORT.md: ON-panel brightening (level, per-panel consistency, frequency/time shape),
texture scale, contamination slices and where scint is lost, on clean / fake-PKS / test renderings.
usage: python src/scint_char.py [probs_npz]   (default: results/probs_baseline.npz from src/bundle_eval.py)"""
import sys, numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
R = Path(__file__).resolve().parent.parent
m = np.load(R / 'bundle/meta.npz', allow_pickle=True); y, fo, L = m['y'], m['folds'], list(m['labels'])
cols = list(m['cols']); ci = {c: i for i, c in enumerate(cols)}
X = {w: np.load(R / f'bundle/feats_{w}.npz')['X'] for w in ['clean', 'deg2', 'deg4', 'test']}
pp = np.load(sys.argv[1] if len(sys.argv) > 1 else R / 'results/probs_baseline.npz')
S, N = 3, 8                                                     # scintillating_star, noise
sc = y == S
def col(w, c): return X[w][:, ci[c]]
def q(a): return np.round(np.percentile(a, [10, 25, 50, 75, 90]), 2).tolist()
P = lambda *a: print(*a, flush=True)

P('==== class sizes', dict(zip(L, np.bincount(y))))
for nm, pr in [('clean', pp['oofc']), ('fakePKS', pp['oofd'])]:
    p = pr.argmax(1); cm = confusion_matrix(y, p)
    P(f'---- {nm}: macro-F1 {f1_score(y, p, average="macro"):.4f}  scint F1 {f1_score(y == S, p == S):.3f}'
      f'  recall {cm[S, S] / cm[S].sum():.3f}  precision {cm[S, S] / cm[:, S].sum():.3f}')
    P('  scint -> predicted as', {L[j]: int(cm[S, j]) for j in range(9) if cm[S, j]})
    P('  predicted scint, true', {L[i]: int(cm[i, S]) for i in range(9) if cm[i, S]})

P('\n==== 1. ON-panel brightening (lvl features, bytes, OFF-mean referenced)')
for w in ['clean', 'deg2', 'deg4']:
    for c in ['lvl_lv_on_off', 'lvl_lv_onmin_offmax', 'lvl_lv_off_spread', 'lvl_lv_on_spread']:
        P(f'  {w:5s} {c:22s} scint {q(col(w, c)[sc])}  noise {q(col(w, c)[y == N])}  pulsar {q(col(w, c)[y == 2])}')
for c in ['lvl_lv_on_off', 'lvl_lv_off_spread', 'lvl_lv_on_spread']:
    P(f'  test  {c:22s} all {q(col("test", c))}   train-all clean {q(col("clean", c))}  deg2 {q(col("deg2", c))}')
for w in ['clean', 'deg2']:
    a = col(w, 'lvl_lv_on_off'); P(f'  AUC lv_on_off scint-vs-noise {w}: {roc_auc_score(sc[(y == S) | (y == N)], a[(y == S) | (y == N)]):.3f}'
                                  f'  scint-vs-rest {roc_auc_score(sc, a):.3f}')
# per-panel consistency: are all three ON panels brighter by the same amount?
for w in ['clean', 'deg2']:
    lv = np.stack([col(w, f'lvl_lv_p{p}') for p in range(6)], 1)       # panel level - OFF mean
    on, off = lv[sc][:, 0::2], lv[sc][:, 1::2]
    r = np.corrcoef(on.T); P(f'  {w} scint ON-panel levels mean {np.round(on.mean(0), 2)} sd {np.round(on.std(0), 2)}; OFF {np.round(off.mean(0), 2)} sd {np.round(off.std(0), 2)}')
    P(f'  {w} scint corr(ON_i, ON_j) {np.round([r[0, 1], r[0, 2], r[1, 2]], 3)}; frac all-3-ON>all-3-OFF {np.mean(on.min(1) > off.max(1)):.3f}'
      f' (noise {np.mean(lv[y == N][:, 0::2].min(1) > lv[y == N][:, 1::2].max(1)):.3f}, null 0.05)')
    rat = on.min(1) / np.maximum(on.max(1), 1e-3); P(f'  {w} scint ON min/max level ratio {q(rat[on.max(1) > 1])}')
# frequency shape of the brightening (8 sub-bands of 16 ch)
for w in ['clean', 'deg2']:
    bd = np.stack([col(w, f'lvl_lv_band{k}') for k in range(8)], 1)
    P(f'  {w} scint ON-OFF level by sub-band (mean) {np.round(bd[sc].mean(0), 2)}  noise {np.round(bd[y == N].mean(0), 2)}')
    P(f'  {w} scint band std/mean |level| {np.round(np.median(bd[sc].std(1) / (np.abs(bd[sc].mean(1)) + 1e-3)), 2)}'
      f'  frac bands>0 {np.round((bd[sc] > 0).mean(), 3)} (noise {np.round((bd[y == N] > 0).mean(), 3)})')
    P(f'  {w} band std: scint {q(col(w, "lvl_lv_band_std")[sc])} noise {q(col(w, "lvl_lv_band_std")[y == N])}')
    P(f'  {w} time-block std ON: scint {q(col(w, "lvl_lv_tblk_std_on")[sc])} noise {q(col(w, "lvl_lv_tblk_std_on")[y == N])};'
      f' OFF scint {q(col(w, "lvl_lv_tblk_std_off")[sc])}')

P('\n==== 2. texture (block-mean variance x blocksize, ON - OFF; white noise = 1): scint vs noise effect size by scale')
for w in ['clean', 'deg2']:
    out = []
    for s in ['2x4', '2x8', '4x8', '4x16', '8x16', '8x32', '16x32', '16x64']:
        a = col(w, f'tex_v{s}_d'); d = (np.median(a[sc]) - np.median(a[y == N])) / (np.subtract(*np.percentile(a[y == N], [75, 25])) + 1e-6)
        au = roc_auc_score(sc[(y == S) | (y == N)], a[(y == S) | (y == N)])
        out.append(f'{s}:{d:+.2f}/{au:.3f}')
    P(f'  {w} (time rows x channels: robust effect / AUC)', ' '.join(out))
    lvA = roc_auc_score(sc[(y == S) | (y == N)], col(w, 'lvl_lv_on_off')[(y == S) | (y == N)])
    P(f'  {w} mean-level AUC {lvA:.3f} vs best texture AUC above -> mean modulation vs variance modulation')

P('\n==== 3. contamination: scint recall by what else is in the cadence (fake-PKS val)')
for nm, pr, w in [('clean', pp['oofc'], 'clean'), ('fakePKS', pp['oofd'], 'deg2')]:
    p = pr.argmax(1)
    sl = {'clean sky': (col(w, 'phys_d0_onmax') < 5) & (col(w, 'phys_dsteep_onmax') < 5) & (col(w, 'phys_bb0_max') < 10),
          'ON zero-drift line (d0_onmax>=5)': col(w, 'phys_d0_onmax') >= 5,
          'OFF line (d0_offmax>=5)': col(w, 'phys_d0_offmax') >= 5,
          'steep sweep (dsteep_onmax>=5)': col(w, 'phys_dsteep_onmax') >= 5,
          'glitch (bb0_max>=10)': col(w, 'phys_bb0_max') >= 10,
          'missing rows': col(w, 'phys_n_missing_rows') > 0}
    for k, msk in sl.items():
        mm = sc & msk
        P(f'  {nm:8s} {k:34s} n={mm.sum():4d} recall {np.mean(p[mm] == S):.3f}  lv_on_off median {np.median(col(w, "lvl_lv_on_off")[mm]):.2f}')
    # recall by brightening strength quartile
    a = col(w, 'lvl_lv_on_off'); b = np.quantile(a[sc], [0, .25, .5, .75, 1])
    P(f'  {nm} recall by lv_on_off quartile', [(round(lo, 1), round(np.mean(p[sc & (a >= lo) & (a <= hi)] == S), 3)) for lo, hi in zip(b[:-1], b[1:])])
    lost = sc & (p != S)
    P(f'  {nm} lost scint: lv_on_off {q(a[lost])} vs found {q(a[sc & (p == S)])}')
    nz = (y == N) & (p == S); P(f'  {nm} noise predicted scint: n={nz.sum()} lv_on_off {q(a[nz])}')

P('\n==== 4. test vs fake-PKS: level statistic distribution (test labels unknown)')
for c in ['lvl_lv_on_off', 'lvl_lv_onmin_offmax']:
    for th in (2, 3, 4):
        P(f'  frac {c} > {th}: clean {np.mean(col("clean", c) > th):.3f}  deg2 {np.mean(col("deg2", c) > th):.3f}  deg4 {np.mean(col("deg4", c) > th):.3f}  test {np.mean(col("test", c) > th):.3f}')
P('  scint share: train', np.round(sc.mean(), 3), ' test predicted (harness)', np.round(np.mean(pp['pte'].argmax(1) == S), 3))
