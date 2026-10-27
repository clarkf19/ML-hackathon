"""Is panel gain jitter independent per panel, or does it drift slowly (so neighbouring OFF panels would predict an ON panel's
gain better than the OFF mean)? Correlation of panel levels (lvl_lv_p*, OFF-mean referenced) in level-neutral classes.
With i.i.d. gains every pair of ON panels has the same correlation (shared OFF reference); drift makes adjacent pairs
higher. usage: python src/scint_gain.py"""
import numpy as np
from pathlib import Path
R = Path(__file__).resolve().parent.parent
m = np.load(R / 'bundle/meta.npz', allow_pickle=True); y = m['y']; ci = {c: i for i, c in enumerate(m['cols'])}
neutral = np.isin(y, [4, 5, 6, 7, 8])                                     # satellite, radar, rfi_int, rfi_pers, noise
for w in ['clean', 'deg2', 'test']:
    X = np.load(R / f'bundle/feats_{w}.npz')['X']; lv = np.stack([X[:, ci[f'lvl_lv_p{p}']] for p in range(6)], 1)
    if w != 'test': lv = lv[neutral]
    else: lv = lv[np.abs(X[:, ci['lvl_lv_on_off']]) < 2]              # test: unlabeled, drop strongly brightened cadences
    c = np.corrcoef(lv.T)
    print(f'{w:5s} n={len(lv):5d} corr ON pairs (0,2) {c[0, 2]:.3f} (2,4) {c[2, 4]:.3f} (0,4) {c[0, 4]:.3f} | adjacent ON-OFF (0,1) {c[0, 1]:.3f}'
          f' (2,3) {c[2, 3]:.3f} (4,5) {c[4, 5]:.3f} vs distant (0,5) {c[0, 5]:.3f} | OFF (1,3) {c[1, 3]:.3f} (3,5) {c[3, 5]:.3f} (1,5) {c[1, 5]:.3f}'
          f' | sd per panel {np.round(lv.std(0), 2)}')
# decomposition: OFF-referenced levels = i.i.d. panel gain (sd g) + a COMMON ON offset (sd c) shared by the 3 ON panels:
#   var(OFF_rel) = 2/3 g^2 ; var(lv_on_off) = 2/3 g^2 + c^2   -> c
# Same selection on fake-PKS val and on test: cadences the harness model predicts as noise / rfi (no labels used on test).
pp = np.load(R / 'results/probs_baseline.npz'); W = {'clean': pp['oofc'], 'deg2': pp['oofd'], 'test': pp['pte']}
for w, P in W.items():
    X = np.load(R / f'bundle/feats_{w}.npz')['X']; lv = np.stack([X[:, ci[f'lvl_lv_p{p}']] for p in range(6)], 1)
    for nm, sel in [('pred noise', P.argmax(1) == 8), ('pred rfi_int/pers', np.isin(P.argmax(1), [6, 7]))]:
        a = lv[sel]; offv = a[:, 1::2].var(0).mean(); g = np.sqrt(offv * 1.5); d = a[:, 0::2].mean(1)
        c = np.sqrt(max(d.var() - 2 / 3 * g ** 2, 0)); r = np.corrcoef(a[:, 0::2].T)
        print(f'{w:5s} {nm:17s} n={sel.sum():5d}: panel gain sd {g:.2f}, common-ON offset sd {c:.2f}, lv_on_off sd {d.std():.2f},'
              f' ON-ON corr {np.mean([r[0, 1], r[0, 2], r[1, 2]]):.3f}')
