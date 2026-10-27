"""Calibration report: per-class quantiles (10/50/90) of key features, REAL train vs GENERATED, and the features whose
per-class distributions differ most (KS statistic). For PKS files: compare the class-pooled distribution to real test.
usage: python src/gen_compare.py GEN.npz [--pks] [--cols=a,b,c] [--worst=15]"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp
B = Path(__file__).resolve().parent.parent / 'bundle'
m = np.load(B / 'meta.npz', allow_pickle=True); y, L, COLS = m['y'], list(m['labels']), list(m['cols'])
KEY = ['phys_n_missing_rows', 'phys_bb0_max', 'lvl_lv_off_spread', 'lvl_lv_on_off', 'phys_spec_max', 'trk_t_obj', 'trk_t_drift',
       'rep_rep_on_k1', 'rep_rep_off_k1', 'rep_rep_k6', 'phys_dsteep_onmax', 'phys_bbD_onmax', 'phys_npulse_on6', 'phys_best_s',
       'tex_v8x32_d', 'rep_acf_on_t0_c1', 'rep_acf_on_t2_wide', 'lvl_lv_tblk_std_on']

if __name__ == '__main__':
    g = np.load([a for a in sys.argv[1:] if not a.startswith('--')][0]); Xg = pd.DataFrame(g['X'], columns=COLS); yg = g['y']
    pks = '--pks' in sys.argv
    cols = next((a.split('=')[1].split(',') for a in sys.argv if a.startswith('--cols=')), KEY)
    nw = int(next((a.split('=')[1] for a in sys.argv if a.startswith('--worst=')), 15))
    q = lambda v: '/'.join(f'{x:.2f}' for x in np.quantile(v, [.1, .5, .9])) if len(v) else '-'
    if pks:
        Xr = pd.DataFrame(np.load(B / 'feats_test.npz')['X'], columns=COLS)
        Xd = pd.DataFrame(np.load(B / 'feats_deg2.npz')['X'], columns=COLS)
        if (yg == 8).all():   # noise-only PKS sample: compare with test cadences confidently predicted noise (best model)
            pte = m['pte']; sel = (pte.argmax(1) == 8) & (pte.max(1) > 0.9); Xr = Xr[sel]; Xd = Xd[y == 8]
            print(f'noise-only: real = {sel.sum()} test cadences predicted noise (p>0.9); deg2 = fake-PKS of real noise class')
        for c in cols: print(f'{c:28s} test {q(Xr[c]):24s} deg2 {q(Xd[c]):24s} gen {q(Xg[c])}')
        ks = pd.Series({c: ks_2samp(Xr[c], Xg[c]).statistic for c in COLS}).sort_values(ascending=False)
        kd = pd.Series({c: ks_2samp(Xr[c], Xd[c]).statistic for c in COLS})
        print('worst KS gen-vs-test (deg2-vs-test in brackets):', {c: f'{ks[c]:.2f} ({kd[c]:.2f})' for c in ks.index[:nw]})
        sys.exit()
    Xr = pd.DataFrame(np.load(B / 'feats_clean.npz')['X'], columns=COLS)
    present = sorted(set(yg.tolist()))
    for c in cols:
        if Xg[c].isna().all(): continue
        print(c)
        for k in present: print(f'   {L[k]:20s} real {q(Xr[c][y == k]):24s} gen {q(Xg[c][yg == k])}')
    have = [c for c in COLS if Xg[c].notna().all()]
    print(f'\nmean KS over {len(have)} computed features per class, and worst features:')
    for k in range(9):
        if not (yg == k).any(): continue
        ks = pd.Series({c: ks_2samp(Xr[c][y == k], Xg[c][yg == k]).statistic for c in have}).sort_values(ascending=False)
        print(f'  {L[k]:20s} meanKS {ks.mean():.3f}  worst {dict(ks.head(6).round(2))}')
