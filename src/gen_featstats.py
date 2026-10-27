"""Feature-space statistics of REAL train (per class) and REAL test from the repo bundle. Used to calibrate the generator
when raw pixels are unavailable.  usage: python src/gen_featstats.py [col ...]  -> prints 10/50/90% quantiles"""
import sys, numpy as np, pandas as pd
from pathlib import Path
B = Path(__file__).resolve().parent.parent / 'bundle'
m = np.load(B / 'meta.npz', allow_pickle=True); cols = list(m['cols']); L = list(m['labels']); y = m['y']
Xc = pd.DataFrame(np.load(B / 'feats_clean.npz')['X'], columns=cols); Xt = pd.DataFrame(np.load(B / 'feats_test.npz')['X'], columns=cols)
Xd = pd.DataFrame(np.load(B / 'feats_deg2.npz')['X'], columns=cols)

if __name__ == '__main__':
    print('train mix', dict(zip(L, np.round(np.bincount(y) / len(y), 4))))
    print('gap train', pd.Series(m['train_gap']).value_counts().sort_index().to_dict(), 'test', pd.Series(m['test_gap']).value_counts().sort_index().to_dict())
    print('missing rows train', Xc.phys_n_missing_rows.value_counts().sort_index().to_dict())
    print('missing rows test ', Xt.phys_n_missing_rows.value_counts().sort_index().to_dict())
    print('mean missing rows train/deg2/test', Xc.phys_n_missing_rows.mean(), Xd.phys_n_missing_rows.mean(), Xt.phys_n_missing_rows.mean())
    qs = [.1, .5, .9]
    for c in sys.argv[1:] or ['phys_bb0_max', 'phys_spec_max', 'lvl_lv_on_off', 'lvl_lv_off_spread', 'trk_t_obj', 'phys_best_s']:
        print(c)
        for k in range(9): print(f'   {L[k]:20s}', np.round(np.quantile(Xc[c][y == k], qs), 2).tolist())
        print(f'   {"ALL train":20s}', np.round(np.quantile(Xc[c], qs), 2).tolist())
        print(f'   {"ALL deg2":20s}', np.round(np.quantile(Xd[c], qs), 2).tolist())
        print(f'   {"TEST":20s}', np.round(np.quantile(Xt[c], qs), 2).tolist())
