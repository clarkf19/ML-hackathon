"""Co-occurrence table: rate of thresholded 'source present' indicators per class, from REAL bundle features.
Also usable on generated features: python src/gen_cooc.py [gen_feats.npz]  (npz with X, y)"""
import sys, numpy as np, pandas as pd
from pathlib import Path
B = Path(__file__).resolve().parent.parent / 'bundle'
m = np.load(B / 'meta.npz', allow_pickle=True); COLS = list(m['cols']); L = list(m['labels'])

def indicators(X):
    X = pd.DataFrame(X, columns=COLS)
    return pd.DataFrame({
        'line_on(k1>4)': X.rep_rep_on_k1 > 4,
        'line_off(k1>4)': X.rep_rep_off_k1 > 4,
        'pers6(k6>3)': X.rep_rep_k6 > 3,
        'eti_trk(obj>2)': X.trk_t_obj > 2,
        'steep(>4)': np.maximum(X.phys_dsteep_onmax, X.phys_dsteep_offmax) > 4,
        'glitch(bb0>10)': X.phys_bb0_max > 10,
        'disp_on(bbD>8)': X.phys_bbD_onmax > 8,
        'disp_off(bbD>8)': X.phys_bbD_offmax > 8,
        'npulse_on6>=2': X.phys_npulse_on6 >= 2,
        'lvl_on(>2)': X.lvl_lv_on_off > 2,
        'tex16x64>1.5': X.tex_v16x64_d > 1.5,
        'radar_acf': X.rep_acf_on_t2_wide > 0.15,
        'missing>0': X.phys_n_missing_rows > 0,
    })

def table(X, y):
    I = indicators(X); t = I.groupby(np.asarray(L)[y]).mean().reindex(L)
    t.loc['ALL'] = I.mean(); return t.round(3)

if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    y = m['y']; Xc = np.load(B / 'feats_clean.npz')['X']
    print('REAL train (GBT)'); print(table(Xc, y).to_string())
    It = indicators(np.load(B / 'feats_test.npz')['X']).mean().round(3)
    Id = indicators(np.load(B / 'feats_deg2.npz')['X']).mean().round(3)
    print('\nALL real test (PKS):'); print(pd.DataFrame({'test': It, 'fakePKS deg2': Id}).T.to_string())
    for f in sys.argv[1:]:
        z = np.load(f); print(f'\n{f}'); print(table(z['X'], z['y']).to_string())
