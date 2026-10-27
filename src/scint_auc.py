"""Single-feature AUCs (scint-vs-noise, scint-vs-rest) on clean and fake-PKS renderings for new feature files vs the
strongest existing bundle scint features.  usage: python src/scint_auc.py results/feats_X.npz [...]"""
import sys, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
R = Path(__file__).resolve().parent.parent
m = np.load(R / 'bundle/meta.npz', allow_pickle=True); y = m['y']; ci = {c: i for i, c in enumerate(m['cols'])}
S = 3; sn = (y == S) | (y == 8)
REF = ['lvl_lv_on_off', 'lvl_lv_onmin_offmax', 'tex_v4x16_d', 'tex_v8x32_d', 'phys_tex_std_on_off']
for w in ['clean', 'deg2']:
    Xb = np.load(R / f'bundle/feats_{w}.npz')['X']; res = [(c, Xb[:, ci[c]]) for c in REF]
    for fn in sys.argv[1:]:
        d = np.load(fn); res += [(c, d['X_' + w][:, j]) for j, c in enumerate(d['cols'])]
    print('==', w)
    for c, a in res:
        a = np.nan_to_num(a); print(f'  {c:26s} scint-vs-noise {roc_auc_score(y[sn] == S, a[sn]):.3f}  scint-vs-rest {roc_auc_score(y == S, a):.3f}')
