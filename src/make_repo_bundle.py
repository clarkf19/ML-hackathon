"""Compact feature bundle committed to the repo for cloud sessions: bundle/feats_{clean,test,deg2,deg4}.npz (float32),
bundle/meta.npz (y, folds, column names, current best OOF/test probabilities of A_lvl)."""
import sys, os; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
SETS = ['phys', 'tex', 'trk', 'rep', 'rep2', 'trk2', 'lvl']
FN = {'clean': lambda s: f'fe_{s}_train.csv', 'test': lambda s: f'fe_{s}_test.csv',
      'deg2': lambda s: {'tex': 'fe_texd_deg2.csv', 'trk': 'fe_trkd_deg2.csv'}.get(s, f'fe_{s}_train_deg2.csv'),
      'deg4': lambda s: {'tex': 'fe_texd_deg4.csv', 'trk': 'fe_trkd_deg4.csv'}.get(s, f'fe_{s}_train_deg4.csv')}
cols = None
for w in FN:
    X = pd.concat([pd.read_csv(os.path.join(D, FN[w](s))).add_prefix(s + '_') for s in SETS], axis=1)
    cols = list(X.columns); np.savez_compressed(ROOT + rf'\bundle\feats_{w}.npz', X=X.values.astype(np.float32))
    print(w, X.shape, round(os.path.getsize(ROOT + rf'\bundle\feats_{w}.npz') / 1e6, 1), 'MB')
np.savez_compressed(ROOT + r'\bundle\meta.npz', y=y_train(), folds=folds(), cols=np.array(cols), labels=np.array(LABELS),
                    oofc=np.load(D + r'\oofc_A_lvl.npy'), oofd=np.load(D + r'\oofd_A_lvl.npy'), pte=np.load(D + r'\pte_A_lvl.npy'),
                    test_ids=meta()[1].id.values, train_gap=meta()[0].slew_gap_rows.values, test_gap=meta()[1].slew_gap_rows.values)
print('meta', round(os.path.getsize(ROOT + r'\bundle\meta.npz') / 1e6, 1), 'MB')
