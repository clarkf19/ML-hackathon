"""LightGBM with augmentation sets, evaluated on clean val AND fake-PKS (deg2) val. Same folds.
usage: gbdt_aug.py TAG AUGS   AUGS = comma list from {deg2, inj1} or 'none'.  Feature sets fixed below (env SETS)."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
SETS = os.environ.get('SETS', 'phys,tex,trk,rep,rep2').split(',')
FN = {'clean': lambda s: f'fe_{s}_train.csv', 'test': lambda s: f'fe_{s}_test.csv',
      'deg2': lambda s: {'tex': 'fe_texd_deg2.csv', 'trk': 'fe_trkd_deg2.csv', 'trk2': 'fe_trk2_train_deg2.csv', 'trk3': 'fe_trk3_train_deg2.csv', 'swp': 'fe_swp_train_deg2.csv', 'tex2': 'fe_tex2_train_deg2.csv'}.get(s, f'fe_{s}_train_deg2.csv'),
      'deg4': lambda s: {'tex': 'fe_texd_deg4.csv', 'trk': 'fe_trkd_deg4.csv', 'trk2': 'fe_trk2_train_deg4.csv', 'trk3': 'fe_trk3_train_deg4.csv', 'swp': 'fe_swp_train_deg4.csv', 'tex2': 'fe_tex2_train_deg4.csv'}.get(s, f'fe_{s}_train_deg4.csv'),
      'deg6': lambda s: {'tex': 'fe_texd_deg6.csv', 'trk': 'fe_trkd_deg6.csv', 'trk2': 'fe_trk2_train_deg6.csv', 'trk3': 'fe_trk3_train_deg6.csv', 'swp': 'fe_swp_train_deg6.csv'}.get(s, f'fe_{s}_train_deg6.csv'),
      'deg8': lambda s: {'tex': 'fe_texd_deg8.csv', 'trk': 'fe_trkd_deg8.csv', 'trk2': 'fe_trk2_train_deg8.csv', 'trk3': 'fe_trk3_train_deg8.csv', 'swp': 'fe_swp_train_deg8.csv'}.get(s, f'fe_{s}_train_deg8.csv'),
      'deg10': lambda s: {'tex': 'fe_texd_deg10.csv', 'trk': 'fe_trkd_deg10.csv', 'trk2': 'fe_trk2_train_deg10.csv', 'trk3': 'fe_trk3_train_deg10.csv', 'swp': 'fe_swp_train_deg10.csv'}.get(s, f'fe_{s}_train_deg10.csv'),
      'deg12': lambda s: {'tex': 'fe_texd_deg12.csv', 'trk': 'fe_trkd_deg12.csv', 'trk2': 'fe_trk2_train_deg12.csv', 'trk3': 'fe_trk3_train_deg12.csv', 'swp': 'fe_swp_train_deg12.csv'}.get(s, f'fe_{s}_train_deg12.csv'),
      'deg14': lambda s: {'tex': 'fe_texd_deg14.csv', 'trk': 'fe_trkd_deg14.csv', 'trk2': 'fe_trk2_train_deg14.csv', 'trk3': 'fe_trk3_train_deg14.csv', 'swp': 'fe_swp_train_deg14.csv'}.get(s, f'fe_{s}_train_deg14.csv'),
      'deg16': lambda s: {'tex': 'fe_texd_deg16.csv', 'trk': 'fe_trkd_deg16.csv', 'trk2': 'fe_trk2_train_deg16.csv', 'trk3': 'fe_trk3_train_deg16.csv', 'swp': 'fe_swp_train_deg16.csv'}.get(s, f'fe_{s}_train_deg16.csv'),
      'inj1': lambda s: f'fe_{s}_inj1.csv'}
def load(which): return pd.concat([pd.read_csv(D + '\\' + FN[which](s)).add_prefix(s + '_') for s in SETS], axis=1)
TAG = sys.argv[1]; AUGS = [a for a in sys.argv[2].split(',') if a != 'none']
Xc, Xt, Xd = load('clean'), load('test'), load('deg2'); X4 = load('deg4'); XX = {a: load(a) for a in AUGS if a in ('deg6', 'deg8', 'deg10', 'deg12', 'deg14', 'deg16')}
y = y_train(); fo = folds()
extra = {}
if 'inj1' in AUGS:
    info = pd.read_csv(D + r'\inj1_info.csv'); extra['inj1'] = (load('inj1'), info.y.values, fo[info.base.values])
prm = dict(n_estimators=int(os.environ.get('NEST', 900)), learning_rate=float(os.environ.get('GLR', 0.05)), num_leaves=int(os.environ.get('LEAVES', 31)), min_child_samples=20, colsample_bytree=0.5,
           subsample=0.8, subsample_freq=1, class_weight='balanced', verbose=-1, num_threads=int(os.environ.get('NT', 8)), random_state=int(os.environ.get('GSEED', 0)) or None)
oc = np.zeros((len(y), 9)); od = np.zeros((len(y), 9)); o4 = np.zeros((len(y), 9)); pt = np.zeros((len(Xt), 9))
for k in range(5):
    tri, vai = fo != k, fo == k
    XS, YS = [Xc[tri]], [y[tri]]
    if 'deg2' in AUGS: XS.append(Xd[tri]); YS.append(y[tri])
    if 'deg4' in AUGS: XS.append(X4[tri]); YS.append(y[tri])
    for a in AUGS:
        if a in ('deg6', 'deg8', 'deg10', 'deg12', 'deg14', 'deg16'): XS.append(XX[a][tri]); YS.append(y[tri])
    for nm, (Xe, ye, fe) in extra.items(): XS.append(Xe[fe != k]); YS.append(ye[fe != k])
    if os.environ.get('PL'):
        pp = np.load(D + rf"\pte_{os.environ['PL']}.npy"); cf = pp.max(1) > float(os.environ.get('PLTH', 0.9))
        XS.append(Xt[cf]); YS.append(pp.argmax(1)[cf])
    m = lgb.LGBMClassifier(**prm).fit(pd.concat(XS), np.concatenate(YS))
    oc[vai] = m.predict_proba(Xc[vai]); od[vai] = m.predict_proba(Xd[vai]); o4[vai] = m.predict_proba(X4[vai]); pt += m.predict_proba(Xt) / 5
for nm, a in [('oofc', oc), ('oofd', od), ('oof4', o4), ('pte', pt)]: np.save(D + rf'\{nm}_{TAG}.npy', a)
msg = f'{TAG} sets={SETS} augs={AUGS}  clean-val {mf1(y, oc.argmax(1)):.4f}  fakePKS-val {mf1(y, od.argmax(1)):.4f}  fPKS-v4-val {mf1(y, o4.argmax(1)):.4f}\n  clean {per_class(y, oc.argmax(1))}\n  fPKS  {per_class(y, od.argmax(1))}\n  test mix {dict(zip(LABELS, np.round(np.bincount(pt.argmax(1), minlength=9) / len(pt), 3)))}'
print(msg); open(ROOT + r'\logs\aug_results.txt', 'a').write(msg + '\n')
