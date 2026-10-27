import sys, os; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
T = pd.concat([pd.read_csv(os.path.join(D, f)) for f in ['fe_phys_test.csv', 'fe_tex_test.csv']], axis=1)
for nm, fn in [('clean', ['fe_phys_train.csv', 'fe_tex_train.csv']), ('deg2', ['fe_phys_train_deg2.csv', 'fe_texd_deg2.csv']), ('deg4', ['fe_phys_train_deg4.csv', 'fe_texd_deg4.csv'])]:
    A = pd.concat([pd.read_csv(os.path.join(D, f)) for f in fn], axis=1); Z = pd.concat([A, T]); ya = np.r_[np.zeros(len(A)), np.ones(len(T))]
    f5 = np.random.default_rng(0).integers(0, 5, len(Z)); pa = np.zeros(len(Z)); imp = 0
    for k in range(5):
        m = lgb.LGBMClassifier(n_estimators=200, verbose=-1, num_threads=6).fit(Z[f5 != k], ya[f5 != k]); pa[f5 == k] = m.predict_proba(Z[f5 == k])[:, 1]; imp = imp + m.booster_.feature_importance('gain')
    print('adv AUC', nm, 'vs test', round(roc_auc_score(ya, pa), 4), pd.Series(imp, Z.columns).sort_values(ascending=False).head(6).round(0).to_dict())
