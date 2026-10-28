"""Second stacker: multinomial logistic regression on the same meta inputs (log-probs of each feature model and each
complete CNN), nested 5-fold on clean + fake-PKS rows like the LightGBM meta; scored alone and averaged with the existing
final blend. usage: meta_lr.py BASE CNNTAG...   (BASE: oofd_final_{BASE}.npy from final_blend with the same inputs)"""
import sys, os; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
BASE, TAGS = sys.argv[1], sys.argv[2:]
y = y_train(); fo = folds(); L = lambda a: np.log(a + 1e-6)
GTS = os.environ.get('GBDT', 'A_rsswp_PL5,A_rsswp_PL5t6,A_rs_PL5t6,A_trk3_PL5t6,A_rsswp_PL4,A_rs_PL3').split(',')
def cnn(t):
    od, oc = np.zeros((len(y), 9)), np.zeros((len(y), 9))
    for k in range(5): v = fo == k; od[v] = np.load(D + rf'\oofdpart_{t}_f{k}.npy'); oc[v] = np.load(D + rf'\oofpart_{t}_f{k}.npy')
    return od, oc, np.mean([np.load(D + rf'\ptepart_{t}_f{k}.npy') for k in range(5)], 0)
CS = [cnn(t) for t in TAGS]
X = np.hstack([L(np.load(D + rf'\oofd_{t}.npy')) for t in GTS] + [L(c[0]) for c in CS])
Xc = np.hstack([L(np.load(D + rf'\oofc_{t}.npy')) for t in GTS] + [L(c[1]) for c in CS])
Xt = np.hstack([L(np.load(D + rf'\pte_{t}.npy')) for t in GTS] + [L(c[2]) for c in CS])
C = float(os.environ.get('LRC', 0.05))
mk = lambda: LogisticRegression(C=C, class_weight='balanced', max_iter=300)
pl = np.zeros((len(y), 9))
for k in range(5):
    t, v = fo != k, fo == k
    sc = StandardScaler().fit(np.vstack([X[t], Xc[t]])); m = mk().fit(sc.transform(np.vstack([X[t], Xc[t]])), np.r_[y[t], y[t]])
    pl[v] = m.predict_proba(sc.transform(X[v]))
F = np.load(os.path.join(D, f'oofd_final_{BASE}.npy'))
print(f'C={C}: LR meta alone {mf1(y, pl.argmax(1)):.4f} | current final {mf1(y, F.argmax(1)):.4f}')
for a in (0.2, 0.33, 0.5):
    M = (1 - a) * F + a * pl; print(f'  final*(1-{a}) + LR*{a}: {mf1(y, M.argmax(1)):.4f}  per-fold {[round(mf1(y[fo == k], M[fo == k].argmax(1)), 4) for k in range(5)]}')
if os.environ.get('SAVE'):
    sc = StandardScaler().fit(np.vstack([X, Xc])); m = mk().fit(sc.transform(np.vstack([X, Xc])), np.r_[y, y])
    np.save(os.path.join(D, f'oofd_lr_{BASE}.npy'), pl); np.save(os.path.join(D, f'pte_lr_{BASE}.npy'), m.predict_proba(sc.transform(Xt))); print('saved')
