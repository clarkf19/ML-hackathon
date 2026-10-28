"""Third stacker: small MLP on the same meta inputs (log-probs of feature models + complete CNNs), nested 5-fold on clean +
fake-PKS rows; class-balanced by dividing by the training class prior. Scored added to submission15's OOF mix.
usage: meta_mlp.py NAME CNNTAG...   (NAME: the final_build name, e.g. submission15)"""
import sys, os; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
NAME, TAGS = sys.argv[1], sys.argv[2:]
y = y_train(); fo = folds(); L = lambda a: np.log(a + 1e-6)
GTS = 'A_rsswp_PL5,A_rsswp_PL5t6,A_rs_PL5t6,A_trk3_PL5t6,A_rsswp_PL4,A_rs_PL3'.split(',')
def cnn(t):
    od, oc = np.zeros((len(y), 9)), np.zeros((len(y), 9))
    for k in range(5): v = fo == k; od[v] = np.load(D + rf'\oofdpart_{t}_f{k}.npy'); oc[v] = np.load(D + rf'\oofpart_{t}_f{k}.npy')
    return od, oc, np.mean([np.load(D + rf'\ptepart_{t}_f{k}.npy') for k in range(5)], 0)
CS = [cnn(t) for t in TAGS]
X = np.hstack([L(np.load(D + rf'\oofd_{t}.npy')) for t in GTS] + [L(c[0]) for c in CS])
Xc = np.hstack([L(np.load(D + rf'\oofc_{t}.npy')) for t in GTS] + [L(c[1]) for c in CS])
Xt = np.hstack([L(np.load(D + rf'\pte_{t}.npy')) for t in GTS] + [L(c[2]) for c in CS])
prior = np.bincount(y, minlength=9) / len(y)
mk = lambda: MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=60, random_state=0)
def fitpred(Xtr, ytr, Xs):
    sc = StandardScaler().fit(Xtr); m = mk().fit(sc.transform(Xtr), ytr)
    out = [m.predict_proba(sc.transform(a)) / prior for a in Xs]; return [o / o.sum(1, keepdims=True) for o in out]
pn = np.zeros((len(y), 9))
for k in range(5):
    t, v = fo != k, fo == k; pn[v] = fitpred(np.vstack([X[t], Xc[t]]), np.r_[y[t], y[t]], [X[v]])[0]
F = 0.8 * np.load(os.path.join(D, f'oofd_final_{NAME}_m_base.npy')) + 0.2 * np.load(os.path.join(D, f'oofd_lr_{NAME}_m_base.npy'))
print(f'MLP alone {mf1(y, pn.argmax(1)):.4f} | {NAME} OOF {mf1(y, F.argmax(1)):.4f} {[round(mf1(y[fo == k], F[fo == k].argmax(1)), 4) for k in range(5)]}', flush=True)
for a in (0.1, 0.2, 0.3):
    M = (1 - a) * F + a * pn; print(f'  +MLP*{a}: {mf1(y, M.argmax(1)):.4f}  per-fold {[round(mf1(y[fo == k], M[fo == k].argmax(1)), 4) for k in range(5)]}', flush=True)
if os.environ.get('SAVE'):
    a = float(os.environ['SAVE']); pt = fitpred(np.vstack([X, Xc]), np.r_[y, y], [Xt])[0]
    P = (1 - a) * np.load(os.path.join(D, f'pte_{NAME}.npy')) + a * pt
    np.save(os.path.join(D, f'pte_{NAME}_mlp.npy'), P); print(write_sub(P.argmax(1), os.environ.get('OUTCSV', f'{NAME}_mlp.csv')))
