"""Per-class log-prob biases for macro-F1, nested: tune on 4 folds' fake-PKS OOF, score on the 5th (honest)."""
import sys; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
TAG = sys.argv[1]; y = y_train(); fo = folds()
P = np.log(np.load(D + rf'\oofd_{TAG}.npy') + 1e-9); Pc = np.log(np.load(D + rf'\oofc_{TAG}.npy') + 1e-9)
def tune(L, yy):
    b = np.zeros(9)
    for it in range(3):
        for c in range(9):
            best = (mf1(yy, (L + b).argmax(1)), b[c])
            for v in np.linspace(-1.5, 1.5, 31):
                bb = b.copy(); bb[c] = v; s = mf1(yy, (L + bb).argmax(1))
                if s > best[0] + 1e-6: best = (s, v)
            b[c] = best[1]
    return b
pr = np.zeros(len(y), int); prc = np.zeros(len(y), int); Bs = []
for k in range(5):
    t, v = fo != k, fo == k
    b = tune(P[t], y[t]); Bs.append(b); pr[v] = (P[v] + b).argmax(1); prc[v] = (Pc[v] + b).argmax(1)
print(f'fake-PKS: argmax {mf1(y, P.argmax(1)):.4f} -> nested-bias {mf1(y, pr):.4f} | clean: {mf1(y, Pc.argmax(1)):.4f} -> {mf1(y, prc):.4f}')
print('mean bias', dict(zip(LABELS, np.round(np.mean(Bs, 0), 2))), ' fold std', np.round(np.std(Bs, 0).mean(), 2))
