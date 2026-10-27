"""F: does EM prior-shift correction help macro-F1 when the test label mix differs? Proxy on fake-PKS OOF probs."""
import sys; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
P = np.load(D + rf'\oofd_{sys.argv[1]}.npy'); y = y_train()
pi_tr = np.ones(9) / 9   # class_weight=balanced -> model priors ~uniform
def em(p, it=50):
    pi = pi_tr.copy()
    for _ in range(it):
        q = p * (pi / pi_tr); q /= q.sum(1, keepdims=True); pi = q.mean(0)
    return q, pi
rng = np.random.default_rng(0); res = []
for r in range(40):
    mix = rng.dirichlet(np.ones(9) * 3); n = 4000
    idx = np.concatenate([rng.choice(np.where(y == c)[0], int(n * mix[c]) + 5) for c in range(9)])
    p, yy = P[idx], y[idx]; q, pi = em(p)
    q_half = p * np.sqrt(pi / pi_tr)
    res.append((mf1(yy, p.argmax(1)), mf1(yy, q.argmax(1)), mf1(yy, q_half.argmax(1)), np.abs(pi - np.bincount(yy, minlength=9) / len(yy)).sum()))
r = np.array(res); print('argmax %.4f  EM %.4f  EM-half %.4f  | EM prior L1 err %.3f' % tuple(r.mean(0)))
print('EM better in', (r[:, 1] > r[:, 0]).mean(), 'of resamples; half better in', (r[:, 2] > r[:, 0]).mean())
# on natural mix (train-like)
q, pi = em(P); print('natural mix: argmax %.4f EM %.4f' % (mf1(y, P.argmax(1)), mf1(y, q.argmax(1))))
Pt = np.load(D + rf'\pte_{sys.argv[1]}.npy'); qt, pit = em(Pt)
print('EM-estimated TEST mix:', dict(zip(LABELS, np.round(pit, 3))))
print('test argmax mix      :', dict(zip(LABELS, np.round(np.bincount(Pt.argmax(1), minlength=9) / len(Pt), 3))))
