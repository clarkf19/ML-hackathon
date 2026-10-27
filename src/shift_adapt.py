"""Label-shift adaptation for macro-F1: evaluate on fake-PKS OOF re-weighted to the BBSE test mix.
Compare plain argmax vs per-class log-bias tuned (nested over folds) on the re-weighted rows."""
import sys; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
from sklearn.metrics import f1_score
y = y_train(); fo = folds(); F = np.load(D + rf'\oofd_final_{sys.argv[1]}.npy'); q = np.load(D + r'\bbse_test_mix.npy')
trm = np.bincount(y, minlength=9) / len(y); w = (q / trm)[y]
wf1 = lambda yy, pp, ww: f1_score(yy, pp, average='macro', sample_weight=ww)
L = np.log(F + 1e-9)
def tune(Lt, yt, wt, grid=np.linspace(-1.2, 1.2, 25)):
    b = np.zeros(9)
    for _ in range(3):
        for c in range(9):
            best = (wf1(yt, (Lt + b).argmax(1), wt), b[c])
            for v in grid:
                bb = b.copy(); bb[c] = v; s = wf1(yt, (Lt + bb).argmax(1), wt)
                if s > best[0] + 1e-6: best = (s, v)
            b[c] = best[1]
    return b
pr = np.zeros(len(y), int); pr_u = np.zeros(len(y), int); Bs = []
for k in range(5):
    t, v = fo != k, fo == k
    b = tune(L[t], y[t], w[t]); Bs.append(b); pr[v] = (L[v] + b).argmax(1)
    bu = tune(L[t], y[t], np.ones(t.sum())); pr_u[v] = (L[v] + bu).argmax(1)
prior = (L + np.log(q / trm)).argmax(1)                                   # closed-form prior correction
print(f'evaluated at TEST mix (re-weighted fake-PKS OOF):')
print(f'  argmax                    {wf1(y, F.argmax(1), w):.4f}')
print(f'  + log prior ratio         {wf1(y, prior, w):.4f}')
print(f'  + bias tuned at train mix {wf1(y, pr_u, w):.4f}')
print(f'  + bias tuned at test mix  {wf1(y, pr, w):.4f}   (nested)')
print(f'evaluated at TRAIN mix: argmax {mf1(y, F.argmax(1)):.4f} | test-mix biases {mf1(y, pr):.4f}')
print('mean bias', dict(zip(LABELS, np.round(np.mean(Bs, 0), 2))))
np.save(D + r'\shift_bias.npy', np.mean(Bs, 0))
