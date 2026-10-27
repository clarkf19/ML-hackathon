"""Error analysis of the current best (base8_3 mixed 0.5 with pretrained CNNs) on fake-PKS folds that have CNNs,
plus per-class bias tuning for macro-F1 with honest fold-split evaluation."""
import sys, os; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
y = y_train(); fo = folds()
F = np.load(os.path.join(D, 'oofd_final_base8_3.npy'))
TAGS = 'P1g,E1g,E2g,E3g,E4g,P3g'.split(',')
M = np.zeros_like(F); ok = np.zeros(len(y), bool)
for k in range(5):
    cs = [np.load(os.path.join(D, f'oofdpart_{t}_f{k}.npy')) for t in TAGS if os.path.exists(os.path.join(D, f'oofdpart_{t}_f{k}.npy'))]
    if cs: v = fo == k; M[v] = 0.5 * F[v] + 0.5 * np.mean(cs, 0); ok |= v
yy, P = y[ok], M[ok]; p = P.argmax(1)
print('n', ok.sum(), 'macroF1', round(mf1(yy, p), 4)); print(per_class(yy, p))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yy, p, labels=range(9)); print('rows=true, cols=pred', [l[:6] for l in LABELS]); print(cm)
off = [(cm[i, j], LABELS[i], LABELS[j]) for i in range(9) for j in range(9) if i != j]; print(sorted(off, reverse=True)[:12])
# bias tuning: coordinate ascent on log-probs, tuned on half the folds, scored on the other half
L = np.log(P + 1e-9); fk = fo[ok]
def tune(idx):
    b = np.zeros(9)
    for it in range(3):
        for c in range(9):
            best = (mf1(yy[idx], (L[idx] + b).argmax(1)), b[c])
            for d in np.linspace(-1.5, 1.5, 31):
                bb = b.copy(); bb[c] = d; s = mf1(yy[idx], (L[idx] + bb).argmax(1))
                if s > best[0]: best = (s, d)
            b[c] = best[1]
    return b
ks = sorted(set(fk))
for A in [ks[:len(ks)//2], ks[len(ks)//2:]]:
    tr = np.isin(fk, A); te = ~tr; b = tune(np.where(tr)[0])
    print('tune on', A, 'bias', np.round(b, 2), '| held-out', round(mf1(yy[te], L[te].argmax(1)), 4), '->', round(mf1(yy[te], (L[te] + b).argmax(1)), 4))
