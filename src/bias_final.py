"""Per-class log-bias for macro-F1 on top of a finished blend, scored honestly: biases tuned on 4 folds, applied to the 5th
(pooled over all 5). If the nested gain is real, biases tuned on all folds (shrunk by SHRINK) are applied to the test probabilities.
usage: bias_final.py BASE TESTNAME OUT.csv   (BASE: oofd_final_{BASE}.npy; TESTNAME: pte_{TESTNAME}.npy)"""
import sys, os; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
BASE, TN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
y = y_train(); fo = folds(); SH = float(os.environ.get('SHRINK', 0.5)); MIN = float(os.environ.get('MINGAIN', 0.0005))
L = np.log(np.load(os.path.join(D, f'oofd_final_{BASE}.npy')) + 1e-9); T = np.log(np.load(os.path.join(D, f'pte_{TN}.npy')) + 1e-9)
def tune(idx):
    b = np.zeros(9)
    for it in range(3):
        for c in range(9):
            best = (mf1(y[idx], (L[idx] + b).argmax(1)), b[c])
            for d in np.linspace(-1, 1, 21):
                bb = b.copy(); bb[c] = d; s = mf1(y[idx], (L[idx] + bb).argmax(1))
                if s > best[0] + 1e-9: best = (s, d)
            b[c] = best[1]
    return b
nest = np.zeros(len(y), int); nest_sh = np.zeros(len(y), int)
for k in range(5):
    b = tune(np.where(fo != k)[0]); v = fo == k; nest[v] = (L[v] + b).argmax(1); nest_sh[v] = (L[v] + SH * b).argmax(1)
base = mf1(y, L.argmax(1)); g_full, g_sh = mf1(y, nest) - base, mf1(y, nest_sh) - base
print(f'base {base:.4f} | nested bias {base + g_full:.4f} ({g_full:+.4f}) | nested bias x{SH} {base + g_sh:.4f} ({g_sh:+.4f})')
bf = tune(np.arange(len(y))); print('biases (all folds)', dict(zip([l[:6] for l in LABELS], np.round(bf, 2))))
if g_sh >= MIN:
    P = T + SH * bf; print(f'applied x{SH}:', write_sub(P.argmax(1), OUT)); np.save(os.path.join(D, f'pte_{OUT[:-4]}.npy'), np.exp(P))
else: print(f'nested gain {g_sh:+.4f} < {MIN}: not applied')
