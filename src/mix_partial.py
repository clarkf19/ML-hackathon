"""Mix not-yet-complete pretrained CNNs (P1: some folds, E1: some folds) into a complete final blend with ONE weight,
chosen on the folds where they exist (honest: a single scalar, checked per fold), then applied to test.
usage: mix_partial.py BASE OUT.csv   (BASE = name used for final_blend output, e.g. base8_3)"""
import sys, os; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
BASE, OUT = sys.argv[1], sys.argv[2]
y = y_train(); fo = folds()
F = np.load(os.path.join(D, f'oofd_final_{BASE}.npy')); Ft = np.load(os.path.join(D, f'pte_{BASE}.npy'))
TAGS = os.environ.get('PTAGS', 'P1g,E1g').split(',')
have = {t: [k for k in range(5) if os.path.exists(os.path.join(D, f'oofdpart_{t}_f{k}.npy'))] for t in TAGS}
print('folds available:', have)
def cnn_fold(k):
    cs = [np.load(os.path.join(D, f'oofdpart_{t}_f{k}.npy')) for t in TAGS if k in have[t]]
    return np.mean(cs, 0) if cs else None
ks = sorted({k for t in TAGS for k in have[t]})
W = np.linspace(0, 0.7, 15); res = {k: [] for k in ks}
for k in ks:
    v = fo == k; c = cnn_fold(k)
    for w in W: res[k].append(mf1(y[v], ((1 - w) * F[v] + w * c).argmax(1)))
full = [k for k in ks if all(k in have[t] for t in TAGS)]
idx = np.concatenate([np.where(fo == k)[0] for k in full]); cf = np.concatenate([cnn_fold(k) for k in full])
pooled = [mf1(y[idx], ((1 - w) * F[idx] + w * cf).argmax(1)) for w in W]
wb = W[int(np.argmax(pooled))]
print(f'pooled folds {full} (all of {TAGS}): base {pooled[0]:.4f} -> best w={wb:.2f}: {max(pooled):.4f}')
for k in ks: print(f'  fold{k}: base {res[k][0]:.4f} | w={wb:.2f}: {res[k][list(W).index(wb)]:.4f} | best in fold {max(res[k]):.4f} at w={W[int(np.argmax(res[k]))]:.2f}')
Ct = np.mean([np.mean([np.load(os.path.join(D, f'ptepart_{t}_f{k}.npy')) for k in have[t]], 0) for t in TAGS], 0)
Pt = (1 - wb) * Ft + wb * Ct
print(write_sub(Pt.argmax(1), OUT)); np.save(os.path.join(D, f'pte_{OUT[:-4]}.npy'), Pt)
