"""Do signal scales differ train (GBT) vs test (PKS)? Frequency/time autocorrelation of the ON-minus-OFF residual
for scintillating stars (train labels vs confident test predictions of submission10), noise as reference."""
import sys, os; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
y = y_train(); P = np.load(os.path.join(D, 'pte_submission10.npy')); pt = P.argmax(1); cf = P.max(1) > 0.9
Zr = np.load(os.path.join(D, 'zr_train.npy'), mmap_mode='r'); Zt = np.load(os.path.join(D, 'zr_test.npy'), mmap_mode='r')
def acf(Z, idx, n=600):
    idx = idx[:n]; Af = np.zeros(20); At = np.zeros(8)
    for i in idx:
        z = np.asarray(Z[i], np.float32); z = np.nan_to_num(z)
        r = z[[0, 2, 4]].mean(0) - z[[1, 3, 5]].mean(0)          # 16 x 128 ON - OFF
        r = r - r.mean(); v = (r * r).mean() + 1e-9
        Af += np.array([(r[:, :128 - d] * r[:, d:]).mean() / v for d in range(20)])
        On = z[[0, 2, 4]] - z[[0, 2, 4]].mean((1, 2), keepdims=True); vo = (On * On).mean() + 1e-9
        At += np.array([(On[:, :16 - d] * On[:, d:]).mean() / vo for d in range(8)])
    return Af / len(idx), At / len(idx)
S = LABELS.index('scintillating_star'); N = LABELS.index('noise')
for nm, Z, idx in [('train scint', Zr, np.where(y == S)[0]), ('test scint p>.9', Zt, np.where((pt == S) & cf)[0]),
                   ('train noise', Zr, np.where(y == N)[0]), ('test noise p>.9', Zt, np.where((pt == N) & cf)[0])]:
    f, t = acf(Z, idx); print(f'{nm:16s} n={len(idx):5d} freq-acf', np.round(f[:10], 3), '| time-acf', np.round(t[:6], 3))
print(Zr.dtype, Zr.shape)
