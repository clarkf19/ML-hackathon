"""Smoke test of the nested decision rules on the stored best-model probabilities (bundle/meta.npz)."""
import time, numpy as np
from common import y, f1m, meta_probs
import rules as r

P = meta_probs()
Sc, Sd = {'p': P['oofc']}, {'p': P['oofd']}
print(f'meta argmax  clean {f1m(y, P["oofc"].argmax(1)):.4f}  fPKS {f1m(y, P["oofd"].argmax(1)):.4f}')
t0 = time.time()
pc, pd, ps, _ = r.nested(r.bias_fit, r.bias_apply, Sc, Sd, y)
print(f'meta +bias   clean {f1m(y, pc):.4f}  fPKS {f1m(y, pd):.4f}  ({time.time() - t0:.0f}s)')
for b in ps: print(np.round(b, 2))
# cascade on the multiclass model's conditional probabilities c_k = p_k / sum_{j>=k} p_j
cs = lambda p: p / np.cumsum(p[:, ::-1], 1)[:, ::-1]
Sc['s'], Sd['s'] = cs(P['oofc']), cs(P['oofd'])
for fb in ['noise', 'argmax']:
    t0 = time.time()
    pc, pd, ps, _ = r.nested(lambda Ss, ys: r.cascade_fit(Ss, ys, fb), lambda t, S: r.cascade_apply(t, S, fb), Sc, Sd, y)
    print(f'meta cascade({fb}) clean {f1m(y, pc):.4f}  fPKS {f1m(y, pd):.4f}  ({time.time() - t0:.0f}s)')
    for t in ps: print(t)
