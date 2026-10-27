"""Do the pairwise specialists separate their two classes better than the multiclass baseline already does?
AUC on rows whose true label is a or b: specialist P(a | a or b) vs baseline p_a / (p_a + p_b), clean-val and fakePKS-val;
plus the same AUC restricted to rows the baseline confuses within the pair (argmax in {a, b}, wrong)."""
import numpy as np
from sklearn.metrics import roc_auc_score
from common import y, SHORT, load_probs
from train_base import PAIRS

H, Q = load_probs('baseline'), load_probs('pairs')
for j, (a, b) in enumerate(PAIRS):
    m = (y == a) | (y == b); t = y[m] == a; s = []
    for k in ['oofc', 'oofd']:
        hb = H[k][m, a] / (H[k][m, a] + H[k][m, b] + 1e-12); sp = Q[k][m, j]
        s.append(f'{k[3:]}: base {roc_auc_score(t, hb):.4f} spec {roc_auc_score(t, sp):.4f} avg {roc_auc_score(t, hb + sp):.4f}')
    print(f'{SHORT[a]:7s} vs {SHORT[b]:8s} n={m.sum():5d} | ' + ' | '.join(s))
