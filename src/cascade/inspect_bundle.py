"""Quick look at the bundle: shapes, label counts, fold sizes, stored best-model probs."""
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
B = Path(__file__).resolve().parents[2] / 'bundle'
m = np.load(B / 'meta.npz', allow_pickle=True)
for k in m.files: print(k, m[k].shape, m[k].dtype)
print(list(m['labels']))
y = m['y']; print(np.bincount(y)); print(np.bincount(m['folds']))
print('meta oofc', f1_score(y, m['oofc'].argmax(1), average='macro'), 'oofd', f1_score(y, m['oofd'].argmax(1), average='macro'))
