"""Candidate submission = argmax of w * (harness model with raw stats) + (1-w) * A_lvl (bundle/meta.npz pte, 4-copy model).
usage: python src/make_blend_sub.py results/probs_X.npz OUT.csv [w=0.5]"""
import numpy as np, pandas as pd, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
m = np.load(ROOT / 'bundle/meta.npz', allow_pickle=True); L = list(m['labels'])
w = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
p = w * np.load(sys.argv[1])['pte'] + (1 - w) * m['pte']
sub = pd.DataFrame({'id': m['test_ids'], 'label': [L[i] for i in p.argmax(1)]})
ss = pd.read_csv(ROOT / 'data/signal-hunters-round2/sample_submission.csv')
assert len(sub) == 10000 and (sub.id.values == ss.id.values).all() and sub.label.isin(L).all()
sub.to_csv(sys.argv[2], index=False)
old = pd.read_csv(ROOT / 'submissions/submission2.csv')
print(sys.argv[2], 'agreement with submission2', (old.label.values == sub.label.values).mean().round(4))
print(sub.label.value_counts(normalize=True).round(3).to_dict())
