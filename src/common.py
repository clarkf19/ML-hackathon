import numpy as np, pandas as pd, os
from sklearn.metrics import f1_score
ROOT = r'C:\Users\clark\hack4hack-r2'
D = ROOT + r'\data'
SRC = D + r'\signal-hunters-round2'
LABELS = ['eti_candidate', 'frb', 'pulsar', 'scintillating_star', 'satellite_pass',
          'radar_chirp', 'rfi_intermittent', 'rfi_persistent', 'noise']   # priority order
L2I = {l: i for i, l in enumerate(LABELS)}

def meta():
    tr = pd.read_csv(SRC + r'\train_meta.csv'); te = pd.read_csv(SRC + r'\test_meta.csv')
    return tr, te

def y_train():
    return meta()[0].label.map(L2I).values

def folds(k=5, seed=42):
    f = D + r'\folds.npy'
    if not os.path.exists(f):
        from sklearn.model_selection import StratifiedKFold
        y = y_train(); fo = np.zeros(len(y), int)
        for i, (_, va) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(y, y)): fo[va] = i
        np.save(f, fo)
    return np.load(f)

def mf1(y, p):
    return f1_score(y, p, average='macro')

def per_class(y, p):
    return dict(zip(LABELS, np.round(f1_score(y, p, average=None, labels=range(9)), 3)))

def write_sub(pred_idx, name):
    te = meta()[1]
    sub = pd.DataFrame({'id': te.id, 'label': [LABELS[i] for i in pred_idx]})
    ss = pd.read_csv(SRC + r'\sample_submission.csv')
    assert len(sub) == 10000 and (sub.id.values == ss.id.values).all() and sub.label.isin(LABELS).all()
    sub.to_csv(ROOT + rf'\submissions\{name}', index=False)
    return sub.label.value_counts(normalize=True).round(3).to_dict()
