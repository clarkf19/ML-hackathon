"""Shared loading / scoring helpers for the decision-layer experiments (bundle features only, no raw data).
Labels in bundle/meta.npz are already in priority order: 0 = eti_candidate (highest) ... 8 = noise (lowest)."""
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
B, R = ROOT / 'bundle', ROOT / 'results'
R.mkdir(exist_ok=True)
_m = np.load(B / 'meta.npz', allow_pickle=True)
y, folds, L = _m['y'], _m['folds'], [str(s) for s in _m['labels']]
K = len(L)
SHORT = ['eti', 'frb', 'pulsar', 'scint', 'sat', 'radar', 'rfi_int', 'rfi_pers', 'noise']
LGB = dict(n_estimators=900, learning_rate=0.05, num_leaves=31, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
           verbose=-1)  # = src/bundle_eval.py params (it adds class_weight='balanced' for the multiclass model)


def load_X():
    return {w: np.load(B / f'feats_{w}.npz')['X'] for w in ['clean', 'test', 'deg2', 'deg4']}


def meta_probs():
    return {k: _m[k] for k in ['oofc', 'oofd', 'pte']}


def load_probs(name):
    z = np.load(R / f'probs_{name}.npz')
    return {k: z[k] for k in z.files}


def conf(yt, yp, k=K):
    return np.bincount(yt * k + yp, minlength=k * k).reshape(k, k)


def f1_per(yt, yp, k=K):
    cm = conf(yt, yp, k); tp = np.diag(cm)
    return 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)


def f1m(yt, yp, k=K):
    return f1_per(yt, yp, k).mean()
