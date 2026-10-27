"""E2: scintillation SPECIALIST stacked into the 9-class model, scored exactly like src/bundle_eval.py (same LightGBM
params, clean + deg2 + deg4 training, clean-val / fakePKS(deg2)-val, same 5 folds) but with HONEST nested stacking:
for outer fold k the specialist score of a training row comes from an inner model that saw neither fold k nor that row.
Specialist = binary LightGBM "is scintillating_star", trained on
  MODE=all   : all classes
  MODE=lower : scint vs the classes BELOW it in label priority (satellite, radar, rfi_int, rfi_pers, noise) — a scint
               star can only hide inside eti/frb/pulsar cadences, never inside lower-priority labels.
FEATS=all (default) gives the specialist every column; FEATS=scint restricts it to level / texture columns (lvl_*, tex_*,
phys_tex_*, phys_mean_on_off + extras) so it cannot learn "a line is present -> not scint" shortcuts and its score is
pure scint evidence; the 9-class model then decides with every feature.
Also writes results/feats_spec_{MODE}[_scint].npz (plain 5-fold OOF score, test = fold mean) for src/bundle_eval.py.
usage: MODE=lower FEATS=scint python src/scint_stack.py [extra.npz ...]"""
import sys, os, numpy as np, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import f1_score
R = Path(__file__).resolve().parent.parent
m = np.load(R / 'bundle/meta.npz', allow_pickle=True); y, fo, L = m['y'], m['folds'], list(m['labels'])
X = {w: np.load(R / f'bundle/feats_{w}.npz')['X'] for w in ['clean', 'test', 'deg2', 'deg4']}
cols = [str(c) for c in m['cols']]
for e in sys.argv[1:]:
    z = np.load(e); cols += [str(c) for c in z['cols']]
    for w in X: X[w] = np.hstack([X[w], z['X_' + w]])
MODE = os.environ.get('MODE', 'lower'); FEATS = os.environ.get('FEATS', 'all'); S = 3
SUF = '' if FEATS == 'all' else '_' + FEATS
sel = np.array([FEATS == 'all' or not c.startswith(('phys_', 'trk', 'rep')) or c.startswith('phys_tex_') or c == 'phys_mean_on_off'
                for c in cols])
print('specialist columns', sel.sum(), 'of', len(cols), flush=True)
keep = {'all': np.ones(len(y), bool), 'lower': np.isin(y, [3, 4, 5, 6, 7, 8]),
        'confusers': np.isin(y, [3, 4, 7, 8])}[MODE]                  # confusers: scint vs noise / rfi_pers / satellite only
prm = dict(n_estimators=900, learning_rate=0.05, num_leaves=31, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
           class_weight='balanced', verbose=-1)
sprm = dict(n_estimators=int(os.environ.get('SNEST', 600)), learning_rate=0.05, num_leaves=31, colsample_bytree=0.5,
            subsample=0.8, subsample_freq=1, is_unbalance=True, verbose=-1)
W = ['clean', 'deg2', 'deg4']

def spec_fit(tr):
    t = tr & keep
    return lgb.LGBMClassifier(**sprm).fit(np.vstack([X[w][t][:, sel] for w in W]), np.tile((y[t] == S).astype(int), 3))

def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def score(mdl, rows): return {w: logit(mdl.predict_proba(X[w][rows][:, sel])[:, 1]) for w in W}

# plain OOF specialist (for the harness npz and for the val rows of the nested scheme)
oof = {w: np.zeros(len(y)) for w in W}; ste = np.zeros(len(X['test']))
for k in range(5):
    v = fo == k; mdl = spec_fit(fo != k)
    for w, a in score(mdl, v).items(): oof[w][v] = a
    ste += logit(mdl.predict_proba(X['test'][:, sel])[:, 1]) / 5
np.savez_compressed(R / f'results/feats_spec_{MODE}{SUF}.npz', cols=np.array([f'spec_{MODE}{SUF}']),
                    X_clean=oof['clean'][:, None], X_deg2=oof['deg2'][:, None], X_deg4=oof['deg4'][:, None], X_test=ste[:, None])
for w in ['clean', 'deg2']:
    print(f'specialist {MODE} alone ({w}): scint F1 at best threshold',
          round(max(f1_score(y == S, oof[w] > t) for t in np.linspace(-4, 4, 81)), 3), flush=True)

# nested stacking
oc = np.zeros((len(y), 9)); od = np.zeros((len(y), 9))
for k in range(5):
    t, v = fo != k, fo == k
    inner = {w: np.zeros(len(y)) for w in W}
    for j in range(5):
        if j == k: continue
        mdl = spec_fit((fo != k) & (fo != j))
        for w, a in score(mdl, fo == j).items(): inner[w][fo == j] = a
    A = lambda w, rows, s: np.hstack([X[w][rows], s[rows][:, None]])
    mdl = lgb.LGBMClassifier(**prm).fit(np.vstack([A(w, t, inner[w]) for w in W]), np.r_[y[t], y[t], y[t]])
    oc[v] = mdl.predict_proba(A('clean', v, oof['clean'])); od[v] = mdl.predict_proba(A('deg2', v, oof['deg2']))
    print('fold', k, 'done', flush=True)
f = lambda a: f1_score(y, a.argmax(1), average='macro')
print(f'NESTED stack spec_{MODE}{SUF} extras={sys.argv[1:]}  clean-val {f(oc):.4f}  fakePKS-val {f(od):.4f}')
print('per-class clean   ', dict(zip(L, np.round(f1_score(y, oc.argmax(1), average=None), 3))))
print('per-class fakePKS ', dict(zip(L, np.round(f1_score(y, od.argmax(1), average=None), 3))))
tag = f'nested_spec_{MODE}{SUF}' + ''.join('_' + Path(a).stem for a in sys.argv[1:])
np.savez_compressed(R / f'results/probs_{tag}.npz', oofc=oc, oofd=od)
