"""bundle_eval.py with EXTRA TRAINING ROWS: generated cadences (features + labels) are appended to the training set of
every fold (they are not real train rows, so no fold leakage). Scored exactly like bundle_eval: clean val and fake-PKS
(deg2) val, same 5 folds, same LightGBM params; prints the baseline (no extra rows) in the same run unless --nobase.
usage: python src/gen_aug_eval.py GEN1.npz [GEN2.npz ...] [--w=0.5] [--nobase]   (npz: X (n,427), y (n,))
--w: sample weight of generated rows (real rows weight 1)."""
import sys, numpy as np, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import f1_score
B = Path(__file__).resolve().parent.parent / 'bundle'
m = np.load(B / 'meta.npz', allow_pickle=True); y, fo, L = m['y'], m['folds'], list(m['labels'])
X = {w: np.load(B / f'feats_{w}.npz')['X'] for w in ['clean', 'test', 'deg2', 'deg4']}
args = [a for a in sys.argv[1:] if not a.startswith('--')]
W = float(next((a.split('=')[1] for a in sys.argv if a.startswith('--w=')), 1.0))
G = [np.load(a) for a in args]; XG = np.vstack([g['X'] for g in G]) if G else np.zeros((0, X['clean'].shape[1]))
YG = np.concatenate([g['y'] for g in G]) if G else np.zeros(0, int)
prm = dict(n_estimators=900, learning_rate=0.05, num_leaves=31, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
           class_weight='balanced', verbose=-1)
f = lambda a: f1_score(y, a.argmax(1), average='macro')


def run(extra):
    oc = np.zeros((len(y), 9)); od = np.zeros((len(y), 9)); pt = np.zeros((10000, 9))
    for k in range(5):
        t, v = fo != k, fo == k
        XS = [X['clean'][t], X['deg2'][t], X['deg4'][t]]; YS = [y[t], y[t], y[t]]; WS = [np.ones(3 * t.sum())]
        if extra: XS.append(XG); YS.append(YG); WS.append(np.full(len(YG), W))
        mdl = lgb.LGBMClassifier(**prm).fit(np.vstack(XS), np.concatenate(YS), sample_weight=np.concatenate(WS))
        oc[v] = mdl.predict_proba(X['clean'][v]); od[v] = mdl.predict_proba(X['deg2'][v]); pt += mdl.predict_proba(X['test']) / 5
    return oc, od, pt


lines = []
for tag, extra in ([('baseline', False)] if '--nobase' not in sys.argv else []) + [('+gen ' + ','.join(Path(a).stem for a in args) + f' w={W}', True)]:
    oc, od, pt = run(extra)
    s = (f'{tag}: n_extra {len(YG) if extra else 0}  clean-val {f(oc):.4f}  fakePKS-val {f(od):.4f}\n'
         f'   per-class clean   {dict(zip(L, np.round(f1_score(y, oc.argmax(1), average=None), 3)))}\n'
         f'   per-class fakePKS {dict(zip(L, np.round(f1_score(y, od.argmax(1), average=None), 3)))}\n'
         f'   test pred mix {dict(zip(L, np.round(np.bincount(pt.argmax(1), minlength=9) / len(pt), 3)))}')
    print(s, flush=True); lines.append(s)
    if extra:
        out = Path('results'); out.mkdir(exist_ok=True)
        np.savez_compressed(out / f'probs_aug_{"_".join(Path(a).stem for a in args)}.npz', oofc=oc, oofd=od, pte=pt)
open('results/aug_eval_log.txt', 'a').write('\n'.join(lines) + '\n')
