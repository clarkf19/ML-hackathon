"""Generate N cadences (generator.py) and featurize them (featurize.py) -> OUT.npz with X (n,427), y, gaps (+ raw cadences
in data/gen/ if --keepraw). --balanced: equal count per class (calibration); --label=k: only class k; default: train mix.
--sets=phys,tex,...: subset of feature sets (others filled with NaN; for fast calibration loops).
Parameter overrides as JSON in env GEN_OVR, e.g. GEN_OVR='{"TEL.GBT.dof": 12, "SRC.eti.amp": [0.3, 3]}'.
usage: python src/gen_make.py N TEL SEED OUT.npz [--balanced] [--label=k] [--sets=...] [--keepraw]"""
import sys, os, json, time, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generator as G
from featurize import featurize, COLS

ALL = ('phys', 'tex', 'trk', 'rep', 'rep2', 'trk2', 'lvl')


def apply_overrides(ovr):
    for k, v in ovr.items():
        top, *path = k.split('.'); d = getattr(G, top)
        for p in path[:-1]: d = d[p]
        d[path[-1]] = tuple(v) if isinstance(v, list) and top in ('SRC',) else v


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    opt = dict(x[2:].split('=', 1) if '=' in x else (x[2:], True) for x in sys.argv[1:] if x.startswith('--'))
    n, tel, seed, out = int(a[0]), a[1], int(a[2]), a[3]
    apply_overrides(json.loads(os.environ.get('GEN_OVR', '{}')))
    labels = np.repeat(np.arange(9), n // 9) if 'balanced' in opt else (np.full(n, int(opt['label'])) if 'label' in opt else None)
    t = time.time(); X, g, y, info = G.generate(n if labels is None else len(labels), tel, seed, labels=labels)
    print(f'generated {X.shape} {tel} in {time.time() - t:.0f}s', flush=True)
    sets = tuple(opt['sets'].split(',')) if 'sets' in opt else ALL
    F = featurize(X, g, pks=(tel == 'PKS'), sets=sets).reindex(columns=COLS)   # missing sets -> NaN
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=F.values.astype(np.float32), y=y, gaps=g)
    info.to_csv(out.replace('.npz', '_info.csv'), index=False)
    if 'keepraw' in opt:
        d = Path('data/gen'); d.mkdir(parents=True, exist_ok=True); np.save(d / (Path(out).stem + '_cadences.npy'), X)
    print('saved', out, F.shape, flush=True)
