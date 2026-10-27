"""Pack selected raw statistics into a bundle_eval extra file: X_clean / X_test / X_deg2 / X_deg4 (+ cols).
usage: python src/make_rs_npz.py OUT.npz FEATLIST|all [DEGSUFFIX]
  FEATLIST = comma list or @file (one name per line); DEGSUFFIX '' (degrade only, default) or 'q' (degrade + raw qmap)"""
import numpy as np, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; D = ROOT / 'data'
out, fl = sys.argv[1], sys.argv[2]; sfx = sys.argv[3] if len(sys.argv) > 3 else ''
cols = list(np.load(D / 'rs_train.npz')['cols'])
sel = cols if fl == 'all' else (open(fl[1:]).read().split() if fl.startswith('@') else fl.split(','))
ix = [cols.index(c) for c in sel]
src = {'X_clean': 'train', 'X_test': 'test', 'X_deg2': 'd2' + sfx, 'X_deg4': 'd4' + sfx}
np.savez_compressed(out, cols=np.array(['rs_' + c for c in sel]), **{k: np.load(D / f'rs_{n}.npz')['X'][:, ix] for k, n in src.items()})
print(out, len(ix), 'features', 'deg suffix', repr(sfx))
