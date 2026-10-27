"""PKS vs GBT pixel-drift scale from bundle features: |drift| quantiles of strong detections (train vs fake-PKS vs test).
Fake-PKS deg2 keeps GBT drifts, so test/deg2 ratios estimate the PKS drift scale in pixel units."""
import numpy as np, pandas as pd
from pathlib import Path
B = Path(__file__).resolve().parent.parent / 'bundle'
m = np.load(B / 'meta.npz', allow_pickle=True); COLS = list(m['cols'])
qs = [.25, .5, .75, .9]
for w in ['clean', 'deg2', 'test']:
    X = pd.DataFrame(np.load(B / f'feats_{w}.npz')['X'], columns=COLS)
    bp = np.stack([X[f'phys_best_p{p}'] for p in range(6)], 1); bd = np.abs(np.stack([X[f'phys_bestd_p{p}'] for p in range(6)], 1))
    s = bp > 8; d = bd[s]; slow = d[(d > 0.1) & (d <= 1.0)]; steep = d[d > 1]
    t = X.trk_t_obj > 3; td = X.trk_t_drift[t]
    print(f'{w:6s} strong lines {s.sum():6d}: frac |d|>1 {np.mean(d > 1):.3f}  slow-drift q {np.round(np.quantile(slow, qs), 3).tolist()}  '
          f'steep q {np.round(np.quantile(steep, qs), 2).tolist()} | ON-tracks obj>3 n={t.sum()} drift q {np.round(np.quantile(td, qs), 3).tolist()}')
