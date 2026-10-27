"""Is the star's power added in LINEAR power (compresses the lower tail of the per-pixel log-power distribution in ON panels,
a gain-invariant signature) or in log/byte units (pure shift)? Checks the bundle's gain-free shape statistics
(tex_pos_d = ON-OFF fraction of pixels > 1.5 sigma after per-panel median removal; small-scale block variance) for scint
vs noise vs pulsar, and the expected size of the effect for a +4 byte additive brightening of log-chi2 noise.
usage: python src/scint_tail.py"""
import numpy as np
from pathlib import Path
R = Path(__file__).resolve().parent.parent
m = np.load(R / 'bundle/meta.npz', allow_pickle=True); y = m['y']; ci = {c: i for i, c in enumerate(m['cols'])}
q = lambda a: np.round(np.percentile(a, [10, 25, 50, 75, 90]), 3).tolist()
for w in ['clean', 'deg2']:
    X = np.load(R / f'bundle/feats_{w}.npz')['X']
    for c in ['tex_pos_d', 'tex_v2x4_d', 'tex_mean_d', 'phys_tex_std_on_off']:
        a = X[:, ci[c]]; print(f'{w:5s} {c:20s} scint {q(a[y == 3])}  noise {q(a[y == 8])}  pulsar {q(a[y == 2])}')
    a = X[:, ci['tex_pos_d']]; lv = X[:, ci['lvl_lv_on_off']]; s = y == 3
    print(f'{w:5s} corr(tex_pos_d, lv_on_off) within scint {np.corrcoef(a[s], lv[s])[0, 1]:.3f}; within noise {np.corrcoef(a[y == 8], lv[y == 8])[0, 1]:.3f}')
# expectation for a uniform additive star: GBT-like noise = 24*log2(chi2_18/18); star adds S = 2^(4/24)-1 of the mean power
rng = np.random.default_rng(0); n = rng.chisquare(18, 2_000_000) / 18; S = 2 ** (4 / 24) - 1
for nm, b in [('log-domain shift (+4 bytes)', 24 * np.log2(n) + 4), ('linear additive (+4 bytes mean)', 24 * np.log2(n + S))]:
    r = 24 * np.log2(n); d = np.percentile(b, [10, 50, 90]) - np.percentile(r, [10, 50, 90])
    sd0 = 1.4826 * np.median(np.abs(r - np.median(r))); sd1 = 1.4826 * np.median(np.abs(b - np.median(b)))
    print(f'{nm:32s}: shift of q10/q50/q90 {np.round(d, 2)} bytes; robust sd {sd0:.2f} -> {sd1:.2f};'
          f' frac > med+1.5sd {np.mean(r > np.median(r) + 1.5 * sd0):.4f} -> {np.mean(b > np.median(b) + 1.5 * sd0):.4f}')
