"""Residual-sweep fallback when the raw cadences are unavailable: cheap ON/OFF contrast statistics DERIVED from the bundle's
per-panel feature families (phys_*_p0..p5, tex_v*_p0..p5, trk_t_*0..5, rep2_*_p0..p5, lvl_lv_p0..p5). Trees see each
panel value separately, so pattern statistics across the 6 panels (adjacent ON-OFF steps, rank of ON vs OFF, scale-free
contrast, trend over the cadence, mid-cadence bump, ...) are information that is in the bundle but hard for LightGBM to form.
Computed identically on clean / test / deg2 / deg4, so the fake-PKS robustness number stays meaningful.
usage: python src/bundle_stats.py [out.npz]   (default data/bstats.npz; arrays X_clean X_test X_deg2 X_deg4 + names)"""
import re, sys, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parent.parent
B = R / 'bundle'
cols = list(np.load(B / 'meta.npz', allow_pickle=True)['cols'])
fam = {}
for i, c in enumerate(cols):
    mt = re.match(r'^(.*?)_?p?([0-5])$', c)
    if mt and (c.endswith(f'_p{mt.group(2)}') or re.match(r'^trk_t_(off|best)[0-5]$', c)):
        fam.setdefault(mt.group(1), {})[int(mt.group(2))] = i
fam = {k: [v[p] for p in range(6)] for k, v in fam.items() if len(v) == 6}
T = np.arange(6) - 2.5
ADJ = [(0, 1), (2, 1), (2, 3), (4, 3), (4, 5)]   # (ON, neighbouring OFF)

def stats(V):
    """V (n,6) one feature family over panels -> dict of contrast statistics."""
    on, off = V[:, 0::2], V[:, 1::2]
    sd = V.std(1) + 1e-6
    adj = np.stack([V[:, a] - V[:, b] for a, b in ADJ], 1)
    s = {}
    s['adjmin'] = adj.min(1)                                         # every ON above both neighbouring OFFs
    s['adjn'] = (adj > 0).sum(1).astype(np.float32)
    s['onrank'] = (on[:, :, None] > off[:, None, :]).sum((1, 2)).astype(np.float32)   # Mann-Whitney U, 0..9
    s['zalt'] = (on.mean(1) - off.mean(1)) / sd                      # scale-free ON-OFF contrast
    s['zadjmin'] = s['adjmin'] / sd
    s['slope'] = (V * T).sum(1) / (T ** 2).sum() / sd                # trend over the cadence
    s['onslope'] = (on[:, 2] - on[:, 0]) / sd
    s['offslope'] = (off[:, 2] - off[:, 0]) / sd
    s['bump'] = ((V[:, 2] + V[:, 3]) - (V[:, [0, 1, 4, 5]].sum(1)) / 2) / sd   # mid-cadence excess
    s['cv'] = V.std(1) / (np.abs(V.mean(1)) + 1e-3)
    s['argmax'] = V.argmax(1).astype(np.float32)
    s['on2_offmax'] = np.sort(on, 1)[:, 1] - off.max(1)              # >=2 ON panels above every OFF
    return s

def build(X):
    out, names = [], []
    for f, idx in fam.items():
        for k, v in stats(X[:, idx].astype(np.float64)).items(): out.append(v); names.append(f'bs_{f}_{k}')
    lv = X[:, fam['lvl_lv']]                                          # brightness vs texture co-variation over panels
    for f in [k for k in fam if k.startswith('tex_v')] + ['phys_tex_std']:
        tx = X[:, fam[f]]
        a, b = lv - lv.mean(1, keepdims=True), tx - tx.mean(1, keepdims=True)
        out.append((a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-9)); names.append(f'bs_corr_lv_{f}')
    return np.stack(out, 1).astype(np.float32), names

if __name__ == '__main__':
    dst = Path(sys.argv[1]) if len(sys.argv) > 1 else R / 'data' / 'bstats.npz'
    res = {}
    for w in ['clean', 'test', 'deg2', 'deg4']:
        res['X_' + w], names = build(np.load(B / f'feats_{w}.npz')['X'])
    print(len(fam), 'families:', list(fam)); print('stats', res['X_clean'].shape)
    np.savez_compressed(dst, names=np.array(names), **res)
