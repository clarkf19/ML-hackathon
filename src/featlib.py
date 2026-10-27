"""Portable (Linux/Windows) feature pipeline: raw uint8 cadences (N,6,16,128) + slew gaps -> the 427 bundle columns, in
bundle order (phys, tex, trk, rep, rep2, trk2, lvl). Reuses the feature functions of src/phys.py etc. unchanged, only the
I/O is new. z-features come from prep.normalise (+ optional quantile map to the test z marginal, as src/qmap.py);
lvl features come from the raw bytes (port of src/lvl.py).
usage (library): from featlib import bundle_features; F = bundle_features(raw, gaps, spike=True, qmap=False)"""
import numpy as np, pandas as pd, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prep, phys, tex, track, rep, rep2, trk2
ROOT = HERE.parent
DATA = ROOT / 'data'
SRC = DATA / 'signal-hunters-round2'
QS = np.r_[0.0001, 0.001, np.linspace(0.01, 0.99, 99), 0.999, 0.9999] * 100


def bundle_cols():
    return list(np.load(ROOT / 'bundle' / 'meta.npz', allow_pickle=True)['cols'])


def zquant(z, n=3000, seed=0):
    """quantiles of the non-missing z pixels on n random cadences (src/qmap.py q())"""
    idx = np.random.default_rng(seed).choice(len(z), min(n, len(z)), replace=False)
    v = np.asarray(z[np.sort(idx)], dtype=np.float32).ravel(); v = v[v != 0]
    return np.percentile(v, QS)


def test_zquant():
    f = DATA / 'zq_test.npy'
    if not f.exists():
        zt, _ = prep.normalise(np.load(SRC / 'test_cadences.npy', mmap_mode='r'), spike=False)
        np.save(f, zquant(zt))
    return np.load(f)


def qmap(Z, qa, qb):
    """monotone map source marginal (quantiles qa) -> target (qb); zeros (missing) stay 0 (src/qmap.py)"""
    O = np.zeros(Z.shape, np.float16)
    for s in range(0, len(Z), 2000):
        x = Z[s:s + 2000].astype(np.float32); m = x == 0
        y = np.interp(x, qa, qb)
        y = np.where(x > qa[-1], qb[-1] + (x - qa[-1]), y); y = np.where(x < qa[0], qb[0] + (x - qa[0]), y)
        y[m] = 0; O[s:s + 2000] = y
    return O


def _tmean(a, axis):
    s = np.sort(a, axis=axis); n = a.shape[axis]; k = n // 10
    return np.take(s, range(k, n - k), axis=axis).mean(axis)


def lvl_feats(X):
    """port of src/lvl.py (panel power level from raw bytes)"""
    rows = []
    for s0 in range(0, len(X), 1000):
        x = np.asarray(X[s0:s0 + 1000]).astype(np.float32); x[x == 0] = np.nan; b = len(x)
        chm = np.nanmedian(x, axis=2)
        ref = np.nanmedian(chm[:, 1::2], axis=1, keepdims=True)
        rel = np.nan_to_num(chm - ref)
        lv = _tmean(rel, 2)
        band = _tmean(rel.reshape(b, 6, 8, 16), 3)
        xr = np.nan_to_num(x - ref[:, :, None, :])
        tb = _tmean(xr.reshape(b, 6, 4, 4 * 128), 3)
        f = {}
        for p in range(6): f[f'lv_p{p}'] = lv[:, p] - lv[:, 1::2].mean(1)
        on, off = lv[:, 0::2], lv[:, 1::2]
        f['lv_on_off'] = on.mean(1) - off.mean(1); f['lv_onmin_offmax'] = on.min(1) - off.max(1)
        f['lv_onmin_offmean'] = on.min(1) - off.mean(1); f['lv_off_spread'] = off.max(1) - off.min(1); f['lv_on_spread'] = on.max(1) - on.min(1)
        bd = band[:, 0::2].mean(1) - band[:, 1::2].mean(1)
        for k in range(8): f[f'lv_band{k}'] = bd[:, k]
        f['lv_band_std'] = bd.std(1); f['lv_band_max'] = bd.max(1); f['lv_band_min'] = bd.min(1)
        f['lv_band_onmin'] = (band[:, 0::2].min(1) - band[:, 1::2].max(1)).max(1)
        td = tb[:, 0::2].reshape(b, -1); to = tb[:, 1::2].reshape(b, -1)
        f['lv_tblk_std_on'] = td.std(1); f['lv_tblk_std_off'] = to.std(1); f['lv_tblk_max_d'] = td.max(1) - to.max(1)
        rows.append(pd.DataFrame(f))
    return pd.concat(rows, ignore_index=True)


def z_features(Z, gaps, verbose=False):
    md = pd.DataFrame({'slew_gap_rows': np.asarray(gaps)})
    parts = []
    for nm, fn in [('phys', lambda: phys.feats(Z, md)), ('tex', lambda: tex.feats(Z)), ('trk', lambda: track.feats(Z, md)),
                   ('rep', lambda: rep.feats(Z)), ('rep2', lambda: rep2.feats(Z)), ('trk2', lambda: trk2.feats(Z, md))]:
        F = fn().add_prefix(nm + '_'); parts.append(F)
        if verbose: print(' ', nm, F.shape, flush=True)
    return pd.concat(parts, axis=1)


def bundle_features(raw, gaps, spike=True, qmap_to_test=False, raw_lvl=None, verbose=False):
    """raw: uint8 (N,6,16,128); gaps: (N,) slew gap rows. spike=True for GBT-like data (centre channel kept by its own
    median), False for PKS-like. qmap_to_test: quantile-map z to the test z marginal (fake-PKS recipe).
    raw_lvl: optional separate raw rendering for the lvl features (bundle deg copies use a GAINSD=1.9 rendering).
    Returns float32 (N,427) in bundle column order."""
    Z, _ = prep.normalise(raw, spike)
    if qmap_to_test: Z = qmap(Z, zquant(Z), test_zquant())
    F = z_features(Z, gaps, verbose)
    L = lvl_feats(raw if raw_lvl is None else raw_lvl).add_prefix('lvl_')
    F = pd.concat([F, L], axis=1)
    cols = bundle_cols()
    assert list(F.columns) == cols, (len(F.columns), [c for c in cols if c not in F.columns][:5])
    return F.values.astype(np.float32)
