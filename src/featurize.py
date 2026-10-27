"""Portable (Linux/Windows) feature extraction for ANY raw cadence array: uint8 (N,6,16,128) + slew gaps (N,)
-> the 427 bundle columns, in bundle/meta.npz['cols'] order. Reuses the unchanged feature code of
prep.normalise / phys / tex / track / rep / rep2 / trk2 (imported) and lvl.py (copied: it runs at import time).
Speed-up only (same arithmetic): cadences are processed in slew-gap-homogeneous groups and the per-(gap,panel) track
gathers of phys.global_track / track.track_sums / trk2 are done as ONE gather over all 6 panels (check_equal() verifies
against the original code). GBT-like data: prep with the ch-64 spike handling (as train / train_deg*); PKS-like data
(--pks): without (as test).
usage: python src/featurize.py CADENCES.npy GAPS.npy OUT.npz [--pks]"""
import sys, os, time, numpy as np, pandas as pd
from pathlib import Path
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
os.environ.setdefault('NT', str(os.cpu_count() or 4))
import torch, torch.nn.functional as F
torch.set_num_threads(int(os.environ['NT']))
from prep import normalise
import phys, tex, track, rep, rep2, trk2
COLS = list(np.load(SRC.parent / 'bundle' / 'meta.npz', allow_pickle=True)['cols'])
ORIG = dict(global_track=phys.global_track, track_sums=track.track_sums, trk2_feats=trk2.feats)


def _chidx(gval, d, cref, W):
    """(6,16,ncref) channel index of straight global tracks, built with the original per-panel expressions."""
    tref = 2 * (16 + gval) + 7.5; out = []
    for p in range(6):
        t = p * (16 + gval) + torch.arange(16) - tref
        out.append((torch.round(cref[None, :] + d * t[:, None]).long() + 128).clamp(0, W - 1))
    return torch.stack(out)


def _gather6(zp, idx):
    """zp (b,6,16,W), idx (6,16,ncref) -> (b,6,ncref) track sums / 4"""
    b = zp.shape[0]
    return torch.gather(zp, 3, idx[None].expand(b, *idx.shape)).sum(2) / 4.0


def track_sums_fast(zp, gaps, d, cref, W):
    g = np.unique(gaps)
    if len(g) != 1: return ORIG['track_sums'](zp, gaps, d, cref, W)
    return _gather6(zp, _chidx(g[0], d, cref, W))


def global_track_fast(zn, gaps):
    g = np.unique(gaps)
    if len(g) != 1: return ORIG['global_track'](zn, gaps)
    b = zn.shape[0]; W = 128 + 2 * 128
    zp = F.pad(zn, (128, 128))
    best_min = torch.full((b,), -1e9); best_diff = torch.full((b,), -1e9)
    rec_min = torch.zeros(b, 8); rec_diff = torch.zeros(b, 8)
    out_all = torch.full((b,), -1e9)
    cref = torch.arange(-64, 192)
    for d in phys.D2:
        P = _gather6(zp, _chidx(g[0], d, cref, W))
        on, off = P[:, 0::2], P[:, 1::2]
        mn = on.min(1).values
        diff = on.sum(1) / 1.732 - off.clamp(min=0).max(1).values * 1.732
        allp = P.min(1).values
        for score, best, rec in [(mn, best_min, rec_min), (diff, best_diff, rec_diff)]:
            v, i = score.max(-1); upd = v > best
            best[upd] = v[upd]
            sel = P[torch.arange(b), :, i]
            rec[upd] = torch.cat([sel, torch.full((b, 1), float(d)), cref[i][:, None].float()], 1)[upd]
        out_all = torch.maximum(out_all, allp.max(-1).values)
    return best_min, rec_min, best_diff, rec_diff, out_all


def trk2_feats_fast(Z, md, bs=400):
    """trk2.feats with the one-gather track sums (same arithmetic)."""
    gaps_all = md.slew_gap_rows.values
    if len(np.unique(gaps_all)) != 1: return ORIG['trk2_feats'](Z, md, bs)
    gval = np.unique(gaps_all)[0]; CREF = trk2.CREF; W = 384; out = []
    tref = 2 * (16 + gval) + 7.5
    IBall = {}
    for s0 in range(0, len(Z), bs):
        z = torch.from_numpy(np.asarray(Z[s0:s0 + bs], dtype=np.float32)); b = len(z)
        valid = (z != 0).any(-1)
        zn = phys.smooth_f((z - z.median(-1, keepdim=True).values) * valid[..., None]).clamp(-3, 6)
        zp = F.pad(zn, (128, 128))
        best = {k: torch.full((b,), -1e9) for k in ['inmin', 'in2nd', 'npres', 'absent_in']}
        rec = torch.zeros(b, 4)
        for d in trk2.D2:
            if d not in IBall:
                ib = []
                for p in range(6):
                    t = p * (16 + gval) + torch.arange(16) - tref
                    chf = torch.round(CREF[None, :] + float(d) * t[:, None])
                    ib.append(((chf >= 0) & (chf <= 127)).float().mean(0))
                IBall[d] = torch.stack(ib)                                              # (6,ncref)
            idx = []
            for p in range(6):
                t = p * (16 + gval) + torch.arange(16) - tref
                chf = torch.round(CREF[None, :] + float(d) * t[:, None])
                idx.append((chf.long() + 128).clamp(0, W - 1))
            P = _gather6(zp, torch.stack(idx)); IB = IBall[d][None].expand(b, -1, -1)
            inb = IB > 0.6; ninb = inb.sum(1)
            Pin = torch.where(inb, P, torch.tensor(1e9)); srt = Pin.sort(1).values
            inmin = torch.where(ninb >= 3, srt[:, 0], torch.tensor(-1e9))
            in2nd = torch.where(ninb >= 3, srt[:, 1], torch.tensor(-1e9))
            pres = (P > 3) & inb; npres = pres.sum(1).float()
            absent = (inb & ~(P > 3)).sum(1).float()
            v, i = in2nd.max(-1); upd = v > best['in2nd']
            best['in2nd'][upd] = v[upd]
            ar = torch.arange(b)
            cand = torch.stack([inmin[ar, i], npres[ar, i], absent[ar, i], ninb[ar, i].float()], 1)
            rec[upd] = cand[upd]
            best['inmin'] = torch.maximum(best['inmin'], inmin.max(-1).values)
            strength = torch.where(pres, P, torch.tensor(0.0)).sum(1)
            v2, i2 = strength.max(-1); upd2 = v2 > best['npres']
            best['npres'][upd2] = v2[upd2]
            best['absent_in'][upd2] = torch.where(upd2, absent[ar, i2] * 100 + npres[ar, i2], best['absent_in'])[upd2]
        f = {'g2_inmin': best['inmin'], 'g2_in2nd': best['in2nd'], 'g2_rec_inmin': rec[:, 0], 'g2_rec_npres': rec[:, 1],
             'g2_rec_absent': rec[:, 2], 'g2_rec_ninb': rec[:, 3], 'g2_strong_sum': best['npres'],
             'g2_strong_absent': torch.div(best['absent_in'], 100, rounding_mode='floor'), 'g2_strong_npres': best['absent_in'] % 100}
        out.append(pd.DataFrame({k: v.numpy() for k, v in f.items()}))
    return pd.concat(out, ignore_index=True)


def use_fast(on=True):
    phys.global_track = global_track_fast if on else ORIG['global_track']
    track.track_sums = track_sums_fast if on else ORIG['track_sums']


def _tmean(a, axis):  # 10% trimmed mean (lvl.py)
    s = np.sort(a, axis=axis); n = a.shape[axis]; k = n // 10
    return np.take(s, range(k, n - k), axis=axis).mean(axis)


def lvl_feats(X, bs=1000):
    """lvl.py verbatim, as a function of the raw uint8 array."""
    import warnings
    rows = []
    for s0 in range(0, len(X), bs):
        x = np.asarray(X[s0:s0 + bs]).astype(np.float32); x[x == 0] = np.nan; b = len(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
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


def _featurize_group(X, z, gaps, sets, verbose):
    md = pd.DataFrame({'slew_gap_rows': np.asarray(gaps).astype(int)})
    fn = {'phys': lambda: phys.feats(z, md), 'tex': lambda: tex.feats(z), 'trk': lambda: track.feats(z, md),
          'rep': lambda: rep.feats(z), 'rep2': lambda: rep2.feats(z), 'trk2': lambda: trk2_feats_fast(z, md), 'lvl': lambda: lvl_feats(X)}
    parts = []
    for s in sets:
        t = time.time(); parts.append(fn[s]().add_prefix(s + '_'))
        if verbose: print(f'    {s}: {time.time() - t:.0f}s', flush=True)
    return pd.concat(parts, axis=1)


def featurize(X, gaps, pks=False, sets=('phys', 'tex', 'trk', 'rep', 'rep2', 'trk2', 'lvl'), verbose=True, fast=True):
    """X raw uint8 (N,6,16,128), gaps (N,) -> DataFrame with bundle column names (subset if sets is restricted)."""
    t0 = time.time(); use_fast(fast)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext():
        X = np.asarray(X); gaps = np.asarray(gaps).astype(int)
        z, _ = normalise(X, spike=not pks)
        out = []
        for gv in np.unique(gaps):
            ix = np.where(gaps == gv)[0]
            if verbose: print(f'  gap {gv}: {len(ix)} cadences', flush=True)
            Fg = _featurize_group(X[ix], z[ix], gaps[ix], sets, verbose); Fg.index = ix; out.append(Fg)
        Fd = pd.concat(out).sort_index()
    if set(sets) == {'phys', 'tex', 'trk', 'rep', 'rep2', 'trk2', 'lvl'}:
        assert list(Fd.columns) == COLS, 'column mismatch with bundle'
    if verbose: print(f'featurize {len(X)} cadences: {time.time() - t0:.0f}s', flush=True)
    return Fd.reset_index(drop=True)


def check_equal(X, gaps, pks=False):
    """fast path vs original code on the same cadences (original: mixed-gap batches, per-panel gathers)."""
    z, _ = normalise(np.asarray(X), spike=not pks); md = pd.DataFrame({'slew_gap_rows': np.asarray(gaps).astype(int)})
    use_fast(False)
    ref = pd.concat([phys.feats(z, md).add_prefix('phys_'), track.feats(z, md).add_prefix('trk_'),
                     ORIG['trk2_feats'](z, md).add_prefix('trk2_')], axis=1)
    fast = featurize(X, gaps, pks, sets=('phys', 'trk', 'trk2'), verbose=False)
    d = (ref - fast[ref.columns]).abs().max().sort_values(ascending=False)
    return d


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    X = np.load(a[0], mmap_mode='r'); g = np.load(a[1])
    Fd = featurize(X, g, pks='--pks' in sys.argv)
    np.savez_compressed(a[2], X=Fd.values.astype(np.float32))
    print('saved', a[2], Fd.shape)
