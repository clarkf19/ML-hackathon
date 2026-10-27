"""Raw-byte scintillation detector (portable paths; NOT run in the cloud session that wrote it: drive.google.com was
blocked there, so it is smoke-tested on random bytes only — see results/REPORT.md).

Subcommands
  degrade NAME SEED [GRATE GLO GHI LNSTD]  bit-exact Linux/Windows port of src/degrade.py with GAINSD=1.9
                                           -> data/NAME_cadences.npy  (same RNG call order => same bytes as pipeline.sh)
  feats   RAW OUT                          RAW in {train, test} or a data/RAW_cadences.npy name -> data/fe_sraw_OUT.npz
  bundle                                   data/fe_sraw_{train,test,train_deg2,train_deg4}.npz -> results/feats_scint_raw.npz
Data dir = $H4H_DATA or <repo>/data ; competition files in <data>/signal-hunters-round2/.

Detector (per cadence, raw uint8 = log power, ~24 bytes per doubling of power as in degrade.py):
  * conservative mask: dropped rows/pixels, broadband glitch rows (row median or lit fraction), persistent narrow
    channels per panel (time-median vs 9-channel running median), isolated bright pixels > 4.5 sigma dilated +-1 ch.
    Scintles (~0.3 sigma per pixel) survive; tex3 masked whole detected tracks and removed them.
  * level (log and LINEAR power) ON-vs-OFF tests on masked data: difference, pooled t, Mann-Whitney pair count.
  * additive-source test: a sky source adds LINEAR power S, so its log-domain brightening 24*log2(1+S/N(f)) follows
    the inverse bandpass 1/N(f); multiplicative gain jitter is flat in frequency. Slope / t of the ON-OFF channel
    level on h(f)=2^(-(ref(f)-median)/24) separates the two (when the bandpass varies enough: sraw_h_spread).
    (Bundle evidence, src/scint_tail.py: the brightening is a pure log/byte-domain shift, no lower-tail compression, so
    this test is expected to be weak; kept as a cheap check.)
  * gain-free multi-scale texture: per-panel level removed, masked block means of z at 8 scales, variance excess
    and positive-tail excess, ON-OFF, ON-min-minus-OFF-max, pair count; log and linear domains.
  * overlapping matched filter at the measured scintle scale (bundle texture effect peaks at 8 rows x 16-32 channels):
    masked box-smoothed z (all offsets, not one block grid) -> mean square excess and 95th percentile, per panel.
  * ON-ON spectral/temporal consistency (same scintillation pattern across the 3 ON panels or independent)."""
import sys, os, numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter
R = Path(__file__).resolve().parent.parent
D = Path(os.environ.get('H4H_DATA', R / 'data')); SRC = D / 'signal-hunters-round2'
SC = [(2, 8), (4, 8), (4, 16), (8, 16), (8, 32), (16, 16), (16, 32), (16, 64)]

def raw_path(nm): return SRC / f'{nm}_cadences.npy' if nm in ('train', 'test') else D / f'{nm}_cadences.npy'

def degrade(name, seed, grate=0.0057, glo=18, ghi=50, lnstd=0.45, gainsd=1.9):
    from scipy.special import polygamma
    from scipy.optimize import brentq
    K = brentq(lambda k: polygamma(1, k) - (lnstd ** 2 - 0.343 ** 2), 1, 200)
    rng = np.random.default_rng(seed); X = np.load(raw_path('train'), mmap_mode='r'); out = np.zeros(X.shape, np.uint8)
    for s in range(0, len(X), 2000):
        x = X[s:s + 2000].astype(np.float32); miss = x == 0
        g = rng.gamma(K, 1 / K, x.shape).astype(np.float32)
        x = x + 24 * np.log2(g) + rng.normal(0, gainsd, (len(x), 6, 1, 1)).astype(np.float32)
        gl = rng.random((len(x), 6, 16)) < grate
        amp = rng.uniform(glo, ghi, gl.shape).astype(np.float32)
        lo = np.where(rng.random(gl.shape) < 0.3, rng.integers(0, 64, gl.shape), 0); hi = np.where(lo > 0, lo + rng.integers(40, 64, gl.shape), 128)
        ch = np.arange(128); band = (ch[None, None, None, :] >= lo[..., None]) & (ch[None, None, None, :] < hi[..., None])
        x = x + (gl[..., None] & band) * amp[..., None]
        drop = rng.random((len(x), 6, 16, 1)) < 0.0023
        x = np.clip(np.round(x), 1, 255); x[miss | np.broadcast_to(drop, x.shape)] = 0
        out[s:s + 2000] = x.astype(np.uint8)
    np.save(D / f'{name}_cadences.npy', out); print('degraded ->', name, 'K', round(K, 3))

def nmean(a, m, ax):  # mean of a over axes ax where mask m (True = keep)
    n = m.sum(ax); return np.where(n > 0, np.where(m, a, 0).sum(ax) / np.maximum(n, 1), 0.0), n

def pairs_gt(on, off): return (on[:, :, None] > off[:, None, :]).sum((1, 2)).astype(np.float32)

def tstat(on, off, floor):
    pooled = (((on - on.mean(1, keepdims=True)) ** 2).sum(1) + ((off - off.mean(1, keepdims=True)) ** 2).sum(1)) / 4
    return (on.mean(1) - off.mean(1)) / np.sqrt(pooled * 2 / 3 + floor)

def onoff(f, key, v, floor):
    on, off = v[:, 0::2], v[:, 1::2]
    f[key + '_d'] = on.mean(1) - off.mean(1); f[key + '_t'] = tstat(on, off, floor)
    f[key + '_u'] = pairs_gt(on, off); f[key + '_mm'] = on.min(1) - off.max(1)

def running_median9(a):  # along last axis, edges clamped
    p = np.concatenate([np.repeat(a[..., :1], 4, -1), a, np.repeat(a[..., -1:], 4, -1)], -1)
    return np.median(np.stack([p[..., k:k + a.shape[-1]] for k in range(9)], -1), -1)

def corr_rows(a, b, m):  # pearson corr per cadence of two (b,n) profiles on common valid entries
    a = np.where(m, a, 0); b = np.where(m, b, 0); n = np.maximum(m.sum(1), 1)
    am = a.sum(1) / n; bm = b.sum(1) / n; da = np.where(m, a - am[:, None], 0); db = np.where(m, b - bm[:, None], 0)
    return (da * db).sum(1) / np.sqrt((da ** 2).sum(1) * (db ** 2).sum(1) + 1e-9)

def feats(X, bs=500):
    out = []
    for s0 in range(0, len(X), bs):
        x = np.asarray(X[s0:s0 + bs]).astype(np.float32); b = len(x); f = {}
        valid = x > 0
        xn = np.where(valid, x, np.nan)
        chm = np.nanmedian(xn, 2)                                                  # (b,6,128)
        ref = np.nanmedian(chm[:, 1::2], 1)                                        # OFF bandpass (b,128)
        ref = np.where(np.isnan(ref), np.nanmedian(ref, 1, keepdims=True), ref)
        r = np.where(valid, x - ref[:, None, None, :], 0.0)
        lev0 = np.nanmedian(np.where(valid, r, np.nan), (2, 3))                     # (b,6)
        lev0 = np.nan_to_num(lev0)
        rr = np.where(valid, r - lev0[..., None, None], 0.0)
        sig = 1.4826 * np.nanmedian(np.where(valid, np.abs(rr), np.nan), (2, 3)); sig = np.nan_to_num(sig, nan=12.0)
        sg = np.median(sig, 1)[:, None, None, None]                                # cadence noise (bytes)
        # --- conservative mask (True = keep)
        rowmed = np.nanmedian(np.where(valid, rr, np.nan), 3)[..., None]           # (b,6,16,1)
        lit = ((rr > 1.5 * sg) & valid).mean(3, keepdims=True)
        glitch = (np.nan_to_num(rowmed) > 0.8 * sg) | (lit > 0.35)
        cm = np.nanmedian(np.where(valid, rr, np.nan), 2); cm = np.nan_to_num(cm)  # (b,6,128)
        dev = cm - running_median9(cm); mad = 1.4826 * np.median(np.abs(dev), 2, keepdims=True) + 1e-3
        line = dev > 4 * mad; line = line | np.roll(line, 1, -1) | np.roll(line, -1, -1)
        bright = rr > 4.5 * sg; bright = bright | np.roll(bright, 1, -1) | np.roll(bright, -1, -1)
        keep = valid & ~glitch & ~line[:, :, None, :] & ~bright
        keep[..., 63:66] &= False                                                  # GBT centre spike region
        kf = keep.mean((2, 3)); f['sraw_kept_on'] = kf[:, 0::2].mean(1); f['sraw_kept_off'] = kf[:, 1::2].mean(1)
        f['sraw_n_glitch'] = glitch[..., 0].sum((1, 2)).astype(np.float32); f['sraw_n_line'] = line.sum((1, 2)).astype(np.float32)
        # --- level: log and linear domains (gain jitter included; tests use pooled within-group variance)
        lv, _ = nmean(r, keep, (2, 3)); onoff(f, 'sraw_lv', lv, 0.25)
        P = np.exp2(r / 24.0); lp, _ = nmean(P, keep, (2, 3)); onoff(f, 'sraw_lin', 24 * np.log2(np.maximum(lp, 1e-3)), 0.25)
        f['sraw_lin_minus_log'] = f['sraw_lin_d'] - f['sraw_lv_d']                  # bright-scintle skew
        # --- additive-source (inverse-bandpass) test
        cmk, nk = nmean(r, keep, 2)                                                # (b,6,128) channel levels
        okc = (nk > 4).all(1)                                                      # channel usable in all panels
        dlt = cmk[:, 0::2].mean(1) - cmk[:, 1::2].mean(1)                          # (b,128)
        h = np.exp2(-(ref - np.median(ref, 1, keepdims=True)) / 24.0)
        n = np.maximum(okc.sum(1), 3); hm = np.where(okc, h, 0).sum(1) / n; dm = np.where(okc, dlt, 0).sum(1) / n
        hc = np.where(okc, h - hm[:, None], 0); dc = np.where(okc, dlt - dm[:, None], 0)
        sxx = (hc ** 2).sum(1) + 1e-9; slope = (hc * dc).sum(1) / sxx
        res = np.where(okc, dc - slope[:, None] * hc, 0); se = np.sqrt((res ** 2).sum(1) / np.maximum(n - 2, 1) / sxx)
        f['sraw_add_slope'] = slope; f['sraw_add_t'] = slope / (se + 1e-6); f['sraw_add_icpt'] = dm - slope * hm
        f['sraw_h_spread'] = np.sqrt(sxx / n)
        # per-panel additive slope (consistency over the 3 ON panels vs OFF)
        ps = np.stack([(hc * np.where(okc, cmk[:, p] - (np.where(okc, cmk[:, p], 0).sum(1) / n)[:, None], 0)).sum(1) / sxx for p in range(6)], 1)
        onoff(f, 'sraw_addp', ps, 0.01)
        # --- gain-free multi-scale texture (per-panel level removed), log and linear
        lvk, _ = nmean(r, keep, (2, 3))
        zl = np.where(keep, (r - lvk[..., None, None]) / sig[..., None, None], 0.0)
        Pk, _ = nmean(P, keep, (2, 3)); Pn = P / np.maximum(Pk, 1e-3)[..., None, None] - 1
        sl = 1.4826 * np.nanmedian(np.where(keep, np.abs(Pn - np.nanmedian(np.where(keep, Pn, np.nan), (2, 3), keepdims=True)), np.nan), (2, 3))
        zp = np.where(keep, Pn / np.nan_to_num(sl, nan=1.0)[..., None, None].clip(1e-3), 0.0)
        for dom, z in (('log', zl), ('lin', zp)):
            for (a, c) in SC:
                zs = z.reshape(b, 6, 16 // a, a, 128 // c, c).sum((3, 5)); ns = keep.reshape(b, 6, 16 // a, a, 128 // c, c).sum((3, 5))
                w = ns >= a * c * 0.5; zb = np.where(w, zs / np.sqrt(np.maximum(ns, 1)), 0)      # ~N(0,1) per block under noise
                nb = np.maximum(w.sum((2, 3)), 1)
                v = (zb ** 2).sum((2, 3)) / nb - 1; pos = (np.clip(zb, 0, None) ** 2).sum((2, 3)) / nb - 0.5
                onoff(f, f'sraw_{dom}{a}x{c}_v', v, 0.01); f[f'sraw_{dom}{a}x{c}_pos_d'] = pos[:, 0::2].mean(1) - pos[:, 1::2].mean(1)
        # --- overlapping matched filter (box kernel at scintle scale), masked: sum(z)/sqrt(n) over every window position
        kc = keep.astype(np.float32)
        for (a, c) in ((8, 16), (8, 32), (4, 16)):
            sz = uniform_filter(zl, (1, 1, a, c), mode='constant') * a * c; sn = uniform_filter(kc, (1, 1, a, c), mode='constant') * a * c
            ok = sn >= a * c * 0.5; zs = np.where(ok, sz / np.sqrt(np.maximum(sn, 1)), 0); no = np.maximum(ok.sum((2, 3)), 1)
            ms = (zs ** 2).sum((2, 3)) / no - 1
            p95 = np.percentile(np.where(ok, zs, np.nan).reshape(b, 6, -1), 95, axis=-1); p95 = np.nan_to_num(p95)
            onoff(f, f'sraw_mf{a}x{c}_ms', ms, 0.01); onoff(f, f'sraw_mf{a}x{c}_p95', p95, 0.01)
        # --- ON-panel pattern consistency: spectra (4-ch bins) and time profiles
        spec = zl.reshape(b, 6, 16, 32, 4).sum((2, 4)); sm = keep.reshape(b, 6, 16, 32, 4).sum((2, 4)) > 16
        tp = zl.sum(3); tm = keep.sum(3) > 32
        for nm, prof, msk in (('spec', spec, sm), ('time', tp, tm)):
            cc = lambda i, j: corr_rows(prof[:, i], prof[:, j], msk[:, i] & msk[:, j])
            onon = (cc(0, 2) + cc(0, 4) + cc(2, 4)) / 3; offoff = (cc(1, 3) + cc(1, 5) + cc(3, 5)) / 3
            onoff_ = (cc(0, 1) + cc(2, 3) + cc(4, 5)) / 3
            f[f'sraw_{nm}corr_onon'] = onon; f[f'sraw_{nm}corr_d'] = onon - offoff; f[f'sraw_{nm}corr_onoff'] = onoff_
        out.append({k: np.asarray(v, np.float32) for k, v in f.items()})
        if s0 % 5000 == 0: print(' ', s0, flush=True)
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'degrade':
        a = sys.argv[2:]; kw = dict(zip(['grate', 'glo', 'ghi', 'lnstd'], map(float, a[2:])))
        degrade(a[0], int(a[1]), **kw)
    elif cmd == 'feats':
        F = feats(np.load(raw_path(sys.argv[2]), mmap_mode='r'))
        np.savez_compressed(D / f'fe_sraw_{sys.argv[3]}.npz', cols=np.array(list(F)), X=np.stack(list(F.values()), 1))
        print(sys.argv[3], len(F), 'features')
    elif cmd == 'bundle':
        Z = {w: np.load(D / f'fe_sraw_{n}.npz') for w, n in [('clean', 'train'), ('test', 'test'), ('deg2', 'train_deg2'), ('deg4', 'train_deg4')]}
        np.savez_compressed(R / 'results/feats_scint_raw.npz', cols=Z['clean']['cols'], **{'X_' + w: z['X'] for w, z in Z.items()})
        print('-> results/feats_scint_raw.npz')
