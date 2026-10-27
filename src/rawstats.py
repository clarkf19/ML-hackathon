"""Residual-information sweep: ~270 cheap RAW-domain statistics per cadence that the bundle features do not contain.
Log domain D = bytes minus the OFF-panel per-channel median (bandpass flattened, panel level KEPT); linear domain
R = power / OFF per-channel median power (power = 2**(v/24-4)). Per-panel statistics are summarised as
_on (mean of ON panels), _off (mean of OFF), _d (on - off), _mm (min ON - max OFF).
usage: python src/rawstats.py NAME [NAME ...]   NAME 'train'/'test' = competition files, else data/raw_NAME.npy
       -> data/rs_NAME.npz (X float32, cols)"""
import numpy as np, sys, time
from pathlib import Path
from scipy.ndimage import median_filter, uniform_filter1d
ROOT = Path(__file__).resolve().parent.parent
D_ = ROOT / 'data'; SRC = D_ / 'signal-hunters-round2'
ON, OFF = slice(0, None, 2), slice(1, None, 2)
PAIRS = {'onon': [(0, 2), (0, 4), (2, 4)], 'offoff': [(1, 3), (1, 5), (3, 5)],
         'onoff': [(a, b) for a in (0, 2, 4) for b in (1, 3, 5)], 'adj': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]}

def summ(f, nm, A, which='oodm'):
    on, off = A[:, ON], A[:, OFF]
    if 'o' in which: f[nm + '_on'] = np.nanmean(on, 1); f[nm + '_off'] = np.nanmean(off, 1)
    if 'd' in which: f[nm + '_d'] = np.nanmean(on, 1) - np.nanmean(off, 1)
    if 'm' in which: f[nm + '_mm'] = np.nanmin(on, 1) - np.nanmax(off, 1)

def moments(a, axis):
    m = np.nanmean(a, axis, keepdims=True); d = a - m; v = np.nanmean(d ** 2, axis)
    return m.squeeze(axis), np.sqrt(v), np.nanmean(d ** 3, axis) / (v ** 1.5 + 1e-9), np.nanmean(d ** 4, axis) / (v ** 2 + 1e-9) - 3

def fillnan(a, axis):
    return np.where(np.isnan(a), np.nanmedian(a, axis, keepdims=True), a)

def ac(a, lag=1):       # autocorrelation along the last axis (no nan)
    a = a - a.mean(-1, keepdims=True)
    return (a[..., lag:] * a[..., :-lag]).sum(-1) / ((a ** 2).sum(-1) + 1e-9)

def pcorr(prof):       # (b,6,n) -> (b,6,6) Pearson correlation between panel profiles
    p = prof - prof.mean(-1, keepdims=True); p = p / (np.sqrt((p ** 2).sum(-1, keepdims=True)) + 1e-9)
    return np.einsum('bpc,bqc->bpq', p, p)

def corrfeats(f, nm, C):
    for k, pr in PAIRS.items():
        v = np.stack([C[:, a, b] for a, b in pr], 1); f[f'{nm}_{k}'] = v.mean(1)
        if k == 'adj': f[f'{nm}_adjmin'] = v.min(1)
    f[f'{nm}_onon_m_onoff'] = f[f'{nm}_onon'] - f[f'{nm}_onoff']

def feats(X):
    v = X.astype(np.float32); miss = v == 0; v[miss] = np.nan; b = len(v)
    refL = np.nanmedian(v[:, OFF].transpose(0, 3, 1, 2).reshape(b, 128, -1), 2)              # (b,128) OFF bandpass, bytes
    D = v - refL[:, None, None, :]                                                           # (b,6,16,128)
    P = 2 ** (v / 24 - 4); refP = 2 ** (refL / 24 - 4); R = P / refP[:, None, None, :]
    Dp = D.reshape(b, 6, -1); Rp = R.reshape(b, 6, -1); D0 = np.nan_to_num(D)
    f = {}
    # G1 pixel distribution per panel, log domain
    mu, sd, sk, ku = moments(Dp, 2)
    for nm, A in [('px_mean', mu), ('px_std', sd), ('px_skew', sk), ('px_kurt', ku)]: summ(f, nm, A)
    qs = np.nanpercentile(Dp, [1, 5, 25, 50, 75, 95, 99, 100], axis=2)                       # (8,b,6)
    for q, A in zip(['q01', 'q05', 'q25', 'q50', 'q75', 'q95', 'q99', 'qmax'], qs): summ(f, 'px_' + q, A)
    rsd = 1.4826 * np.nanmedian(np.abs(Dp - qs[3][..., None]), 2); summ(f, 'px_rsd', rsd)
    f['cad_rsd'] = np.nanmedian(rsd, 1)
    # G2 linear power per panel
    mu, sd, sk, ku = moments(Rp, 2)
    summ(f, 'lin_logmean', np.log(mu)); summ(f, 'lin_std', sd); summ(f, 'lin_skew', sk); summ(f, 'lin_kurt', ku)
    summ(f, 'lin_logq99', np.log(np.nanpercentile(Rp, 99, axis=2)))
    # G3 row (time) profiles per panel
    rowmed = np.nanmedian(D, 3); rowL = fillnan(rowmed, 2); rowP = fillnan(np.log(np.nanmean(R, 3)), 2)       # (b,6,16)
    t = np.arange(16) - 7.5
    for nm, A in [('rowL', rowL), ('rowP', rowP)]:
        summ(f, nm + '_std', A.std(2)); summ(f, nm + '_ac1', ac(A)); summ(f, nm + '_rng', A.max(2) - A.min(2))
        summ(f, nm + '_exc', A.max(2) - np.median(A, 2)); summ(f, nm + '_slope', np.abs((A * t).sum(2) / (t ** 2).sum()))
    # G4 channel profiles per panel
    chL = fillnan(np.nanmedian(D, 2), 2); chP = fillnan(np.log(np.nanmean(R, 2)), 2)         # (b,6,128)
    for nm, A in [('chL', chL), ('chP', chP)]:
        summ(f, nm + '_std', A.std(2)); summ(f, nm + '_ac1', ac(A)); summ(f, nm + '_ac4', ac(A, 4))
        summ(f, nm + '_kurt', moments(A, 2)[3]); summ(f, nm + '_max', A.max(2))
    # G5 ON-minus-OFF channel profile
    for nm, A, th in [('dchL', chL, 2.0), ('dchP', chP, 0.05)]:
        d = A[:, ON].mean(1) - A[:, OFF].mean(1)
        f[nm + '_std'] = d.std(1); f[nm + '_ac1'] = ac(d); f[nm + '_ac4'] = ac(d, 4); f[nm + '_max'] = d.max(1)
        f[nm + '_min'] = d.min(1); f[nm + '_skew'] = moments(d, 1)[2]; f[nm + '_fpos'] = (d > th).mean(1)
    # G6 time x sub-band blocks (log domain, level kept)
    for a, c in [(2, 32), (4, 16), (8, 8)]:
        bm = D0.reshape(b, 6, 16 // a, a, 128 // c, c).mean((3, 5)).reshape(b, 6, -1)
        summ(f, f'blk{a}x{c}_std', bm.std(2))
        dd = bm[:, ON].mean(1) - bm[:, OFF].mean(1)
        f[f'blk{a}x{c}_dstd'] = dd.std(1); f[f'blk{a}x{c}_dmax'] = dd.max(1); f[f'blk{a}x{c}_dmin'] = dd.min(1)
        f[f'blk{a}x{c}_mmmax'] = (bm[:, ON].min(1) - bm[:, OFF].max(1)).max(1)
    # G7 dropped / glitch / clipping by panel
    drop = miss.all(3)                                                                        # (b,6,16)
    f['drop_on'] = drop[:, ON].sum((1, 2)); f['drop_off'] = drop[:, OFF].sum((1, 2)); f['drop_edge'] = drop[:, :, [0, 15]].sum((1, 2))
    rz = np.nan_to_num((rowmed - np.nanmedian(rowmed, 2, keepdims=True)) / (rsd[..., None] * 1.2533 / np.sqrt(128) + 1e-6))
    for k in (5, 10): g = (rz > k).sum(2); f[f'glitch{k}_on'] = g[:, ON].sum(1); f[f'glitch{k}_off'] = g[:, OFF].sum(1)
    f['dip5_on'] = (rz < -5).sum(2)[:, ON].sum(1); f['dip5_off'] = (rz < -5).sum(2)[:, OFF].sum(1)
    gm = rz.max(2); f['glitchamp_on'] = gm[:, ON].max(1); f['glitchamp_off'] = gm[:, OFF].max(1)
    q75r = np.nan_to_num(np.nanpercentile(D, 75, axis=3) - rowmed)                          # partial-band glitch rows
    summ(f, 'rowq75_max', (q75r - np.median(q75r, 2, keepdims=True)).max(2), 'dm')
    f['v1_on'] = (v[:, ON] == 1).mean((1, 2, 3)); f['v1_off'] = (v[:, OFF] == 1).mean((1, 2, 3))
    summ(f, 'vmax', np.nanmax(v.reshape(b, 6, -1), 2), 'od')
    # G8 inter-panel correlations of channel profiles (all / narrow / broad) and of 2D residual maps
    hp = chL - median_filter(chL, size=(1, 1, 9), mode='nearest'); lp = uniform_filter1d(chL, 16, axis=2, mode='nearest')
    corrfeats(f, 'cc', pcorr(chL)); corrfeats(f, 'cchp', pcorr(hp)); corrfeats(f, 'cclp', pcorr(lp))
    M = (D0 - chL[:, :, None, :]).reshape(b, 6, -1); C = pcorr(M)
    for k in ('onon', 'offoff', 'onoff'): f[f'cc2d_{k}'] = np.stack([C[:, a, c] for a, c in PAIRS[k]], 1).mean(1)
    # G9 edge channels / centre channel
    cen = np.nanmedian(v[..., 40:88].reshape(b, -1), 1)
    f['edge_lo'] = np.nanmean(v[..., :8].reshape(b, -1), 1) - cen; f['edge_hi'] = np.nanmean(v[..., -8:].reshape(b, -1), 1) - cen
    ed = np.nanmean(D[..., np.r_[0:8, 120:128]].reshape(b, 6, -1), 2); summ(f, 'edgeD', ed, 'dm')
    f['c64_d'] = np.nanmean(D[:, ON, :, 64], (1, 2)) - np.nanmean(D[:, OFF, :, 64], (1, 2))
    # G10 byte-value quantisation patterns
    h = np.zeros((b * 6, 256), np.float32); vi = X.reshape(b * 6, -1).astype(np.int64)
    np.add.at(h, (np.repeat(np.arange(b * 6), vi.shape[1]), vi.ravel()), 1); h[:, 0] = 0
    p = h / np.maximum(h.sum(1, keepdims=True), 1)
    summ(f, 'q_distinct', (h > 0).sum(1).reshape(b, 6), 'od'); summ(f, 'q_entropy', -(p * np.log(p + 1e-12)).sum(1).reshape(b, 6), 'od')
    f['q_even'] = p[:, 0::2].sum(1).reshape(b, 6).mean(1) - 0.5
    f['q_mod24'] = np.abs((p.reshape(-1, 256)[:, :240].reshape(-1, 10, 24).sum(1) * 24 - 1)).mean(1).reshape(b, 6).mean(1)
    # G11 spectral kurtosis (linear power)
    skc = np.nanvar(P, 2) / (np.nanmean(P, 2) ** 2 + 1e-12)                                  # (b,6,128) per channel over time
    summ(f, 'skc_med', np.nanmedian(skc, 2), 'odm'); summ(f, 'skc_max', np.nanmax(skc, 2), 'odm')
    summ(f, 'skc_frac', (skc > 2 * np.nanmedian(skc, (1, 2))[:, None, None]).mean(2), 'od')
    skr = np.nanvar(R, 3) / (np.nanmean(R, 3) ** 2 + 1e-12)                                  # (b,6,16) per row over channels
    summ(f, 'skr_med', np.nanmedian(skr, 2), 'odm'); summ(f, 'skr_max', np.nanmax(skr, 2), 'odm')
    # G12 pixel-level time / frequency correlation (log domain, channel profile removed)
    Mt = D0 - chL[:, :, None, :]
    summ(f, 'pac_t1', ac(Mt.transpose(0, 1, 3, 2)).mean(2)); summ(f, 'pac_f1', ac(Mt).mean(2))
    # G13 tail fractions
    for nm, th in [('fup24', 24), ('fup48', 48)]: summ(f, nm, (Dp > th).mean(2), 'odm')
    summ(f, 'fdn36', (Dp < -36).mean(2), 'od')
    return f

def run(X, bs=500):
    out = [];  cols = None
    for s in range(0, len(X), bs):
        f = feats(np.asarray(X[s:s + bs])); cols = list(f); out.append(np.stack([f[c] for c in cols], 1).astype(np.float32))
    return np.vstack(out), cols

if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')
    for nm in sys.argv[1:]:
        t0 = time.time()
        X = np.load(SRC / f'{nm}_cadences.npy' if nm in ('train', 'test') else D_ / f'raw_{nm}.npy', mmap_mode='r')
        F, cols = run(X); np.savez_compressed(D_ / f'rs_{nm}.npz', X=F, cols=np.array(cols))
        print(nm, F.shape, f'{time.time() - t0:.0f}s', 'nan frac', np.isnan(F).mean().round(4), flush=True)
