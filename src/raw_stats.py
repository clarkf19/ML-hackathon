"""Residual-information sweep: ~170 cheap RAW-domain statistics per cadence that the bundle features do not contain
(prep.py removes bandpass, per-panel level and noise scale before every other feature; lvl.py keeps only trimmed-mean levels).
Bandpass bp = per-cadence channel median over panels+time; r = x - bp (log units, 24 bytes = x2 power); lr = 2**(r/24) (linear
power relative to bandpass); P = 2**(x/24-4) (absolute linear power). Dropped pixels (byte 0) are ignored.
Groups (name rs_{group}_...): lvl (linear/log level and quantiles ON vs OFF), var (std/MAD/modulation index), shape
(skew/kurtosis), frac (zero/saturation/low-clip), rows (glitch & dropped rows by panel, row variability, trend), spec (ON-OFF
spectrum, edge channels, inter-panel spectral correlation, bandpass shape), band (sub-band linear ON/OFF), tblk (time-block ON/OFF),
acf (lag-1 time/frequency autocorrelation), byte (quantisation pattern of byte values).
usage: python src/raw_stats.py out.npz clean=PATH test=PATH deg2=PATH deg4=PATH   (PATH = .npy uint8 N x 6 x 16 x 128)"""
import sys, warnings, numpy as np
from pathlib import Path
ON, OFF = [0, 2, 4], [1, 3, 5]
QS = [.01, .1, .25, .5, .75, .9, .99]
for msg in ['Mean of empty slice', 'Degrees of freedom', 'All-NaN slice']: warnings.filterwarnings('ignore', msg)   # dropped rows/panels -> NaN

def nanq(a, qs):
    """quantiles along the last axis ignoring NaN (sort-based, nearest rank) -> (..., len(qs))."""
    s = np.sort(a, -1); n = (~np.isnan(a)).sum(-1)
    idx = np.clip(np.round(np.asarray(qs)[None] * (n[..., None] - 1)), 0, a.shape[-1] - 1).astype(int).reshape(*n.shape, len(qs))
    out = np.take_along_axis(s, idx, -1); out[n == 0] = np.nan; return out

def nmean(a, ax):
    with np.errstate(invalid='ignore', divide='ignore'): return np.nanmean(a, ax)

def moments(a):
    m = nmean(a, -1); d = a - m[..., None]; m2 = nmean(d ** 2, -1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return m, np.sqrt(m2), nmean(d ** 3, -1) / m2 ** 1.5, nmean(d ** 4, -1) / m2 ** 2 - 3

def onoff(f, nm, S, full=False):
    """aggregate a per-panel statistic S (b,6) into ON-vs-OFF features."""
    on, off = S[:, ON], S[:, OFF]
    f[nm + '_d'] = on.mean(1) - off.mean(1); f[nm + '_dmin'] = on.min(1) - off.max(1)
    if full:
        for p in range(6): f[f'{nm}_p{p}'] = S[:, p] - off.mean(1)
        adj = np.stack([S[:, a] - S[:, c] for a, c in [(0, 1), (2, 1), (2, 3), (4, 3), (4, 5)]], 1)
        f[nm + '_adjmin'] = adj.min(1); f[nm + '_onrank'] = (on[:, :, None] > off[:, None, :]).sum((1, 2)).astype(np.float32)

def chunk(x):
    b = len(x); x = x.astype(np.float32); miss = x == 0; xn = np.where(miss, np.nan, x)
    bp = nanq(xn.transpose(0, 3, 1, 2).reshape(b, 128, 96), [.5])[..., 0]                    # (b,128)
    r = xn - bp[:, None, None, :]
    rp = r.reshape(b, 6, -1)
    s0 = 1.4826 * nanq(np.abs(rp.reshape(b, -1) - nanq(rp.reshape(b, -1), [.5])), [.5])[:, 0]  # robust noise (b,)
    lr = 2 ** (r / 24); P = 2 ** (xn / 24 - 4)
    f = {}
    # --- lvl: level in log and linear units, quantiles
    mean_r, std_r, sk, ku = moments(rp)
    qv = nanq(rp, QS)                                                                         # (b,6,7)
    lin = nmean(lr.reshape(b, 6, -1), -1); Pm = nmean(P.reshape(b, 6, -1), -1)
    onoff(f, 'rs_lvl_mean', mean_r, full=True); onoff(f, 'rs_lvl_lin', lin, full=True)
    f['rs_lvl_absP_log2ratio'] = np.log2(Pm[:, ON].mean(1) / Pm[:, OFF].mean(1))
    f['rs_lvl_lin_log2ratio'] = np.log2(lin[:, ON].mean(1) / lin[:, OFF].mean(1))
    f['rs_lvl_lin_onmin_log2offmax'] = np.log2(lin[:, ON].min(1) / lin[:, OFF].max(1))
    for k, qq in enumerate(QS): onoff(f, f'rs_lvl_q{int(qq * 100):02d}', qv[..., k])
    onoff(f, 'rs_lvl_iqr', qv[..., 4] - qv[..., 2]); onoff(f, 'rs_lvl_tail', qv[..., 6] - qv[..., 3])
    # --- var: spread, modulation indices (scintillation = multiplicative fluctuation in time and frequency)
    mad = 1.4826 * nanq(np.abs(rp - qv[..., 3:4]), [.5])[..., 0]
    lin_std = np.nanstd(lr.reshape(b, 6, -1), -1)
    ts = nmean(lr, 3); tf = nmean(lr, 2)                                                     # (b,6,16), (b,6,128)
    m_t = np.nanstd(ts, -1) / nmean(ts, -1); m_f = np.nanstd(tf, -1) / nmean(tf, -1); m_tf = lin_std / lin
    onoff(f, 'rs_var_std', std_r, full=True); onoff(f, 'rs_var_mad', mad); onoff(f, 'rs_var_linstd', lin_std)
    onoff(f, 'rs_var_mt', m_t, full=True); onoff(f, 'rs_var_mf', m_f, full=True); onoff(f, 'rs_var_mtf', m_tf)
    f['rs_var_std_log2ratio'] = np.log2(std_r[:, ON].mean(1) / std_r[:, OFF].mean(1)); f['rs_var_s0'] = s0
    ra = r.reshape(b, -1); lo_hi = nanq(ra, [.02, .98])                                         # 2-98% trimmed std (MAD is byte-quantised)
    f['rs_var_s0t'] = np.nanstd(np.where((ra > lo_hi[:, :1]) & (ra < lo_hi[:, 1:]), ra, np.nan), 1) / 0.8905
    f['rs_var_mad_over_std'] = (mad / std_r).mean(1)
    # --- shape
    onoff(f, 'rs_shape_skew', sk, full=True); onoff(f, 'rs_shape_kurt', ku, full=True)
    _, _, csk, cku = moments(r.reshape(b, -1)); f['rs_shape_cad_skew'] = csk; f['rs_shape_cad_kurt'] = cku
    # --- frac: dropped / saturated / clipped pixels
    drow = miss.all(3)                                                                        # (b,6,16) dropped rows
    zpix = (miss & ~drow[..., None]).reshape(b, 6, -1).mean(-1)
    sat = (x == 255).reshape(b, 6, -1).mean(-1); low = (x == 1).reshape(b, 6, -1).mean(-1)
    onoff(f, 'rs_frac_zeropix', zpix); onoff(f, 'rs_frac_sat', sat); onoff(f, 'rs_frac_low', low)
    f['rs_frac_sat_all'] = sat.mean(1); f['rs_frac_low_all'] = low.mean(1); f['rs_frac_zeropix_all'] = zpix.mean(1)
    # --- rows: glitch rows, dropped rows, row-level variability and trend
    rowm = nmean(r, 3)                                                                        # (b,6,16)
    rmed = nanq(rowm, [.5])
    rex = (rowm - rmed) / (s0[:, None, None] / np.sqrt(128) + 1e-6)
    gl = np.nan_to_num(rex) > 5
    f['rs_rows_glitch_on'] = gl[:, ON].sum((1, 2)).astype(np.float32); f['rs_rows_glitch_off'] = gl[:, OFF].sum((1, 2)).astype(np.float32)
    f['rs_rows_glitch_npanel'] = gl.any(2).sum(1).astype(np.float32)
    f['rs_rows_drop_on'] = drow[:, ON].sum((1, 2)).astype(np.float32); f['rs_rows_drop_off'] = drow[:, OFF].sum((1, 2)).astype(np.float32)
    rexm = np.nanmax(np.where(np.isnan(rex), -np.inf, rex), 2); rexm[~np.isfinite(rexm)] = np.nan
    onoff(f, 'rs_rows_maxex', rexm)
    rown = np.where(gl, np.nan, rowm)                                                         # glitch-free row means
    onoff(f, 'rs_rows_std', np.nanstd(rown, 2), full=True)
    t = np.arange(16) - 7.5; tv = np.where(np.isnan(rown), 0, t)
    rc = np.nan_to_num(rown - nmean(rown, 2)[..., None])
    slope = (rc * tv).sum(2) / ((tv ** 2).sum(2) + 1e-6)
    onoff(f, 'rs_rows_slope', slope, full=True)
    f['rs_rows_slope_all'] = slope.mean(1)
    # --- spec: ON-OFF residual spectrum, edges, inter-panel spectral correlation, bandpass shape
    chm = nmean(r, 2)                                                                         # (b,6,128)
    d = nmean(chm[:, ON], 1) - nmean(chm[:, OFF], 1)
    dn = d / (s0[:, None] / np.sqrt(48) + 1e-6)
    f['rs_spec_d_std'] = np.nanstd(dn, 1); f['rs_spec_d_max'] = np.nanmax(dn, 1); f['rs_spec_d_min'] = np.nanmin(dn, 1)
    f['rs_spec_d_n4'] = (np.abs(np.nan_to_num(dn)) > 4).sum(1).astype(np.float32); f['rs_spec_d_npos4'] = (np.nan_to_num(dn) > 4).sum(1).astype(np.float32)
    f['rs_spec_d_kurt'] = moments(dn)[3]
    f['rs_spec_edge_lo'] = nmean(dn[:, :8], 1); f['rs_spec_edge_hi'] = nmean(dn[:, -8:], 1); f['rs_spec_ctr'] = nmean(dn[:, 56:72], 1)
    f['rs_spec_edge_minus_ctr'] = (f['rs_spec_edge_lo'] + f['rs_spec_edge_hi']) / 2 - f['rs_spec_ctr']
    c = np.nan_to_num(chm - nmean(chm, 2)[..., None]); c = c / (np.sqrt((c ** 2).sum(2, keepdims=True)) + 1e-9)
    cc = np.einsum('bpc,bqc->bpq', c, c)
    f['rs_spec_corr_onon'] = (cc[:, 0, 2] + cc[:, 0, 4] + cc[:, 2, 4]) / 3
    f['rs_spec_corr_offoff'] = (cc[:, 1, 3] + cc[:, 1, 5] + cc[:, 3, 5]) / 3
    f['rs_spec_corr_onoff'] = cc[:, ON][:, :, OFF].mean((1, 2))
    f['rs_spec_corr_on_minus_onoff'] = f['rs_spec_corr_onon'] - f['rs_spec_corr_onoff']
    f['rs_spec_corr_adj'] = np.mean([cc[:, p, p + 1] for p in range(5)], 0)
    bpc = bp - np.median(bp, 1, keepdims=True)
    f['rs_spec_bp_rolloff'] = (bpc[:, :8].mean(1) + bpc[:, -8:].mean(1)) / 2 - bpc[:, 56:72].mean(1)
    f['rs_spec_bp_std'] = bpc[:, 8:-8].std(1); f['rs_spec_bp_tilt'] = bpc[:, 64:120].mean(1) - bpc[:, 8:64].mean(1)
    f['rs_spec_bp_level'] = np.median(bp, 1)
    # --- band: 8 sub-bands x 16 channels, linear ON/OFF ratio and log-unit difference
    sb = nmean(lr.reshape(b, 6, 16, 8, 16), (2, 4))                                          # (b,6,8)
    lsb = np.log2(nmean(sb[:, ON], 1) / nmean(sb[:, OFF], 1))
    for k in range(8): f[f'rs_band_lin{k}'] = lsb[:, k]
    f['rs_band_lin_std'] = np.nanstd(lsb, 1); f['rs_band_lin_max'] = np.nanmax(lsb, 1); f['rs_band_lin_min'] = np.nanmin(lsb, 1)
    f['rs_band_lin_npos'] = (np.nan_to_num(lsb) > 0).sum(1).astype(np.float32)
    sbm = np.log2(sb[:, ON] / nmean(sb[:, OFF], 1)[:, None])                                  # (b,3,8) each ON panel vs OFF
    f['rs_band_onpanel_min_max'] = np.nanmax(np.nanmin(sbm, 1), 1)                              # sub-band bright in ALL ON panels
    f['rs_band_onpanel_std'] = np.nanstd(sbm.reshape(b, -1), 1)
    # --- tblk: 4 time blocks x 4 rows, linear ON/OFF
    tb = nmean(lr.reshape(b, 6, 4, 4 * 128), 3)                                               # (b,6,4)
    ltb = np.log2(tb[:, ON] / nmean(tb[:, OFF], (1, 2))[:, None, None])                       # (b,3,4)
    for k in range(4): f[f'rs_tblk_lin{k}'] = nmean(ltb[:, :, k], 1)
    f['rs_tblk_lin_std'] = np.nanstd(ltb.reshape(b, -1), 1); f['rs_tblk_lin_min'] = np.nanmin(ltb.reshape(b, -1), 1)
    f['rs_tblk_lin_max'] = np.nanmax(ltb.reshape(b, -1), 1)
    toff = np.log2(tb[:, OFF] / nmean(tb[:, OFF], (1, 2))[:, None, None])
    f['rs_tblk_off_std'] = np.nanstd(toff.reshape(b, -1), 1)
    # --- acf: lag-1 autocorrelation in time and frequency (dropped pixels -> 0 after centring)
    rc = np.nan_to_num(r - nmean(rp, -1)[:, :, None, None]); den = (rc ** 2).sum((2, 3)) + 1e-6
    onoff(f, 'rs_acf_t1', (rc[:, :, 1:] * rc[:, :, :-1]).sum((2, 3)) / den, full=True)
    onoff(f, 'rs_acf_f1', (rc[..., 1:] * rc[..., :-1]).sum((2, 3)) / den, full=True)
    onoff(f, 'rs_acf_f4', (rc[..., 4:] * rc[..., :-4]).sum((2, 3)) / den)
    # --- byte: quantisation pattern of the uint8 values
    xi = x.reshape(b, -1).astype(np.int64)
    h = np.bincount((np.arange(b)[:, None] * 256 + xi).ravel(), minlength=b * 256).reshape(b, 256)[:, 1:].astype(np.float64)
    hs = h.sum(1, keepdims=True) + 1e-9; p = h / hs
    f['rs_byte_even'] = p[:, 1::2].sum(1)                                                      # byte values 2,4,... (index 1 = value 2)
    f['rs_byte_nuniq'] = (h > 0).sum(1).astype(np.float32); f['rs_byte_top'] = p.max(1)
    with np.errstate(divide='ignore', invalid='ignore'): f['rs_byte_entropy'] = -np.nansum(np.where(p > 0, p * np.log(p), 0), 1)
    f['rs_byte_comb'] = np.abs(h[:, 1:-1] - (h[:, :-2] + h[:, 2:]) / 2).sum(1) / hs[:, 0]
    f['rs_byte_mod3'] = p[:, 2::3].sum(1)                                                      # values 3,6,9,...
    return f

def compute(path, bs=500):
    X = np.load(path, mmap_mode='r'); parts = []
    for s in range(0, len(X), bs):
        f = chunk(np.asarray(X[s:s + bs])); parts.append(f)
    names = list(parts[0]); A = np.concatenate([np.stack([np.asarray(p[n], np.float32) for n in names], 1) for p in parts])
    A[~np.isfinite(A)] = np.nan
    return A, names

if __name__ == '__main__':
    out = Path(sys.argv[1]); res = {}
    for a in sys.argv[2:]:
        w, p = a.split('=', 1); res['X_' + w], names = compute(p); print(w, res['X_' + w].shape, flush=True)
    np.savez_compressed(out, names=np.array(names), **res)
