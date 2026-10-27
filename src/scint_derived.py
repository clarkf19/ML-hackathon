"""E1: scintillation statistics DERIVED from bundle columns (no raw data needed, so they exist for clean / test / deg2 / deg4
alike). GBDT splits on one column at a time, so it cannot form the ratios / rank counts / pooled-variance tests below.
  level   : ON-vs-OFF panel-level tests that are robust to per-panel gain jitter (t-statistic with pooled within-group
            variance, Mann-Whitney pair count, min-ON vs OFF z), sub-band consistency (median / trimmed mean / count of
            brightened sub-bands, max-minus-median = narrowband contamination of the level).
  texture : log ON/OFF block-variance ratio per scale, per-panel ON>OFF pair counts, joint level+texture panel counts.
-> results/feats_scint_derived.npz with X_clean, X_test, X_deg2, X_deg4 (+ cols)"""
import numpy as np
from pathlib import Path
R = Path(__file__).resolve().parent.parent
m = np.load(R / 'bundle/meta.npz', allow_pickle=True); ci = {c: i for i, c in enumerate(m['cols'])}
SC = ['2x4', '2x8', '4x8', '4x16', '8x16', '8x32', '16x32', '16x64']

def pairs_gt(on, off):                                   # Mann-Whitney count of (ON_i > OFF_j), 0..9
    return (on[:, :, None] > off[:, None, :]).sum((1, 2)).astype(np.float32)

def tstat(on, off, floor):
    pooled = (((on - on.mean(1, keepdims=True)) ** 2).sum(1) + ((off - off.mean(1, keepdims=True)) ** 2).sum(1)) / 4
    return (on.mean(1) - off.mean(1)) / np.sqrt(pooled * 2 / 3 + floor)

def derive(X):
    g = lambda c: X[:, ci[c]].astype(np.float64); f = {}
    lv = np.stack([g(f'lvl_lv_p{p}') for p in range(6)], 1); on, off = lv[:, 0::2], lv[:, 1::2]
    f['sd_lv_t'] = tstat(on, off, 0.25)
    f['sd_lv_u'] = pairs_gt(on, off)
    f['sd_lv_med_d'] = np.median(on, 1) - np.median(off, 1)
    f['sd_lv_on2nd_offmax'] = np.sort(on, 1)[:, 1] - off.max(1)
    f['sd_lv_onmin_z'] = (on.min(1) - off.mean(1)) / (off.std(1) + 0.5)
    f['sd_lv_ratio_spread'] = g('lvl_lv_on_off') / (g('lvl_lv_off_spread') + 1.0)
    f['sd_lv_on_cv'] = on.std(1) / (np.abs(on.mean(1)) + 1.0)                   # star: same brightening in all 3 ON
    bd = np.stack([g(f'lvl_lv_band{k}') for k in range(8)], 1); sb = np.sort(bd, 1)
    f['sd_band_med'] = np.median(bd, 1); f['sd_band_trim'] = sb[:, 1:-1].mean(1)
    f['sd_band_t'] = bd.mean(1) / (bd.std(1) / np.sqrt(8) + 0.3)
    for th in (1, 2, 3): f[f'sd_band_n{th}'] = (bd > th).sum(1).astype(np.float32)
    f['sd_band_max_minus_med'] = bd.max(1) - np.median(bd, 1)
    f['sd_band_min_over_med'] = bd.min(1) / (np.abs(np.median(bd, 1)) + 1.0)
    f['sd_band_lo_hi'] = bd[:, :4].mean(1) - bd[:, 4:].mean(1)                 # frequency tilt of the brightening
    f['sd_tblk_ratio'] = np.log((g('lvl_lv_tblk_std_on') + 0.3) / (g('lvl_lv_tblk_std_off') + 0.3))
    # texture
    best = np.full(len(X), -9.0); arg = np.zeros(len(X)); usum = 0
    for k, s in enumerate(SC):
        v = np.stack([g(f'tex_v{s}_p{p}') for p in range(6)], 1); von, voff = v[:, 0::2], v[:, 1::2]
        lr = np.log((g(f'tex_v{s}_on') + 0.05) / (g(f'tex_v{s}_off') + 0.05)); f[f'sd_tex{s}_lr'] = lr
        u = pairs_gt(von, voff); f[f'sd_tex{s}_u'] = u; usum = usum + u
        f[f'sd_tex{s}_t'] = tstat(np.log(von + 0.05), np.log(voff + 0.05), 0.01)
        upd = lr > best; best[upd] = lr[upd]; arg[upd] = k
    f['sd_tex_lr_best'] = best; f['sd_tex_lr_argscale'] = arg; f['sd_tex_usum'] = usum
    ts = np.stack([g(f'phys_tex_std_p{p}') for p in range(6)], 1)
    f['sd_texstd_t'] = tstat(np.log(ts[:, 0::2] + 0.05), np.log(ts[:, 1::2] + 0.05), 0.01); f['sd_texstd_u'] = pairs_gt(ts[:, 0::2], ts[:, 1::2])
    # joint: per ON panel, is it both brighter AND more textured than every OFF panel?
    v = np.stack([g(f'tex_v4x16_p{p}') for p in range(6)], 1)
    both = (on > off.max(1, keepdims=True)) & (v[:, 0::2] > v[:, 1::2].max(1, keepdims=True))
    f['sd_joint_n'] = both.sum(1).astype(np.float32)
    f['sd_joint_score'] = f['sd_lv_t'] + f['sd_tex4x16_t']
    return {k: np.asarray(a, np.float32) for k, a in f.items()}

if __name__ == '__main__':
    out = {}
    for w in ['clean', 'test', 'deg2', 'deg4']:
        d = derive(np.load(R / f'bundle/feats_{w}.npz')['X']); out['X_' + w] = np.stack(list(d.values()), 1); cols = list(d)
    (R / 'results').mkdir(exist_ok=True)
    np.savez_compressed(R / 'results/feats_scint_derived.npz', cols=np.array(cols), **out)
    print('derived', out['X_clean'].shape, 'nan', [int(np.isnan(a).sum()) for a in out.values()])
