import sys; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
tr, te = meta(); y = y_train()
def peaks(nm, md):
    z = np.load(D + rf'\z_{nm}.npy', mmap_mode='r'); out = []
    for s in range(0, len(z), 2000):
        x = np.asarray(z[s:s + 2000], dtype=np.float32)
        spec = x.mean((1, 2)) * np.sqrt(96)                       # zero-drift line SNR, all panels
        spec_any = x.mean(2).max(1) * 4                            # strongest single-panel zero-drift
        out.append(np.c_[spec.argmax(1), spec.max(1), spec_any.argmax(1), spec_any.max(1)])
    o = np.concatenate(out); f = md.center_freq_mhz.values * 1e6 + (o[:, 0] - 63.5) * md.channel_width_hz.values
    fa = md.center_freq_mhz.values * 1e6 + (o[:, 2] - 63.5) * md.channel_width_hz.values
    return pd.DataFrame(dict(ch=o[:, 0], snr=o[:, 1], f=f, ch1=o[:, 2], snr1=o[:, 3], f1=fa))
P = peaks('train', tr); Q = peaks('test', te)
for nm, df, md in [('train', P, tr), ('test', Q, te)]:
    cf = md.center_freq_mhz.values
    print(nm, 'centre freq: unique', len(np.unique(np.round(cf, 6))), 'of', len(cf), '| unique at 1 kHz', len(np.unique(np.round(cf * 1e3))), '| at 100 kHz', len(np.unique(np.round(cf * 10))))
    strong = df.snr1 > 8
    fr = np.round(df.f1[strong] / 200.0)                          # 200 Hz bins of absolute line freq
    vc = fr.value_counts(); print('  strong lines', strong.sum(), '| bins', len(vc), '| lines in bins shared by >=2 cadences', int(vc[vc >= 2].sum()), '| top bin counts', vc.head(8).tolist())
    fr_k = np.round(df.f1[strong] / 1e5)                          # 100 kHz bins
    vk = fr_k.value_counts(); print('  100kHz bins: n', len(vk), 'top counts', vk.head(8).tolist(), 'expected if uniform ~', round(strong.sum() / (np.ptp(cf) * 10 + 1), 2))
# label vs being in a shared bin (train)
strong = P.snr1 > 8; key = np.round(P.f1 / 200.0); cnt = key[strong].value_counts()
shared = strong & key.map(cnt).fillna(0).ge(2).values
print('train: frac shared-bin among strong-line cadences by label:', {LABELS[c]: round(shared[(y == c) & strong].mean(), 3) for c in range(9)})
kk = np.round(P.f1 / 1e5); ck = kk[strong].value_counts(); dense = strong & kk.map(ck).fillna(0).ge(3).values
print('train: frac in dense 100kHz bin (>=3) by label:', {LABELS[c]: round(dense[(y == c) & strong].mean(), 3) for c in range(9)})
print('train label mix of strong-line cadences:', {LABELS[c]: int(((y == c) & strong).sum()) for c in range(9)})
